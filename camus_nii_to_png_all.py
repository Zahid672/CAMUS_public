# camus_nii_to_png_all.py
import os
import sys
import csv
import argparse
from pathlib import Path
import numpy as np

try:
    import nibabel as nib
except ImportError:
    print("Missing dependency: nibabel. Install with: pip install nibabel", file=sys.stderr)
    sys.exit(1)

try:
    import imageio.v2 as imageio
except ImportError:
    print("Missing dependency: imageio. Install with: pip install imageio", file=sys.stderr)
    sys.exit(1)


# -------------------- helpers --------------------
def is_mask_like(arr: np.ndarray, max_unique: int = 16) -> bool:
    """Heuristic: mask volumes are small-set integers."""
    a = arr
    if a.size > 1_000_000:
        rng = np.random.default_rng(0)
        idx = rng.choice(a.size, 1_000_000, replace=False)
        a = a.reshape(-1)[idx]
    uniques = np.unique(a.astype(np.int64))
    return len(uniques) <= max_unique and np.allclose(a, a.astype(np.int64))

def minmax_scale_to_uint8(x: np.ndarray, p_lo: float = 1.0, p_hi: float = 99.0) -> np.ndarray:
    """Percentile clamp + min-max scale -> uint8."""
    x = x.astype(np.float32)
    lo = np.percentile(x, p_lo)
    hi = np.percentile(x, p_hi)
    if hi <= lo:
        lo, hi = x.min(), x.max()
        if hi <= lo:
            return np.zeros_like(x, dtype=np.uint8)
    x = np.clip(x, lo, hi)
    x = (x - lo) / (hi - lo + 1e-8)
    return (x * 255.0).round().astype(np.uint8)

def ensure_dir(p: Path) -> Path:
    p.mkdir(parents=True, exist_ok=True)
    return p

def colorize_mask_uint8(mask: np.ndarray, palette=None) -> np.ndarray:
    """(H,W)->(H,W,3) uint8"""
    if palette is None:
        palette = {0:(0,0,0), 1:(255,0,0), 2:(0,255,0), 3:(0,0,255)}
    h, w = mask.shape
    out = np.zeros((h, w, 3), dtype=np.uint8)
    for k, rgb in palette.items():
        out[mask == k] = rgb
    return out

def save_png(img: np.ndarray, path: Path):
    if img.ndim == 2 or (img.ndim == 3 and img.shape[2] in (1,3,4)):
        imageio.imwrite(path.as_posix(), img)
    else:
        raise ValueError(f"Bad image shape {img.shape} for PNG.")

def iter_slices(vol: np.ndarray, axis: int):
    n = vol.shape[axis]
    for i in range(n):
        yield i, np.take(vol, indices=i, axis=axis)


# -------------------- core conversion --------------------
def convert_file(
    in_path: Path,
    out_root: Path,
    axis: int = -1,
    all_slices: bool = True,            # default changed → export ALL frames by default
    pct_lo: float = 1.0,
    pct_hi: float = 99.0,
    colorize_masks: bool = True,
    keep_structure: bool = True,
    manifest_writer=None,
):
    """
    Convert a single .nii/.nii.gz file to PNG(s).
    - If 2D → one PNG.
    - If 3D → all frames by default (_frameNNN suffix).
    Heuristically detects masks vs images; colorizes masks.
    """
    try:
        nii = nib.load(in_path.as_posix())
        vol = nii.get_fdata()
    except Exception as e:
        print(f"[WARN] Failed to load {in_path}: {e}", file=sys.stderr)
        return

    vol = np.nan_to_num(vol, nan=0.0, posinf=0.0, neginf=0.0)
    vol = np.squeeze(vol)

    rel_parent = in_path.parent.name if keep_structure else ""
    subdir = out_root / rel_parent
    ensure_dir(subdir)

    is_mask = is_mask_like(vol)

    if vol.ndim == 2:
        slices = [(0, vol)]
    elif vol.ndim == 3:
        slices = list(iter_slices(vol, axis=axis)) if all_slices else [(vol.shape[axis]//2, np.take(vol, vol.shape[axis]//2, axis=axis))]
    else:
        print(f"[WARN] Skipping {in_path}, unsupported dims {vol.shape}", file=sys.stderr)
        return

    base = in_path.name.replace(".nii.gz", "").replace(".nii", "")

    for idx, sl in slices:
        if not is_mask:
            img = minmax_scale_to_uint8(sl, p_lo=pct_lo, p_hi=pct_hi)
            out_name = f"{base}_frame{idx:03d}.png" if vol.ndim == 3 else f"{base}.png"
            out_path = subdir / out_name
            save_png(img, out_path)
            if manifest_writer:
                manifest_writer.writerow([in_path.as_posix(), out_path.as_posix(), idx, "image", img.shape[0], img.shape[1]])
        else:
            lab = sl.astype(np.int32)
            lab = np.clip(lab, 0, 255).astype(np.uint8)
            out_name = f"{base}_frame{idx:03d}_mask.png" if vol.ndim == 3 else f"{base}_mask.png"
            out_path = subdir / out_name
            if colorize_masks:
                rgb = colorize_mask_uint8(lab)
                save_png(rgb, out_path)
                typ = "mask_rgb"
                h, w = rgb.shape[:2]
            else:
                save_png(lab, out_path)
                typ = "mask_label"
                h, w = lab.shape
            if manifest_writer:
                manifest_writer.writerow([in_path.as_posix(), out_path.as_posix(), idx, typ, h, w])


def main():
    ap = argparse.ArgumentParser("CAMUS NIfTI → PNG (ALL frames by default)")
    ap.add_argument("--input", type=str, default="database_nifti", help="Input root with .nii/.nii.gz")
    ap.add_argument("--outdir", type=str, default="camus_png_all", help="Output folder for PNG files")
    ap.add_argument("--glob", type=str, default="**/*.nii*", help="Glob under input (recursive)")
    ap.add_argument("--axis", type=int, default=-1, help="Axis to slice 3D volumes (time is usually last)")
    ap.add_argument("--all_slices", action="store_true", help="(Optional) explicit all-slices; default already True")
    ap.add_argument("--pct_lo", type=float, default=1.0, help="Lower percentile for intensity clamp")
    ap.add_argument("--pct_hi", type=float, default=99.0, help="Upper percentile for intensity clamp")
    ap.add_argument("--no_colorize_masks", action="store_true", help="Write masks as label PNG (not RGB)")
    ap.add_argument("--flat", action="store_true", help="Don’t mirror input subfolders; dump into outdir flat")
    ap.add_argument("--manifest", type=str, default="manifest.csv", help="CSV manifest filename (written in outdir)")
    args = ap.parse_args()

    in_root = Path(args.input).resolve()
    out_root = ensure_dir(Path(args.outdir).resolve())
    colorize = not args.no_colorize_masks
    keep_structure = not args.flat

    nii_files = sorted(in_root.glob(args.glob))
    if not nii_files:
        print(f"[ERROR] No NIfTI files found under {in_root} with pattern '{args.glob}'.", file=sys.stderr)
        sys.exit(2)

    manifest_path = out_root / args.manifest
    with open(manifest_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["src_path", "png_path", "slice_index", "type", "H", "W"])
        for p in nii_files:
            try:
                convert_file(
                    in_path=p,
                    out_root=out_root,
                    axis=args.axis,
                    all_slices=True if not args.all_slices else True,  # default True
                    pct_lo=args.pct_lo,
                    pct_hi=args.pct_hi,
                    colorize_masks=colorize,
                    keep_structure=keep_structure,
                    manifest_writer=w,
                )
            except Exception as e:
                print(f"[WARN] Failed on {p}: {e}", file=sys.stderr)

    print(f"\n✅ Done. PNGs under: {out_root}")
    print(f"   Manifest: {manifest_path}")

if __name__ == "__main__":
    main()
