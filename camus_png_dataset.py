# camus_png_dataset.py
import csv
from pathlib import Path
import re
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset

# -------- RGB -> label mapping (for colorized masks) --------
RGB2LBL = {
    (0,   0,   0): 0,  # background
    (255, 0,   0): 1,  # LV endocardium
    (0, 255,   0): 2,  # LV myocardium
    (0,   0, 255): 3,  # Left atrium
}

def rgb_to_label(rgb: np.ndarray) -> np.ndarray:
    H, W, _ = rgb.shape
    out = np.zeros((H, W), dtype=np.uint8)
    flat = rgb.reshape(-1, 3)
    out_flat = out.reshape(-1)
    for c, l in RGB2LBL.items():
        match = np.all(flat == c, axis=1)
        out_flat[match] = l
    return out

def normalize_csv_path(p: str, root: Path) -> Path:
    pp = Path(p)
    return pp if pp.is_absolute() else (root / pp).resolve()

# --- suffix handling: strip common mask suffix tokens from filename stems ---
# Handles cases like:
#   foo_mask.png, foo_gt_mask.png, foo_gt.png, foo_seg.png, foo_label.png, foo-label.png, foo_manual.png, etc.
_MASK_SUFFIX_PATTERNS = [
    r"_gt_mask$", r"_mask$", r"_gt$", r"_seg$", r"_label$", r"_labels$",
    r"-mask$", r"-label$", r"_manual$", r"_annotation$", r"_pred$", r"-pred$"
]
_MASK_SUFFIX_RE = re.compile("(" + "|".join(_MASK_SUFFIX_PATTERNS) + ")", re.IGNORECASE)

def normalize_stem(stem: str) -> str:
    """Recursively strip known mask suffixes to get a common pairing key."""
    prev = None
    s = stem
    # strip extension-like suffixes repeatedly (e.g., *_gt_mask -> *_gt -> base)
    while prev != s:
        prev = s
        s = _MASK_SUFFIX_RE.sub("", s)
    return s

def looks_like_mask_name(stem: str) -> bool:
    """Heuristic: filename suggests it's a mask."""
    return bool(_MASK_SUFFIX_RE.search(stem))

# -------- Audit helper --------
def audit_manifest(manifest_path: Path, root: Path):
    if not manifest_path.exists():
        print(f"[AUDIT] manifest not found: {manifest_path}")
        return
    n_rows = n_img = n_mask_rgb = n_mask_label = 0
    sample_img = sample_mask = None
    with open(manifest_path, "r", newline="") as f:
        r = csv.DictReader(f)
        for row in r:
            n_rows += 1
            typ = (row.get("type", "") or "").lower()
            if typ == "image":
                n_img += 1
                if sample_img is None:
                    sample_img = row.get("png_path")
            elif typ == "mask_rgb":
                n_mask_rgb += 1
                if sample_mask is None:
                    sample_mask = row.get("png_path")
            elif typ == "mask_label":
                n_mask_label += 1
                if sample_mask is None:
                    sample_mask = row.get("png_path")
    print(f"[AUDIT] manifest: {manifest_path}")
    print(f"        rows: {n_rows}, images: {n_img}, mask_rgb: {n_mask_rgb}, mask_label: {n_mask_label}")
    if sample_img:
        print(f"        e.g. image: {normalize_csv_path(sample_img, root)}")
    if sample_mask:
        print(f"        e.g. mask : {normalize_csv_path(sample_mask, root)}")

class CAMUSPNGSliceDataset(Dataset):
    """
    Robust loader for PNG slices + masks created by camus_nii_to_png.py.

    Pairs are built in this order:
      1) From manifest.csv rows (type: image, mask_rgb/mask_label). Keys normalized via suffix stripping.
      2) Fallback: directory scan using name normalization (handles *_gt_mask.png etc).

    Tolerates:
      - absolute or relative manifest paths
      - presence/absence of 'type' column
      - RGB masks (colorized) and label masks
    """
    def __init__(self, root="camus_png", manifest="manifest.csv", img_size=None, verbose=True):
        self.root = Path(root).resolve()
        self.manifest_path = (self.root / manifest).resolve()
        self.img_size = img_size  # (W,H) or scalar or None
        self.verbose = verbose
        self.items = []

        if self.manifest_path.exists():
            self._load_from_manifest()
        else:
            if self.verbose:
                print(f"[INFO] manifest not found at {self.manifest_path}, falling back to directory scan.")
            self._fallback_scan()

        if self.verbose:
            if len(self.items) == 0:
                print(
                    f"[WARN] No (image, mask) pairs found.\n"
                    f"  - Checked manifest: {self.manifest_path}\n"
                    f"  - Root folder:      {self.root}\n"
                    f"Quick checks:\n"
                    f"  1) Do images end like *.png and masks like *_gt_mask.png / *_mask.png?\n"
                    f"  2) Do the png_path files in manifest actually exist on disk?\n"
                )
            else:
                print(f"[OK] Found {len(self.items)} pairs.")

    # ---------- Manifest pairing ----------
    def _load_from_manifest(self):
        audit_manifest(self.manifest_path, self.root)

        images = {}  # key -> Path
        masks  = {}  # key -> Path

        with open(self.manifest_path, "r", newline="") as f:
            r = csv.DictReader(f)
            for row in r:
                p = normalize_csv_path(row.get("png_path", ""), self.root)
                if not p.exists():
                    continue
                typ = (row.get("type", "") or "").lower()
                stem = p.stem

                # decide image vs mask (prefer explicit type, fallback to filename)
                is_mask = typ.startswith("mask") or looks_like_mask_name(stem)

                key = normalize_stem(stem)
                if is_mask:
                    masks[key] = p
                else:
                    images[key] = p

        keys = sorted(set(images.keys()) & set(masks.keys()))

        # If nothing paired, try direct filename-based guess for each image
        if len(keys) == 0 and images:
            for key, img_p in images.items():
                # try common mask names in same directory
                cand_stems = [
                    img_p.stem + "_gt_mask",
                    img_p.stem + "_mask",
                    img_p.stem + "_gt",
                    img_p.stem + "_seg",
                    img_p.stem + "_label",
                ]
                for cs in cand_stems:
                    c = img_p.with_name(cs + img_p.suffix)
                    if c.exists():
                        masks[normalize_stem(cs)] = c
            keys = sorted(set(images.keys()) & set(masks.keys()))

        self.items = [(images[k], masks[k]) for k in keys]

        if len(self.items) == 0 and self.verbose:
            print("[INFO] Manifest pairing failed; trying fallback directory scan.")
            self._fallback_scan()

    # ---------- Directory scan fallback ----------
    def _fallback_scan(self):
        images = {}
        masks  = {}
        for p in self.root.rglob("*.png"):
            stem_norm = normalize_stem(p.stem)
            if looks_like_mask_name(p.stem):
                masks[stem_norm] = p.resolve()
            else:
                images[stem_norm] = p.resolve()
        common = sorted(set(images.keys()) & set(masks.keys()))
        self.items = [(images[k], masks[k]) for k in common]

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        img_path, mask_path = self.items[i]
        img = np.array(Image.open(img_path).convert("L"))  # [H,W]
        m_img = Image.open(mask_path)
        if m_img.mode in ("RGB", "RGBA"):
            m = rgb_to_label(np.array(m_img.convert("RGB")))
        else:
            m = np.array(m_img)

        if self.img_size is not None:
            if isinstance(self.img_size, (tuple, list)):
                w, h = int(self.img_size[0]), int(self.img_size[1])
            else:
                w = h = int(self.img_size)
            img = np.array(Image.fromarray(img).resize((w, h), Image.BILINEAR))
            m   = np.array(Image.fromarray(m).resize((w, h), Image.NEAREST))

        img_t  = torch.from_numpy(img).float().unsqueeze(0) / 255.0  # [1,H,W]
        mask_t = torch.from_numpy(m).long()                          # [H,W]
        return img_t, mask_t

# ---------- Quick self-test ----------
if __name__ == "__main__":
    from torch.utils.data import DataLoader
    ds = CAMUSPNGSliceDataset(root="camus_png", manifest="manifest.csv", img_size=(256, 256), verbose=True)
    print("Dataset size:", len(ds))
    if len(ds) > 0:
        dl = DataLoader(ds, batch_size=4, shuffle=True)
        imgs, masks = next(iter(dl))
        print("Batch images:", imgs.shape, "Batch masks:", masks.shape)
