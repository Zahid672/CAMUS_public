# camus_png_dataset.py
import csv
from pathlib import Path
import re
import math
import numpy as np
from PIL import Image, ImageFilter
import torch
from torch.utils.data import Dataset

# --- optional OpenCV for CLAHE (quiet fallback if not installed) ---
try:
    import cv2
    _HAS_CV2 = True
except Exception:
    _HAS_CV2 = False

# --- torchvision FNs for geometry & photometric augs ---
try:
    import torchvision.transforms.functional as TF
    from torchvision.transforms import InterpolationMode
    _HAS_TV = True
except Exception:
    _HAS_TV = False
    InterpolationMode = None  # type: ignore

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
_MASK_SUFFIX_PATTERNS = [
    r"_gt_mask$", r"_mask$", r"_gt$", r"_seg$", r"_label$", r"_labels$",
    r"-mask$", r"-label$", r"_manual$", r"_annotation$", r"_pred$", r"-pred$"
]
_MASK_SUFFIX_RE = re.compile("(" + "|".join(_MASK_SUFFIX_PATTERNS) + ")", re.IGNORECASE)

def normalize_stem(stem: str) -> str:
    """Recursively strip known mask suffixes to get a common pairing key."""
    prev = None
    s = stem
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

# ===================== Preprocessing & Augmentations =====================
def _pil_to_numpy_gray(pil_img: Image.Image) -> np.ndarray:
    return np.array(pil_img)  # uint8 [H,W]

def _apply_clahe_uint8(gray_hw: np.ndarray) -> np.ndarray:
    """CLAHE via OpenCV if available; otherwise no-op."""
    if not _HAS_CV2:
        return gray_hw
    # OpenCV expects HxW uint8
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return clahe.apply(gray_hw)

def _zscore(gray_hw: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    m = gray_hw.mean()
    s = gray_hw.std()
    if s < eps:  # avoid div by tiny std
        return np.zeros_like(gray_hw, dtype=np.float32)
    return ((gray_hw - m) / s).astype(np.float32)

def _minmax01(gray_hw: np.ndarray) -> np.ndarray:
    mn, mx = float(gray_hw.min()), float(gray_hw.max())
    if mx <= mn:
        return np.zeros_like(gray_hw, dtype=np.float32)
    return ((gray_hw - mn) / (mx - mn)).astype(np.float32)

def _random_bool(p: float) -> bool:
    return np.random.rand() < p

def _rand_uniform(a, b):
    return float(np.random.uniform(a, b))

def _rand_int(a, b):
    return int(np.random.randint(a, b + 1))

def _apply_photometric_augs(img_pil: Image.Image,
                            gamma_p=0.2, brightness_contrast_p=0.2,
                            noise_p=0.15, blur_p=0.1) -> Image.Image:
    """Apply light photometric augs to PIL grayscale image."""
    # Gamma (power-law)
    if _random_bool(gamma_p):
        g = _rand_uniform(0.8, 1.2)  # mild
        # gamma correction on uint8 via LUT
        lut = np.array([((i / 255.0) ** g) * 255.0 for i in range(256)]).clip(0, 255).astype(np.uint8)
        img_pil = img_pil.point(lambda i: int(lut[i]))

    # Brightness/Contrast (via torchvision if present, else PIL point ops)
    if _random_bool(brightness_contrast_p):
        if _HAS_TV:
            b = _rand_uniform(0.9, 1.1)
            c = _rand_uniform(0.9, 1.1)
            img_pil = TF.adjust_brightness(img_pil, b)
            img_pil = TF.adjust_contrast(img_pil, c)
        else:
            # simple linear transform around mid-gray 127
            arr = np.array(img_pil).astype(np.float32)
            b = _rand_uniform(-15, 15)   # add -15..15
            c = _rand_uniform(0.9, 1.1)  # multiply
            arr = (arr - 127.0) * c + 127.0 + b
            img_pil = Image.fromarray(arr.clip(0, 255).astype(np.uint8))

    # Gaussian or speckle noise (light)
    if _random_bool(noise_p):
        arr = np.array(img_pil).astype(np.float32)
        if _random_bool(0.5):
            # Gaussian
            sigma = _rand_uniform(2.0, 8.0)
            arr = arr + np.random.normal(0.0, sigma, size=arr.shape).astype(np.float32)
        else:
            # Speckle: arr + arr * noise
            sigma = _rand_uniform(0.01, 0.03)
            arr = arr + arr * (np.random.normal(0.0, sigma, size=arr.shape)).astype(np.float32)
        img_pil = Image.fromarray(arr.clip(0, 255).astype(np.uint8))

    # Slight blur (optional)
    if _random_bool(blur_p):
        img_pil = img_pil.filter(ImageFilter.GaussianBlur(radius=_rand_uniform(0.5, 1.2)))

    return img_pil

def _apply_geometric_augs(img_pil: Image.Image,
                          mask_pil: Image.Image,
                          rotate_deg=8.0, scale_range=(0.95, 1.05),
                          translate_frac=0.05, hflip_p=0.5):
    """
    Consistent geometry on image/mask using torchvision (if available).
    - Small rotation, scale, and translation
    - Horizontal flip (vertical flip disabled by default for echo orientation)
    """
    if not _HAS_TV:
        # fallback: only horizontal flip using PIL
        if _random_bool(hflip_p):
            img_pil = img_pil.transpose(Image.FLIP_LEFT_RIGHT)
            mask_pil = mask_pil.transpose(Image.FLIP_LEFT_RIGHT)
        return img_pil, mask_pil

    # random params
    angle   = _rand_uniform(-rotate_deg, rotate_deg)
    scale   = _rand_uniform(scale_range[0], scale_range[1])
    tx_frac = _rand_uniform(-translate_frac, translate_frac)
    ty_frac = _rand_uniform(-translate_frac, translate_frac)

    w, h = img_pil.size
    max_tx = tx_frac * w
    max_ty = ty_frac * h
    translate = (int(max_tx), int(max_ty))  # pixels

    # affine
    img_pil  = TF.affine(img_pil, angle=angle, translate=translate, scale=scale, shear=[0.0, 0.0],
                         interpolation=InterpolationMode.BILINEAR, fill=0)
    mask_pil = TF.affine(mask_pil, angle=angle, translate=translate, scale=scale, shear=[0.0, 0.0],
                         interpolation=InterpolationMode.NEAREST,  fill=0)

    # horizontal flip
    if _random_bool(hflip_p):
        img_pil  = TF.hflip(img_pil)
        mask_pil = TF.hflip(mask_pil)

    return img_pil, mask_pil

# ===================== Dataset =====================
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

    NEW: preprocessing & augmentations
      - normalization: 'minmax' (0..1) or 'zscore' or 'none'
      - clahe: bool (requires OpenCV; otherwise skipped)
      - augment: apply safe geometry + photometric augs (train only)
    """
    def __init__(self,
                 root="camus_png",
                 manifest="manifest.csv",
                 img_size=None,
                 verbose=True,
                 # preprocessing
                 normalization: str = "minmax",   # 'minmax' | 'zscore' | 'none'
                 clahe: bool = False,
                 # augmentations
                 augment: bool = False,
                 rotate_deg: float = 8.0,
                 scale_range=(0.95, 1.05),
                 translate_frac: float = 0.05,
                 hflip_p: float = 0.5,
                 gamma_p: float = 0.2,
                 bc_p: float = 0.2,               # brightness/contrast prob
                 noise_p: float = 0.15,
                 blur_p: float = 0.1):
        self.root = Path(root).resolve()
        self.manifest_path = (self.root / manifest).resolve()
        self.img_size = img_size  # (W,H) or scalar or None
        self.verbose = verbose
        self.items = []

        # preprocessing/aug config
        self.normalization = normalization.lower()
        self.clahe = bool(clahe)
        self.augment = bool(augment)

        self.rotate_deg = float(rotate_deg)
        self.scale_range = tuple(scale_range)
        self.translate_frac = float(translate_frac)
        self.hflip_p = float(hflip_p)
        self.gamma_p = float(gamma_p)
        self.bc_p = float(bc_p)
        self.noise_p = float(noise_p)
        self.blur_p = float(blur_p)

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

                is_mask = typ.startswith("mask") or looks_like_mask_name(stem)
                key = normalize_stem(stem)
                if is_mask:
                    masks[key] = p
                else:
                    images[key] = p

        keys = sorted(set(images.keys()) & set(masks.keys()))

        if len(keys) == 0 and images:
            for key, img_p in images.items():
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

    def _resize_pair(self, img_pil: Image.Image, mask_pil: Image.Image):
        """Resize to img_size (W,H) or scalar if requested."""
        if self.img_size is None:
            return img_pil, mask_pil
        if isinstance(self.img_size, (tuple, list)):
            w, h = int(self.img_size[0]), int(self.img_size[1])
        else:
            w = h = int(self.img_size)
        img_pil  = img_pil.resize((w, h), Image.BILINEAR)
        mask_pil = mask_pil.resize((w, h), Image.NEAREST)
        return img_pil, mask_pil

    def __getitem__(self, i):
        img_path, mask_path = self.items[i]

        # load
        img_pil = Image.open(img_path).convert("L")   # PIL grayscale
        m_img   = Image.open(mask_path)
        if m_img.mode in ("RGB", "RGBA"):
            mask = rgb_to_label(np.array(m_img.convert("RGB")))
            mask_pil = Image.fromarray(mask, mode="L")
        else:
            mask_pil = m_img.convert("L")

        # ---- optional CLAHE (before other photometric ops) ----
        if self.clahe:
            gray = _pil_to_numpy_gray(img_pil)
            gray = _apply_clahe_uint8(gray)
            img_pil = Image.fromarray(gray, mode="L")

        # ---- augmentations (train only) ----
        if self.augment:
            # geometric (consistent)
            img_pil, mask_pil = _apply_geometric_augs(
                img_pil, mask_pil,
                rotate_deg=self.rotate_deg,
                scale_range=self.scale_range,
                translate_frac=self.translate_frac,
                hflip_p=self.hflip_p,
            )
            # photometric (image only)
            img_pil = _apply_photometric_augs(
                img_pil, gamma_p=self.gamma_p, brightness_contrast_p=self.bc_p,
                noise_p=self.noise_p, blur_p=self.blur_p
            )

        # ---- resize (after geometry) ----
        img_pil, mask_pil = self._resize_pair(img_pil, mask_pil)

        # ---- to numpy
        img = np.array(img_pil)  # uint8 [H,W]
        m   = np.array(mask_pil) # uint8 [H,W] labels expected 0..C-1

        # ---- normalization to float tensor ----
        if self.normalization == "zscore":
            img_f = _zscore(img)     # float32 ~ N(0,1)
        elif self.normalization == "minmax":
            img_f = _minmax01(img)   # float32 in [0,1]
        else:
            img_f = img.astype(np.float32) / 255.0

        img_t  = torch.from_numpy(img_f).float().unsqueeze(0)  # [1,H,W]
        mask_t = torch.from_numpy(m).long()                    # [H,W]
        return img_t, mask_t

# ---------- Quick self-test ----------
if __name__ == "__main__":
    from torch.utils.data import DataLoader
    # Example: enable preprocessing + augs for a quick sanity run
    ds = CAMUSPNGSliceDataset(
        root="camus_png", manifest="manifest.csv",
        img_size=(256, 256), verbose=True,
        normalization="minmax", clahe=False,
        augment=True  # turn OFF for validation
    )
    print("Dataset size:", len(ds))
    if len(ds) > 0:
        dl = DataLoader(ds, batch_size=4, shuffle=True)
        imgs, masks = next(iter(dl))
        print("Batch images:", imgs.shape, "Batch masks:", masks.shape)
