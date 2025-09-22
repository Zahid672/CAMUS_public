# save as: unet_dice_ce_camus_anylist.py
import os
import csv
import glob
import random
import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

# ======================= Config =======================
NUM_CLASSES   = 4
IN_CHANNELS   = 1          # 1 for grayscale echo, 3 for RGB
IMG_SIZE      = 256
VIEW          = '2CH'      # or '4CH'
BATCH_SIZE    = 2
EPOCHS        = 60
LR            = 1e-4
WEIGHT_DECAY  = 1e-4
STEP_SIZE     = 10
GAMMA         = 0.1
SAVE_N        = 8
NUM_WORKERS   = 4
SEED          = 42
DEVICE        = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Paths
DATA_DIR    = 'database_nifti'
SPLIT_DIR   = 'prepared_data'
TRAIN_LIST  = os.path.join(SPLIT_DIR, 'train_samples.npy')
TEST_LIST   = os.path.join(SPLIT_DIR, 'test_ED.npy')   # or 'test_ES.npy'

# Outputs
RESULTS_DIR = "UNet_Preprocessed_Data_results_Dice_CE"
METRICS_CSV = os.path.join(RESULTS_DIR, "UNet_preprocessed_metrics_Dice_CE.csv")
SAVE_ROOT   = "qualitative_Preprocessed_UNet_Dice_CE"

# Directory name candidates
CAND_IMAGE_DIRS = ["images", "image", "imgs", "img", "x", "inputs", "img2d", "image2d", VIEW]
CAND_MASK_DIRS  = ["masks", "mask", "labels", "label", "y", "gt", "gts", "seg", "segmentation", "annotations", f"{VIEW}_gt"]

# Supported file extensions (order matters for preference)
IMG_EXTS  = [".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".npy", ".nii.gz", ".nii", ".mhd"]
MASK_EXTS = [".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".npy", ".nii.gz", ".nii", ".mhd"]

# Filename templates for CAMUS-style assets (no extension; we’ll try IMG_EXTS/MASK_EXTS)
# Tokens available: {patient}  {phase}  {view}
FILENAME_TEMPLATES_IMG = [
    "{patient}_{view}_{phase}",
    "{patient}-{view}-{phase}",
    "{patient}_{phase}_{view}",
    "{patient}{view}{phase}",
    "{patient}_{phase}",           # fallback
]
FILENAME_TEMPLATES_MSK = [
    "{patient}_{view}_{phase}_gt",
    "{patient}_{view}_{phase}_mask",
    "{patient}-{view}-{phase}-gt",
    "{patient}_{phase}_{view}_gt",
    "{patient}_{phase}_gt",        # fallback
]

# ======================= Small utils =======================
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True

def ensure_dir(p): os.makedirs(p, exist_ok=True); return p

def _strip_double_ext(fn: str):
    if fn.lower().endswith(".nii.gz"):
        return fn[:-7], ".nii.gz"
    b, e = os.path.splitext(fn); return b, e

def _is_file_path(p: str) -> bool:
    return (isinstance(p, str) and (p.lower().endswith(".nii.gz") or os.path.splitext(p)[1] != ""))

def _to_abs(p: str, root: str) -> str:
    if os.path.isabs(p): return p
    return os.path.normpath(os.path.join(root, p))

# ======================= I/O helpers =======================
def _load_image_any(path: str, in_channels: int):
    if path.lower().endswith(".npy"):
        arr = np.load(path, allow_pickle=False)
        if arr.ndim == 2: img = arr[None, ...]
        elif arr.ndim == 3:
            if arr.shape[0] in (1,3) and arr.shape[1] > 8 and arr.shape[2] > 8: img = arr
            else: img = np.transpose(arr, (2,0,1))
        else: raise ValueError(f"Unexpected image shape for {path}: {arr.shape}")
        img = img.astype(np.float32)
        mx = img.max()
        if mx > 1.0:
            img = img / 255.0 if mx <= 255 else (img - img.min())/(mx - img.min() + 1e-8)
        return torch.from_numpy(img)
    else:
        pil = Image.open(path)
        if in_channels == 1:
            pil = pil.convert("L")
            arr = np.array(pil, dtype=np.float32)[None, ...]
        else:
            pil = pil.convert("RGB")
            arr = np.array(pil, dtype=np.float32); arr = np.transpose(arr, (2,0,1))
        if arr.max() > 1.0: arr = arr / 255.0
        return torch.from_numpy(arr)

def _load_mask_any(path: str):
    if path.lower().endswith(".npy"):
        arr = np.load(path, allow_pickle=False)
        if arr.ndim == 2: m = arr
        elif arr.ndim == 3:
            if arr.shape[0] > 1 and arr.max() <= 1.0: m = np.argmax(arr, axis=0)
            elif arr.shape[-1] > 1 and arr.max() <= 1.0: m = np.argmax(arr, axis=-1)
            else: m = arr[0] if arr.shape[0] in (1,3) else arr[...,0]
        else: raise ValueError(f"Unexpected mask shape for {path}: {arr.shape}")
        return torch.from_numpy(m.astype(np.int64))
    else:
        pil = Image.open(path).convert("L")
        arr = np.array(pil, dtype=np.int64)
        return torch.from_numpy(arr)

# ======================= CAMUS resolver =======================
def _normalize_phase(phase: str) -> str:
    p = phase.strip().upper()
    if p not in {"ED","ES"}:
        # tolerate typos like 'ed ', 'Es', etc.
        if "ED" in p: return "ED"
        if "ES" in p: return "ES"
    return p

def _make_basenames(patient: str, phase: str, view: str):
    toks = {"patient": patient, "phase": phase, "view": view}
    bases = []
    for tmpl in FILENAME_TEMPLATES_IMG:
        bases.append(tmpl.format(**toks))
    return list(dict.fromkeys(bases))  # unique, keep order

def _make_mask_basenames(patient: str, phase: str, view: str):
    toks = {"patient": patient, "phase": phase, "view": view}
    bases = []
    for tmpl in FILENAME_TEMPLATES_MSK:
        bases.append(tmpl.format(**toks))
    # also admit plain image basenames; we'll look for *_gt later by inference
    bases += _make_basenames(patient, phase, view)
    return list(dict.fromkeys(bases))

def _candidate_dirs(root: str, preferred: list):
    out = []
    for d in preferred:
        p = os.path.join(root, d)
        if os.path.isdir(p): out.append(p)
    if not out: out = [root]
    return out

def _search_by_basenames(root_dirs, basenames, exts):
    hits = []
    for rd in root_dirs:
        for base in basenames:
            for ext in exts:
                patt = os.path.join(rd, "**", base + ext)
                found = glob.glob(patt, recursive=True)
                if found: hits.extend(found)
    return hits

def _loose_search(root_dirs, patient, view, phase, exts, require_mask_token=False):
    # very loose: *patient*view*phase*.[ext] , optionally require 'mask|gt|seg' token
    hits = []
    mask_tokens = ["mask","gt","seg","label"]
    for rd in root_dirs:
        for ext in exts:
            patt = os.path.join(rd, "**", f"*{patient}*{view}*{phase}*{ext}")
            found = glob.glob(patt, recursive=True)
            if require_mask_token:
                found = [f for f in found if any(tok in os.path.basename(f).lower() for tok in mask_tokens)]
            hits.extend(found)
    return hits

def _infer_mask_from_image(img_path: str):
    base_dir = os.path.dirname(img_path)
    fname = os.path.basename(img_path)
    stem, ext = _strip_double_ext(fname)

    # Try suffix swaps
    swaps = [("_img","_mask"), ("-img","-mask"), (" image"," mask"), ("Image","Mask"), ("","_gt")]
    for a,b in swaps:
        if a in stem or a == "":
            cand = os.path.join(base_dir, (stem.replace(a,b) if a else stem + b) + ext)
            if os.path.isfile(cand): return cand

    # Try sibling mask directories
    parent = os.path.dirname(base_dir)
    for md in CAND_MASK_DIRS:
        cand = os.path.join(parent, md, stem + ext)
        if os.path.isfile(cand): return cand
        for me in MASK_EXTS:
            cand2 = os.path.join(parent, md, stem + me)
            if os.path.isfile(cand2): return cand2

    # Last resort: global search for same stem+mask hints
    mask_tokens = ["mask","gt","seg","label"]
    root = os.path.abspath(os.path.join(base_dir, ".."))
    for me in MASK_EXTS:
        patt = os.path.join(root, "**", f"{stem}*{me}")
        for f in glob.glob(patt, recursive=True):
            if any(tok in os.path.basename(f).lower() for tok in mask_tokens):
                return f
    return None

# ======================= Flexible dataset (now supports [patient, phase]) =======================
class CAMUSAnyListDataset(Dataset):
    IMG_KEYS = ['image','img','x','path','image_path','img_path','input','input_path','file','filepath']
    MSK_KEYS = ['mask','msk','label','y','mask_path','gt','gt_path','seg','seg_path','annotation','ann']
    ID_KEYS  = ['id','case','name','stem','basename','uid','file_id','sample','sample_id','patient']

    def __init__(self, data_root, list_file, view=VIEW, in_channels=IN_CHANNELS):
        super().__init__()
        self.data_root = data_root
        self.view = view
        self.in_channels = int(in_channels)

        if not (_is_file_path(list_file) and list_file.lower().endswith(".npy")):
            raise ValueError(f"Expected a .npy list file, got: {list_file}")
        if not os.path.isfile(list_file):
            raise FileNotFoundError(f"List file not found: {list_file}")

        # discover preferred dirs (if present)
        self.image_dirs_abs = _candidate_dirs(self.data_root, CAND_IMAGE_DIRS)
        self.mask_dirs_abs  = _candidate_dirs(self.data_root, CAND_MASK_DIRS)

        # cache for (patient,phase)->(img,msk)
        self._pp_cache = {}

        self.samples = self._parse_list(list_file)
        if len(self.samples) == 0:
            raise RuntimeError(f"No samples loaded from {list_file}")

    def __len__(self): return len(self.samples)

    def __getitem__(self, idx):
        img_path, msk_path = self.samples[idx]
        img = _load_image_any(img_path, self.in_channels).float()
        msk = _load_mask_any(msk_path).long()
        return img, msk

    # ---------- parsing ----------
    def _parse_list(self, list_file):
        arr = np.load(list_file, allow_pickle=True)
        samples, bad = [], None

        for it in arr:
            try:
                img_p, msk_p = self._resolve_item(it)
                if not os.path.isabs(img_p): img_p = _to_abs(img_p, self.data_root)
                if not os.path.isabs(msk_p): msk_p = _to_abs(msk_p, self.data_root)
                if not os.path.isfile(img_p) or not os.path.isfile(msk_p):
                    raise FileNotFoundError(f"Resolved paths do not exist:\n  img: {img_p}\n  msk: {msk_p}")
                samples.append((img_p, msk_p))
            except Exception as e:
                if bad is None:
                    bad = (repr(it), str(e))

        if bad is not None and len(samples) == 0:
            it_repr, err = bad
            raise ValueError(
                "Failed to parse any items from the list. First problematic entry:\n"
                f"  item: {it_repr}\n  error: {err}\n"
                "Expected one of:\n"
                "  1) dict with keys like {'image':..., 'mask':...}\n"
                "  2) tuple/list/ndarray like (patient_id, phase) or (img_path, mask_path)\n"
                "  3) single string path (mask inferred)\n"
                "  4) dict with only an ID (e.g., {'id':'patient0359', 'phase':'ED'})"
            )
        elif bad is not None:
            print("[CAMUSAnyListDataset] Warning: some items could not be parsed. "
                  f"Example bad item: {bad[0]}\nReason: {bad[1]}")

        return samples

    def _resolve_item(self, it):
        # dict with explicit paths
        if isinstance(it, dict):
            img_p = self._get_first_key(it, self.IMG_KEYS)
            msk_p = self._get_first_key(it, self.MSK_KEYS)
            if img_p is not None and msk_p is not None:
                return str(img_p), str(msk_p)
            # dict with id (+ phase)
            pid = self._get_first_key(it, self.ID_KEYS)
            phase = it.get('phase') or it.get('time') or it.get('frame') or it.get('phase_id')
            if pid is not None and phase is not None:
                return self._resolve_patient_phase(str(pid), str(phase))
            if img_p is not None:  # infer mask
                return str(img_p), self._infer_mask_from_image(str(img_p))
            raise ValueError("dict lacks paths and (patient, phase).")

        # tuple/list/ndarray: handle (patient, phase) OR (img_path, mask_path)
        if isinstance(it, (list, tuple, np.ndarray)):
            flat = list(it)
            if len(flat) >= 2 and all(isinstance(x, str) for x in flat[:2]):
                a, b = flat[0], flat[1]
                # If both look like paths -> (img, mask)
                if _is_file_path(a) and (_is_file_path(b) or b is None):
                    return a, (b if b is not None else self._infer_mask_from_image(a))
                # else treat as (patient, phase)
                return self._resolve_patient_phase(a, b)
            elif len(flat) == 1 and isinstance(flat[0], str):
                # single string can be path or patient id (phase default to ED)
                s = flat[0]
                if _is_file_path(s):
                    return s, self._infer_mask_from_image(s)
                return self._resolve_patient_phase(s, "ED")
            raise ValueError(f"Unsupported tuple/list/ndarray form: {it}")

        # single string: path or id (default ED)
        if isinstance(it, str):
            if _is_file_path(it): return it, self._infer_mask_from_image(it)
            return self._resolve_patient_phase(it, "ED")

        raise ValueError(f"Unsupported entry type: {type(it)}")

    def _get_first_key(self, d, keys):
        for k in keys:
            if k in d and d[k] is not None:
                return d[k]
        return None

    # ---------- patient+phase resolution ----------
    def _resolve_patient_phase(self, patient: str, phase: str):
        patient = str(patient).strip()
        phase   = _normalize_phase(str(phase))
        key = (patient, phase, self.view)
        if key in self._pp_cache: return self._pp_cache[key]

        # 1) Try by basenames and preferred dirs
        bases_img = _make_basenames(patient, phase, self.view)
        bases_msk = _make_mask_basenames(patient, phase, self.view)

        img_hits = _search_by_basenames(self.image_dirs_abs, bases_img, IMG_EXTS)
        msk_hits = _search_by_basenames(self.mask_dirs_abs,  bases_msk, MASK_EXTS)

        # 2) Loose search if needed
        if not img_hits:
            img_hits = _loose_search(self.image_dirs_abs, patient, self.view, phase, IMG_EXTS, require_mask_token=False)
        if not msk_hits:
            # require mask token here to avoid picking images
            msk_hits = _loose_search(self.mask_dirs_abs, patient, self.view, phase, MASK_EXTS, require_mask_token=True)

        if not img_hits:
            # Search whole data_root as last resort
            img_hits = _loose_search([self.data_root], patient, self.view, phase, IMG_EXTS, require_mask_token=False)
        if not msk_hits:
            msk_hits = _loose_search([self.data_root], patient, self.view, phase, MASK_EXTS, require_mask_token=True)

        if not img_hits:
            raise FileNotFoundError(f"No image found for patient='{patient}', phase='{phase}', view='{self.view}'")
        img_p = img_hits[0]

        if not msk_hits:
            # try infer from the chosen image
            inferred = _infer_mask_from_image(img_p)
            if inferred is None:
                raise FileNotFoundError(f"No mask found for patient='{patient}', phase='{phase}', view='{self.view}' "
                                        f"and could not infer from image:\n  {img_p}")
            msk_p = inferred
        else:
            # prefer mask sharing stem with image if present
            img_stem, _ = _strip_double_ext(os.path.basename(img_p))
            candidates = [m for m in msk_hits if _strip_double_ext(os.path.basename(m))[0].startswith(img_stem)]
            msk_p = candidates[0] if candidates else msk_hits[0]

        self._pp_cache[key] = (img_p, msk_p)
        return img_p, msk_p

    def _infer_mask_from_image(self, img_p: str):
        m = _infer_mask_from_image(img_p)
        if m is None:
            raise FileNotFoundError(f"Could not infer mask for image: {img_p}")
        return m

# ======================= Viz helpers =======================
PALETTE = {0:(0,0,0), 1:(255,0,0), 2:(0,255,0), 3:(0,0,255)}

def tensor_to_uint8_image(img_t):
    img = img_t.detach().cpu()
    if img.dim()!=3: raise ValueError(f"Expected [C,H,W], got {img.shape}")
    if img.size(0)==1: img = img.repeat(3,1,1)
    img = img.numpy().transpose(1,2,0)
    mn, mx = img.min(), img.max()
    if mx-mn<1e-8: img = np.zeros_like(img)
    else:
        if mn<0.0 or mx>1.0: img = (img-mn)/(mx-mn+1e-8)
    return (img*255).clip(0,255).astype(np.uint8)

def mask_to_color(mask_hw, palette=PALETTE):
    mask = mask_hw.detach().cpu().numpy().astype(np.int64)
    h,w = mask.shape
    color = np.zeros((h,w,3), dtype=np.uint8)
    for c,rgb in palette.items():
        color[mask==c] = rgb
    return color

def overlay_image(base_rgb, mask_rgb, alpha=0.45):
    base = base_rgb.astype(np.float32)
    mask = mask_rgb.astype(np.float32)
    out = (1-alpha)*base + alpha*mask
    return out.clip(0,255).astype(np.uint8)

def save_visuals(img_t, pred_hw, gt_hw, out_dir, name, alpha=0.45):
    os.makedirs(out_dir, exist_ok=True)
    img_rgb  = tensor_to_uint8_image(img_t)
    pred_rgb = mask_to_color(pred_hw)
    gt_rgb   = mask_to_color(gt_hw)
    Image.fromarray(pred_rgb).save(os.path.join(out_dir, f"{name}_pred.png"))
    Image.fromarray(gt_rgb).save(os.path.join(out_dir, f"{name}_gt.png"))
    Image.fromarray(img_rgb).save(os.path.join(out_dir, f"{name}_img.png"))
    Image.fromarray(overlay_image(img_rgb, pred_rgb, alpha)).save(os.path.join(out_dir, f"{name}_overlay_pred.png"))

def save_confusion_matrix(cm, out_path):
    np.savetxt(out_path, np.asarray(cm.cpu(), dtype=np.int64), fmt='%d', delimiter=',')

def log_metrics_csv(csv_path, epoch, tr_loss, te_loss, mDice, mIoU, dice_list, iou_list, lr):
    header = (["epoch","train_loss","val_loss","mDice","mIoU","lr"] +
              [f"dice_c{i}" for i in range(len(dice_list))] +
              [f"iou_c{i}"  for i in range(len(iou_list))])
    row = [epoch, f"{tr_loss:.6f}", f"{te_loss:.6f}", f"{mDice:.6f}", f"{mIoU:.6f}", f"{lr:.8f}"] + \
          [f"{d:.6f}" for d in dice_list] + [f"{i:.6f}" for i in iou_list]
    write_header = not os.path.exists(csv_path)
    ensure_dir(os.path.dirname(csv_path))
    with open(csv_path, "a", newline="") as f:
        w = csv.writer(f)
        if write_header: w.writerow(header)
        w.writerow(row)

# ======================= Losses =======================
def one_hot(target, num_classes, ignore_index=None):
    B,H,W = target.shape
    oh = torch.zeros(B, num_classes, H, W, device=target.device, dtype=torch.float32)
    if ignore_index is not None:
        valid = (target != ignore_index)
        idx = (target * valid).long()
        oh.scatter_(1, idx.unsqueeze(1), 1.0)
        oh = oh * valid.unsqueeze(1)
    else:
        oh.scatter_(1, target.unsqueeze(1), 1.0)
    return oh

class SoftDiceLoss(nn.Module):
    def __init__(self, smooth=1.0, ignore_index=None, class_weights=None):
        super().__init__()
        self.smooth = smooth
        self.ignore_index = ignore_index
        self.class_weights = class_weights
    def forward(self, pred, target):
        C = pred.shape[1]
        prob = F.softmax(pred, dim=1)
        tgt  = one_hot(target, C, self.ignore_index)
        dims = (0,2,3)
        inter = torch.sum(prob * tgt, dims)
        card  = torch.sum(prob + tgt, dims)
        dice_c = (2.*inter + self.smooth) / (card + self.smooth)
        loss_c = 1.0 - dice_c
        if self.class_weights is not None:
            w = self.class_weights.to(loss_c.device)
            return (loss_c * w).sum() / (w.sum() + 1e-8)
        return loss_c.mean()

class DiceCELoss(nn.Module):
    def __init__(self, dice_w=0.5, ce_w=0.5, class_weights=None, ignore_index=None, smooth=1.0):
        super().__init__()
        self.dice = SoftDiceLoss(smooth=smooth, ignore_index=ignore_index, class_weights=class_weights)
        self.dice_w = float(dice_w); self.ce_w = float(ce_w)
        self.class_weights = class_weights; self.ignore_index = ignore_index
    def forward(self, pred, target):
        ld = self.dice(pred, target)
        weight = self.class_weights.to(pred.device) if self.class_weights is not None else None
        if self.ignore_index is None:
            lce = F.cross_entropy(pred, target, weight=weight)
        else:
            lce = F.cross_entropy(pred, target, weight=weight, ignore_index=self.ignore_index)
        return self.dice_w * ld + self.ce_w * lce

# ======================= U-Net =======================
class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )
    def forward(self, x): return self.block(x)

class Down(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.pool = nn.MaxPool2d(2); self.conv = DoubleConv(in_ch, out_ch)
    def forward(self, x): return self.conv(self.pool(x))

class Up(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.conv = DoubleConv(in_ch, out_ch)
    def forward(self, x1, x2):
        x1 = self.up(x1)
        diffY = x2.size(2) - x1.size(2)
        diffX = x2.size(3) - x1.size(3)
        if diffY!=0 or diffX!=0:
            x1 = F.pad(x1, [diffX//2, diffX-diffX//2, diffY//2, diffY-diffY//2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

class OutConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=1)
    def forward(self, x): return self.conv(x)

class UNet(nn.Module):
    def __init__(self, in_channels=IN_CHANNELS, out_channels=NUM_CLASSES, base_ch=64):
        super().__init__()
        self.inc   = DoubleConv(in_channels, base_ch)
        self.down1 = Down(base_ch, base_ch*2)
        self.down2 = Down(base_ch*2, base_ch*4)
        self.down3 = Down(base_ch*4, base_ch*8)
        self.down4 = Down(base_ch*8, base_ch*8)
        self.up1   = Up(base_ch*16, base_ch*4)
        self.up2   = Up(base_ch*8,  base_ch*2)
        self.up3   = Up(base_ch*4,  base_ch)
        self.up4   = Up(base_ch*2,  base_ch)
        self.outc  = OutConv(base_ch, out_channels)
    def forward(self, x):
        x1 = self.inc(x); x2 = self.down1(x1); x3 = self.down2(x2); x4 = self.down3(x3); x5 = self.down4(x4)
        x  = self.up1(x5, x4); x = self.up2(x, x3); x = self.up3(x, x2); x = self.up4(x, x1)
        return self.outc(x)  # logits [B,C,H,W]

# ======================= Size/Channel helpers =======================
def resize_batch_to(imgs, masks, size_hw):
    Ht,Wt = size_hw
    if imgs.shape[-2:] != (Ht,Wt):
        imgs = F.interpolate(imgs, size=(Ht,Wt), mode='bilinear', align_corners=False)
    if masks.shape[-2:] != (Ht,Wt):
        masks_f = masks.unsqueeze(1).float()
        masks_r = F.interpolate(masks_f, size=(Ht,Wt), mode='nearest')
        masks   = masks_r.squeeze(1).long()
    return imgs, masks

def align_logits_to_masks(logits, masks):
    if logits.shape[-2:] != masks.shape[-2:]:
        logits = F.interpolate(logits, size=masks.shape[-2:], mode='bilinear', align_corners=False)
    return logits

def ensure_channels(imgs, required_c):
    B,C,H,W = imgs.shape
    if C == required_c: return imgs
    if C == 1 and required_c == 3: return imgs.repeat(1,3,1,1)
    if C == 3 and required_c == 1: return imgs.mean(dim=1, keepdim=True)
    raise ValueError(f"Got {C} channels, expected {required_c}")

# ======================= Train / Eval =======================
def train_one_epoch(model, loader, optimizer, criterion, device, img_size):
    model.train(); running = 0.0
    for imgs, masks in loader:
        imgs, masks = imgs.to(device), masks.to(device)
        imgs = ensure_channels(imgs, IN_CHANNELS)
        imgs, masks = resize_batch_to(imgs, masks, (img_size, img_size))
        optimizer.zero_grad()
        logits = model(imgs); logits = align_logits_to_masks(logits, masks)
        loss = criterion(logits, masks); loss.backward(); optimizer.step()
        running += loss.item()
    return running / max(1, len(loader))

@torch.no_grad()
def evaluate(model, loader, criterion, device, num_classes, save_dir=None, epoch=None, save_n=8, img_size=IMG_SIZE):
    model.eval(); running = 0.0
    cm = torch.zeros(num_classes, num_classes, dtype=torch.double)
    saved = 0; out_dir = None
    if save_dir is not None and epoch is not None and save_n > 0:
        out_dir = os.path.join(save_dir, f"epoch_{epoch:03d}"); os.makedirs(out_dir, exist_ok=True)

    for bidx, (imgs, masks) in enumerate(loader):
        imgs, masks = imgs.to(device), masks.to(device)
        imgs = ensure_channels(imgs, IN_CHANNELS)
        imgs, masks = resize_batch_to(imgs, masks, (img_size, img_size))
        logits = model(imgs); logits = align_logits_to_masks(logits, masks)
        loss = criterion(logits, masks); running += loss.item()
        preds = torch.argmax(logits, dim=1)
        k = (masks.cpu() * num_classes + preds.cpu()).view(-1)
        binc = torch.bincount(k, minlength=num_classes**2); cm += binc.reshape(num_classes, num_classes).to(cm.dtype)
        if out_dir is not None and saved < save_n:
            B = imgs.size(0); take = min(B, save_n - saved)
            for i in range(take):
                name = f"b{bidx}_i{i}"
                save_visuals(imgs[i], preds[i], masks[i], out_dir, name, alpha=0.45)
            saved += take

    tp = cm.diag(); fp = cm.sum(0) - tp; fn = cm.sum(1) - tp; eps = 1e-6
    per_class_dice = ((2*tp + eps) / (2*tp + fp + fn + eps)).tolist()
    mDice = float(cm.new_tensor(per_class_dice).mean().item())
    per_class_iou  = ((tp + eps) / (tp + fp + fn + eps)).tolist()
    mIoU = float(cm.new_tensor(per_class_iou).mean().item())
    return running / max(1, len(loader)), per_class_dice, mDice, per_class_iou, mIoU, cm

# ======================= Main =======================
def main():
    set_seed(SEED)
    ensure_dir(RESULTS_DIR)

    if not os.path.isfile(TRAIN_LIST): raise FileNotFoundError(f"Missing list file: {TRAIN_LIST}")
    if not os.path.isfile(TEST_LIST):  raise FileNotFoundError(f"Missing list file: {TEST_LIST}")

    # This dataset handles entries like ('patient0359','ED') and maps them to files.
    train_ds = CAMUSAnyListDataset(DATA_DIR, TRAIN_LIST, view=VIEW, in_channels=IN_CHANNELS)
    test_ds  = CAMUSAnyListDataset(DATA_DIR, TEST_LIST,  view=VIEW, in_channels=IN_CHANNELS)
    print(f"Loaded {len(train_ds)} train and {len(test_ds)} test samples | View={VIEW}")

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=NUM_WORKERS, pin_memory=True)
    test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE, shuffle=False,
                              num_workers=NUM_WORKERS, pin_memory=True)

    model = UNet(in_channels=IN_CHANNELS, out_channels=NUM_CLASSES).to(DEVICE)

    class_weights = None
    criterion = DiceCELoss(dice_w=0.5, ce_w=0.5,
                           class_weights=class_weights,
                           ignore_index=None,
                           smooth=1.0)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=STEP_SIZE, gamma=GAMMA)

    best_mdice = 0.0; patience, bad = 20, 0

    for epoch in range(EPOCHS):
        tr_loss = train_one_epoch(model, train_loader, optimizer, criterion, DEVICE, IMG_SIZE)
        curr_lr = optimizer.param_groups[0]['lr']
        te_loss, per_cls_dice, mDice, per_cls_iou, mIoU, cm = evaluate(
            model, test_loader, criterion, DEVICE, NUM_CLASSES,
            save_dir=SAVE_ROOT, epoch=epoch+1, save_n=SAVE_N, img_size=IMG_SIZE
        )
        print(
            f"[UNet] Epoch {epoch+1:03d} | "
            f"train {tr_loss:.4f} | val {te_loss:.4f} | "
            f"mDice {mDice:.4f} | mIoU {mIoU:.4f} | "
            f"Dice {['%.3f'%d for d in per_cls_dice]} | "
            f"IoU  {['%.3f'%i for i in per_cls_iou]} | "
            f"LR {curr_lr:.2e}"
        )

        log_metrics_csv(METRICS_CSV, epoch+1, tr_loss, te_loss, mDice, mIoU,
                        per_cls_dice, per_cls_iou, curr_lr)

        cm_path = os.path.join(RESULTS_DIR, f"confusion_matrix_epoch_{epoch+1:03d}.csv")
        save_confusion_matrix(cm, cm_path)

        scheduler.step()
        if mDice > best_mdice:
            best_mdice = mDice; bad = 0
            torch.save(model.state_dict(), f"best_UNet_dicece_mdice_{best_mdice:.4f}.pt")
        else:
            bad += 1
            if bad >= patience:
                print("Early stopping."); break

    print(f"\n✅ Metrics logged to: {METRICS_CSV}")
    print(f"   Confusion matrices: {RESULTS_DIR}/confusion_matrix_epoch_XXX.csv")

if __name__ == "__main__":
    main()
