# train_transunet_dice_ce_focal_raw.py
import os
import csv
import time
import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

# --- your modules ---
# from dataset import CAMUS_loader             # loads directly from NIfTI using patient lists
from preprocess_dataset import CAMUSPreprocessor  # builds splits and writes .npz files
# Try to import TransUNet variants from your file
from Trans_Unet import __dict__ as TU_DICT   # we’ll pick the class at runtime

# ======================= Config =======================
NUM_CLASSES = 4
IMG_SIZE    = 256                 # keep 256 so H/16=W/16=16 at bottleneck
VIEW        = '2CH'               # or '4CH'
BATCH_SIZE  = 32
EPOCHS      = 60
LR          = 1e-4
WEIGHT_DECAY= 1e-4
STEP_SIZE   = 10
GAMMA       = 0.1
SAVE_N      = 8
NUM_WORKERS = 0 if os.name == "nt" else 4  # Windows-friendly default
SEED        = 42
DEVICE      = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Paths (must already exist and point to .npy split files your pipeline created)
DATA_DIR    = 'database_nifti'
SPLIT_DIR   = 'prepared_data'
TRAIN_LIST  = os.path.join(SPLIT_DIR, 'train_samples.npy')
VAL_LIST    = os.path.join(SPLIT_DIR, 'test_ED.npy')   # or 'test_ES.npy'

# Outputs
RESULTS_DIR = "TransUNet_Preprocessed_results_Dice_CE_Focal"
METRICS_CSV = os.path.join(RESULTS_DIR, "TransUNet_preprocessed_metrics_Dice_CE_Focal.csv")
SAVE_ROOT   = "qualitative_Preprocessed_TransUNet_Dice_CE_Focal"

# ===== Visualization palette =====
PALETTE = {
    0: (0, 0, 0),       # background
    1: (255, 0, 0),     # LV endocardium
    2: (0, 255, 0),     # LV myocardium
    3: (0, 0, 255),     # Left atrium
}

# ======================= Utils =======================
def set_seed(seed=42):
    import random
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True

def ensure_dir(p):
    os.makedirs(p, exist_ok=True); return p

def tensor_to_uint8_image(img_t):
    """img_t: [C,H,W] -> uint8 RGB [H,W,3]"""
    img = img_t.detach().cpu()
    if img.dim() != 3: raise ValueError(f"Expected [C,H,W], got {img.shape}")
    if img.size(0) == 1: img = img.repeat(3, 1, 1)
    img = img.numpy().transpose(1, 2, 0)
    mn, mx = img.min(), img.max()
    if mx - mn < 1e-8: img = np.zeros_like(img)
    else:
        if mn < 0.0 or mx > 1.0:
            img = (img - mn) / (mx - mn + 1e-8)
    return (img * 255.0).clip(0, 255).astype(np.uint8)

def mask_to_color(mask_hw, palette=PALETTE):
    mask = mask_hw.detach().cpu().numpy().astype(np.int64)
    h, w = mask.shape
    color = np.zeros((h, w, 3), dtype=np.uint8)
    for c, rgb in palette.items(): color[mask == c] = rgb
    return color

def overlay_image(base_rgb, mask_rgb, alpha=0.45):
    base = base_rgb.astype(np.float32); mask = mask_rgb.astype(np.float32)
    return ((1 - alpha) * base + alpha * mask).clip(0, 255).astype(np.uint8)

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
    header = (["epoch","train_loss","val_loss","mDice","mIoU","lr"]
              + [f"dice_c{i}" for i in range(len(dice_list))]
              + [f"iou_c{i}"  for i in range(len(iou_list))])
    row = [epoch, f"{tr_loss:.6f}", f"{te_loss:.6f}", f"{mDice:.6f}", f"{mIoU:.6f}", f"{lr:.8f}"]
    row += [f"{d:.6f}" for d in dice_list] + [f"{i:.6f}" for i in iou_list]
    write_header = not os.path.exists(csv_path)
    ensure_dir(os.path.dirname(csv_path))
    with open(csv_path, "a", newline="") as f:
        w = csv.writer(f)
        if write_header: w.writerow(header)
        w.writerow(row)

# ======================= Losses (Dice + CE + Focal) =======================
def one_hot(target, num_classes, ignore_index=None):
    """[B,H,W] -> [B,C,H,W] float32 one-hot (zeros on ignored)."""
    B, H, W = target.shape
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
        self.smooth = float(smooth)
        self.ignore_index = ignore_index
        self.class_weights = class_weights
    def forward(self, pred, target):
        C = pred.shape[1]
        prob = F.softmax(pred, dim=1)
        tgt  = one_hot(target, C, self.ignore_index)
        dims = (0, 2, 3)
        inter = torch.sum(prob * tgt, dims)
        card  = torch.sum(prob + tgt, dims)
        dice_c = (2. * inter + self.smooth) / (card + self.smooth)
        loss_c = 1.0 - dice_c
        if self.class_weights is not None:
            w = self.class_weights.to(loss_c.device)
            return (loss_c * w).sum() / (w.sum() + 1e-8)
        return loss_c.mean()

class FocalLossCE(nn.Module):
    def __init__(self, gamma=2.0, alpha=None, class_weights=None, ignore_index=None, reduction="mean"):
        super().__init__()
        self.gamma = float(gamma)
        self.alpha = alpha
        self.class_weights = class_weights
        self.ignore_index = ignore_index
        self.reduction = reduction
    def forward(self, pred, target):
        ce = F.cross_entropy(
            pred, target,
            weight=(self.class_weights.to(pred.device) if self.class_weights is not None else None),
            reduction="none",
            ignore_index=(self.ignore_index if self.ignore_index is not None else -100)
        )
        if self.ignore_index is not None:
            valid = (target != self.ignore_index)
            ce = ce[valid]
            if ce.numel() == 0:
                return torch.zeros((), device=pred.device, dtype=pred.dtype)
            tgt_valid = target[valid]
        else:
            tgt_valid = target.view(-1)
        pt = torch.exp(-ce)                  # pt = exp(-CE)
        focal = (1.0 - pt) ** self.gamma * ce
        if self.alpha is not None:
            if isinstance(self.alpha, (list, tuple, np.ndarray)):
                alpha_vec = torch.tensor(self.alpha, dtype=focal.dtype, device=pred.device)
                focal = alpha_vec[tgt_valid] * focal
            elif torch.is_tensor(self.alpha):
                focal = self.alpha.to(pred.device, dtype=focal.dtype)[tgt_valid] * focal
            else:
                focal = float(self.alpha) * focal
        if self.reduction == "mean": return focal.mean()
        if self.reduction == "sum":  return focal.sum()
        return focal

class DiceCEFocalLoss(nn.Module):
    def __init__(self,
                 dice_w=0.4, ce_w=0.4, focal_w=0.2,
                 smooth=1.0,
                 class_weights=None,
                 ignore_index=None,
                 focal_gamma=2.0,
                 focal_alpha=None):
        super().__init__()
        self.dice_w   = float(dice_w)
        self.ce_w     = float(ce_w)
        self.focal_w  = float(focal_w)
        self.ignore_index  = ignore_index
        self.class_weights = class_weights
        self.dice  = SoftDiceLoss(smooth=smooth, ignore_index=ignore_index, class_weights=class_weights)
        self.focal = FocalLossCE(gamma=focal_gamma, alpha=focal_alpha,
                                 class_weights=class_weights, ignore_index=ignore_index, reduction="mean")
    def forward(self, pred, target):
        ld  = self.dice(pred, target)
        weight = self.class_weights.to(pred.device) if self.class_weights is not None else None
        if self.ignore_index is None:
            lce = F.cross_entropy(pred, target, weight=weight)
        else:
            lce = F.cross_entropy(pred, target, weight=weight, ignore_index=self.ignore_index)
        lf  = self.focal(pred, target)
        return self.dice_w * ld + self.ce_w * lce + self.focal_w * lf

# ================== Size Guards & Alignment ==================
def resize_batch_to(imgs, masks, size_hw):
    """Force images & masks to (H,W)=size_hw inside the training loop."""
    Ht, Wt = size_hw
    if imgs.shape[-2:] != (Ht, Wt):
        imgs = F.interpolate(imgs, size=(Ht, Wt), mode='bilinear', align_corners=False)
    if masks.shape[-2:] != (Ht, Wt):
        masks_f = masks.unsqueeze(1).float()
        masks_r = F.interpolate(masks_f, size=(Ht, Wt), mode='nearest')
        masks   = masks_r.squeeze(1).long()
    return imgs, masks

def align_logits_to_masks(logits, masks):
    if logits.shape[-2:] != masks.shape[-2:]:
        logits = F.interpolate(logits, size=masks.shape[-2:], mode='bilinear', align_corners=False)
    return logits

# ======================= TransUNet factory =======================
def make_transunet(img_size=256, in_ch=1, n_classes=4):
    """
    Tries common class names and constructor signatures across TransUNet repos.
    """
    candidates = [
        "TransUNetLite", "TransUNet", "UNet"   # different files export different names
    ]
    cls = None
    for name in candidates:
        if name in TU_DICT and isinstance(TU_DICT[name], type):
            cls = TU_DICT[name]; break
    if cls is None:
        raise ImportError("Could not find TransUNet class (tried TransUNetLite/TransUNet/UNet) in Trans_Unet.py")

    # Try common signatures
    tried = []
    for kwargs in [
        {"in_channels": in_ch, "num_classes": n_classes, "img_size": img_size, "patch_size": 16},
        {"in_ch": in_ch, "n_classes": n_classes, "img_size": img_size, "patch_size": 16},
        {"in_channels": in_ch, "num_classes": n_classes, "img_size": img_size},
        {"in_ch": in_ch, "n_classes": n_classes, "img_size": img_size},
        {"in_channels": in_ch, "num_classes": n_classes},
        {"in_ch": in_ch, "n_classes": n_classes},
        {},  # as a last resort
    ]:
        try:
            model = cls(**kwargs)
            print(f"[TransUNet] Initialized with args: {kwargs}")
            return model
        except TypeError as e:
            tried.append((kwargs, str(e)))
        except Exception as e:
            tried.append((kwargs, f"{type(e).__name__}: {e}"))

    msg = "Failed to construct TransUNet with tried kwargs:\n" + \
          "\n".join([f"  {kw} -> {err}" for kw, err in tried])
    raise RuntimeError(msg)

# ======================= Train / Eval =======================
def train_one_epoch(model, loader, optimizer, criterion, device, img_size):
    model.train(); running_loss = 0.0
    for imgs, masks in loader:
        imgs, masks = imgs.to(device), masks.to(device)
        imgs, masks = resize_batch_to(imgs, masks, (img_size, img_size))
        optimizer.zero_grad(set_to_none=True)
        logits = model(imgs)                   # [B,C,H,W]
        logits = align_logits_to_masks(logits, masks)
        loss = criterion(logits, masks)
        loss.backward(); optimizer.step()
        running_loss += loss.item()
    return running_loss / max(1, len(loader))

@torch.no_grad()
def evaluate(model, loader, criterion, device, num_classes, save_dir=None, epoch=None, save_n=8, img_size=IMG_SIZE):
    model.eval(); running_loss = 0.0
    cm = torch.zeros(num_classes, num_classes, dtype=torch.double)
    saved = 0
    out_dir = None
    if save_dir is not None and epoch is not None and save_n > 0:
        out_dir = os.path.join(save_dir, f"epoch_{epoch:03d}"); os.makedirs(out_dir, exist_ok=True)
    for bidx, (imgs, masks) in enumerate(loader):
        imgs, masks = imgs.to(device), masks.to(device)
        imgs, masks = resize_batch_to(imgs, masks, (img_size, img_size))
        logits = model(imgs)
        logits = align_logits_to_masks(logits, masks)
        loss = criterion(logits, masks); running_loss += loss.item()
        preds = torch.argmax(logits, dim=1)
        k = (masks.cpu() * num_classes + preds.cpu()).view(-1)
        binc = torch.bincount(k, minlength=num_classes**2)
        cm += binc.reshape(num_classes, num_classes).to(cm.dtype)
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
    return running_loss / max(1, len(loader)), per_class_dice, mDice, per_class_iou, mIoU, cm

# ======================= Main =======================
def main():
    set_seed(SEED)
    ensure_dir(RESULTS_DIR)

    # --------- Datasets & Loaders from RAW NIfTI via CAMUS_loader ----------
    if not (os.path.isfile(TRAIN_LIST) and os.path.isfile(VAL_LIST)):
        raise FileNotFoundError(
            f"Missing split files.\nExpected:\n  {TRAIN_LIST}\n  {VAL_LIST}\n"
            "Create them first (your previous split/preprocess step)."
        )

    train_ds = CAMUSPreprocessor(DATA_DIR, TRAIN_LIST, view=VIEW)
    val_ds   = CAMUSPreprocessor(DATA_DIR, VAL_LIST,   view=VIEW)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=NUM_WORKERS, pin_memory=True)
    val_loader   = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False,
                              num_workers=NUM_WORKERS, pin_memory=True)

    # --------- Model (TransUNet) ----------
    model = make_transunet(img_size=IMG_SIZE, in_ch=1, n_classes=NUM_CLASSES).to(DEVICE)

    # --------- Loss: Dice + CrossEntropy + Focal ----------
    class_weights = None  # or torch.tensor([...], device=DEVICE)
    criterion = DiceCEFocalLoss(
        dice_w=0.4, ce_w=0.4, focal_w=0.2,
        smooth=1.0,
        class_weights=class_weights,
        ignore_index=None,
        focal_gamma=2.0,
        focal_alpha=None
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=STEP_SIZE, gamma=GAMMA)

    best_mdice = 0.0
    patience, bad = 20, 0
    t0 = time.time()

    for epoch in range(EPOCHS):
        tr_loss = train_one_epoch(model, train_loader, optimizer, criterion, DEVICE, IMG_SIZE)
        curr_lr = optimizer.param_groups[0]['lr']

        te_loss, per_cls_dice, mDice, per_cls_iou, mIoU, cm = evaluate(
            model, val_loader, criterion, DEVICE, NUM_CLASSES,
            save_dir=SAVE_ROOT, epoch=epoch+1, save_n=SAVE_N, img_size=IMG_SIZE
        )

        print(
            f"[TransUNet+Dice+CE+Focal] Epoch {epoch+1:03d} | "
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
            torch.save(model.state_dict(), f"best_transunet_dice_ce_focal_mdice_{best_mdice:.4f}.pt")
        else:
            bad += 1
            if bad >= patience:
                print("Early stopping."); break

    dt = time.time() - t0
    print(f"\n✅ Metrics logged to: {METRICS_CSV}")
    print(f"   Confusion matrices: {RESULTS_DIR}/confusion_matrix_epoch_XXX.csv")

if __name__ == "__main__":
    main()
