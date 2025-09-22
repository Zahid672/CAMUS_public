# train_unet_dice_ce_focal.py
import os
import csv
import time
import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

# --- your modules ---
from dataset import CAMUS_loader  # reads .npz files written by CAMUSPreprocessor
# from preprocess_dataset import CAMUSPreprocessor  # builds splits and writes .npz files if missing
from Unet import UNet
# from Attention_Unet import UNet                   # your UNet (1 in-channel -> 4 classes)

# ======================= Config =======================
NUM_CLASSES = 4
IN_CHANNELS = 1             # Attention_Unet expects 1-channel input
IMG_SIZE    = 256
VIEW        = '2CH'         # or '4CH'
BATCH_SIZE  = 32
EPOCHS      = 60
LR          = 1e-4
WEIGHT_DECAY= 1e-4
STEP_SIZE   = 10
GAMMA       = 0.1
SAVE_N      = 8
NUM_WORKERS = 0 if os.name == "nt" else 4  # safer default on Windows
SEED        = 42
DEVICE      = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Paths produced by your preprocessor
DATA_DIR    = 'database_nifti'
SPLIT_DIR   = 'prepared_data'
PREPROC_DIR = 'preprocessed'   # where .npz will be stored
TRAIN_SPLIT = 'train'
VAL_SPLIT   = 'test_ED'        # or 'test_ES'

# Outputs
RESULTS_DIR = "UNet_Raw_Data_results_Dice_CE_Focal"
METRICS_CSV = os.path.join(RESULTS_DIR, "UNet_raw_metrics_Dice_CE_Focal.csv")
SAVE_ROOT   = "qualitative_Raw_UNet_Dice_CE_Focal"

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
    os.makedirs(p, exist_ok=True)
    return p

def tensor_to_uint8_image(img_t):
    """img_t: [C,H,W] -> uint8 RGB [H,W,3]"""
    img = img_t.detach().cpu()
    if img.dim() != 3:
        raise ValueError(f"Expected [C,H,W], got {img.shape}")
    if img.size(0) == 1:
        img = img.repeat(3, 1, 1)
    img = img.numpy().transpose(1, 2, 0)
    mn, mx = img.min(), img.max()
    if mx - mn < 1e-8:
        img = np.zeros_like(img)
    else:
        if mn < 0.0 or mx > 1.0:
            img = (img - mn) / (mx - mn + 1e-8)
    return (img * 255.0).clip(0, 255).astype(np.uint8)

def mask_to_color(mask_hw, palette=PALETTE):
    """mask_hw: [H,W] int -> color RGB [H,W,3] uint8"""
    mask = mask_hw.detach().cpu().numpy().astype(np.int64)
    h, w = mask.shape
    color = np.zeros((h, w, 3), dtype=np.uint8)
    for c, rgb in palette.items():
        color[mask == c] = rgb
    return color

def overlay_image(base_rgb, mask_rgb, alpha=0.45):
    base = base_rgb.astype(np.float32)
    mask = mask_rgb.astype(np.float32)
    out = (1 - alpha) * base + alpha * mask
    return out.clip(0, 255).astype(np.uint8)

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
        if write_header:
            w.writerow(header)
        w.writerow(row)

# ======================= Dataset (reads .npz) =======================
class CAMUSNPZDataset(Dataset):
    """
    Expects folders created by CAMUSPreprocessor:
        preprocessed/train/*.npz
        preprocessed/test_ED/*.npz
        preprocessed/test_ES/*.npz
    Each .npz contains:
        image [H,W] float32 in [0,1], mask [H,W] uint8, meta [patient, instant]
    """
    def __init__(self, root_dir, split, in_channels=1):
        super().__init__()
        self.dir = os.path.join(root_dir, split)
        if not os.path.isdir(self.dir):
            raise FileNotFoundError(f"Split directory not found: {self.dir}")
        self.files = sorted([os.path.join(self.dir, f) for f in os.listdir(self.dir) if f.endswith(".npz")])
        if not self.files:
            raise RuntimeError(f"No .npz files found in {self.dir}")
        self.in_channels = int(in_channels)

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        arr = np.load(self.files[idx], allow_pickle=True)
        img = arr["image"]  # [H,W] float32
        msk = arr["mask"]   # [H,W] uint8
        img_t = torch.from_numpy(img).float().unsqueeze(0)  # [1,H,W]
        if self.in_channels == 3:
            img_t = img_t.repeat(3, 1, 1)
        msk_t = torch.from_numpy(msk).long()                # [H,W]
        return img_t, msk_t

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
    """
    Multi-class soft Dice (1 - mean Dice) on logits.
    """
    def __init__(self, smooth=1.0, ignore_index=None, class_weights=None):
        super().__init__()
        self.smooth = float(smooth)
        self.ignore_index = ignore_index
        self.class_weights = class_weights

    def forward(self, pred, target):
        # pred: [B,C,H,W] logits; target: [B,H,W] long
        C = pred.shape[1]
        prob = F.softmax(pred, dim=1)                # [B,C,H,W]
        tgt  = one_hot(target, C, self.ignore_index) # [B,C,H,W]

        dims = (0, 2, 3)
        inter = torch.sum(prob * tgt, dims)
        card  = torch.sum(prob + tgt, dims)
        dice_c = (2. * inter + self.smooth) / (card + self.smooth)   # [C]
        loss_c = 1.0 - dice_c

        if self.class_weights is not None:
            w = self.class_weights.to(loss_c.device)
            return (loss_c * w).sum() / (w.sum() + 1e-8)
        return loss_c.mean()

class FocalLossCE(nn.Module):
    """
    Multi-class Focal Loss built on top of CrossEntropy.
    Focal term: ((1 - pt) ** gamma) * CE, with optional alpha per class.
      - gamma: focusing parameter
      - alpha: None | float | list/1D tensor of shape [C]
      - class_weights: passed into CE as 'weight' (PyTorch per-class weights)
      - ignore_index: standard CE masking
    """
    def __init__(self, gamma=2.0, alpha=None, class_weights=None, ignore_index=None, reduction="mean"):
        super().__init__()
        self.gamma = float(gamma)
        self.alpha = alpha
        self.class_weights = class_weights
        self.ignore_index = ignore_index
        self.reduction = reduction

    def forward(self, pred, target):
        # pred: [B,C,H,W] logits; target: [B,H,W] long
        ce = F.cross_entropy(
            pred, target,
            weight=(self.class_weights.to(pred.device) if self.class_weights is not None else None),
            reduction="none",
            ignore_index=(self.ignore_index if self.ignore_index is not None else -100)
        )
        # mask ignored
        if self.ignore_index is not None:
            valid = (target != self.ignore_index)
            ce = ce[valid]
            if ce.numel() == 0:
                return torch.zeros((), device=pred.device, dtype=pred.dtype)
            tgt_valid = target[valid]
        else:
            tgt_valid = target.view(-1)

        pt = torch.exp(-ce)                       # pt = exp(-CE)
        focal = (1.0 - pt) ** self.gamma * ce     # focal term

        # alpha handling
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
    """
    Total = dice_w * SoftDice + ce_w * CrossEntropy + focal_w * FocalLoss
    """
    def __init__(self,
                 dice_w=0.4, ce_w=0.4, focal_w=0.2,
                 smooth=1.0,
                 class_weights=None,      # per-class weights for Dice/CE/Focal
                 ignore_index=None,
                 focal_gamma=2.0,
                 focal_alpha=None):       # None | float | list/1D tensor
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
        # Dice
        ld  = self.dice(pred, target)

        # CE
        weight = self.class_weights.to(pred.device) if self.class_weights is not None else None
        if self.ignore_index is None:
            lce = F.cross_entropy(pred, target, weight=weight)
        else:
            lce = F.cross_entropy(pred, target, weight=weight, ignore_index=self.ignore_index)

        # Focal
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

# ======================= Train / Eval =======================
def train_one_epoch(model, loader, optimizer, criterion, device, img_size):
    model.train()
    running_loss = 0.0
    for imgs, masks in loader:
        imgs, masks = imgs.to(device), masks.to(device)
        imgs, masks = resize_batch_to(imgs, masks, (img_size, img_size))
        optimizer.zero_grad()
        logits = model(imgs)                   # [B,C,H,W]
        logits = align_logits_to_masks(logits, masks)
        loss = criterion(logits, masks)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    return running_loss / max(1, len(loader))

@torch.no_grad()
def evaluate(model, loader, criterion, device, num_classes, save_dir=None, epoch=None, save_n=8, img_size=IMG_SIZE):
    model.eval()
    running_loss = 0.0
    cm = torch.zeros(num_classes, num_classes, dtype=torch.double)
    saved = 0
    out_dir = None
    if save_dir is not None and epoch is not None and save_n > 0:
        out_dir = os.path.join(save_dir, f"epoch_{epoch:03d}")
        os.makedirs(out_dir, exist_ok=True)

    for bidx, (imgs, masks) in enumerate(loader):
        imgs, masks = imgs.to(device), masks.to(device)
        imgs, masks = resize_batch_to(imgs, masks, (img_size, img_size))
        logits = model(imgs)
        logits = align_logits_to_masks(logits, masks)
        loss = criterion(logits, masks)
        running_loss += loss.item()

        preds = torch.argmax(logits, dim=1)
        # update confusion matrix
        k = (masks.cpu() * num_classes + preds.cpu()).view(-1)
        binc = torch.bincount(k, minlength=num_classes**2)
        cm += binc.reshape(num_classes, num_classes).to(cm.dtype)

        # save a few visuals
        if out_dir is not None and saved < save_n:
            B = imgs.size(0)
            take = min(B, save_n - saved)
            for i in range(take):
                name = f"b{bidx}_i{i}"
                save_visuals(imgs[i], preds[i], masks[i], out_dir, name, alpha=0.45)
            saved += take

    # dataset-level Dice/IoU from cm
    tp = cm.diag()
    fp = cm.sum(0) - tp
    fn = cm.sum(1) - tp
    eps = 1e-6
    per_class_dice = ((2*tp + eps) / (2*tp + fp + fn + eps)).tolist()
    mDice = float(cm.new_tensor(per_class_dice).mean().item())
    per_class_iou  = ((tp + eps) / (tp + fp + fn + eps)).tolist()
    mIoU = float(cm.new_tensor(per_class_iou).mean().item())
    return running_loss / max(1, len(loader)), per_class_dice, mDice, per_class_iou, mIoU, cm

# ======================= Main =======================
def main():
    set_seed(SEED)
    ensure_dir(RESULTS_DIR)

    # --------- Build splits + preprocess to .npz (idempotent) ----------
    pp = CAMUS_loader(
        data_dir=DATA_DIR,
        split_dir=SPLIT_DIR,
        out_dir=PREPROC_DIR,
        view=VIEW,
        img_size=IMG_SIZE,
        do_clahe=True,
        denoise="median",
        overwrite=False,
        seed=1234,
    )
    pp.build_splits_if_missing()
    # Only preprocess if missing or empty
    need_pre = (not os.path.isdir(os.path.join(PREPROC_DIR, TRAIN_SPLIT))) or \
               (len([f for f in os.listdir(os.path.join(PREPROC_DIR, TRAIN_SPLIT))]) == 0)
    if need_pre:
        pp.preprocess_all()
    else:
        print("Preprocessed .npz found. Skipping preprocessing.")

    # --------- Datasets & Loaders from .npz ----------
    train_ds = CAMUSNPZDataset(PREPROC_DIR, TRAIN_SPLIT, in_channels=IN_CHANNELS)
    val_ds   = CAMUSNPZDataset(PREPROC_DIR, VAL_SPLIT,   in_channels=IN_CHANNELS)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=NUM_WORKERS, pin_memory=True)
    val_loader   = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False,
                              num_workers=NUM_WORKERS, pin_memory=True)

    # --------- Model (Attention U-Net) ----------
    model = UNet().to(DEVICE)  # (expects 1->4 by your implementation)

    # --------- Loss: Dice + CrossEntropy + Focal ----------
    # Optional per-class weights (for CE/Dice/Focal) — leave None to start.
    class_weights = None  # e.g., torch.tensor([1.0, 2.0, 2.0, 1.5], device=DEVICE)

    criterion = DiceCEFocalLoss(
        dice_w=0.4,
        ce_w=0.4,
        focal_w=0.2,          # try 0.1–0.3 to start; tune as needed
        smooth=1.0,
        class_weights=class_weights,
        ignore_index=None,    # set to 255 if you use an ignore label
        focal_gamma=2.0,      # common choice
        focal_alpha=None      # or scalar like 0.25, or per-class list/tensor of length 4
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
            f"[UNet+Dice+CE+Focal] Epoch {epoch+1:03d} | "
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
            best_mdice = mDice
            bad = 0
            torch.save(model.state_dict(), f"best_unet_dice_ce_focal_mdice_{best_mdice:.4f}.pt")
        else:
            bad += 1
            if bad >= patience:
                print("Early stopping.")
                break

    dt = time.time() - t0
    print(f"\n✅ Metrics logged to: {METRICS_CSV}")
    print(f"   Confusion matrices: {RESULTS_DIR}/confusion_matrix_epoch_XXX.csv")

if __name__ == "__main__":
    main()
