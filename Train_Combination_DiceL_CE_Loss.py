import os
import csv
import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

# --- your modules ---
from preprocess_dataset import CAMUSPreprocessor  # your dataset loader
# from Unet import UNet  # UNet model (if you want to compare)
from Trans_Unet import TransUNetLite  # TransUNet model

# ======================= Config =======================
NUM_CLASSES = 4
IN_CHANNELS = 1
IMG_SIZE    = 256        # MUST be divisible by the internal patch size (usually 16). 256 -> bottleneck 16x16
VIEW        = '2CH'      # or '4CH'
BATCH_SIZE  = 2          # TransUNet is heavy; reduce if OOM
EPOCHS      = 60
LR          = 1e-4
WEIGHT_DECAY= 1e-4
STEP_SIZE   = 10
GAMMA       = 0.1
SAVE_N      = 8          # number of qualitative samples per epoch
DEVICE      = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

RESULTS_DIR = "TransUNet_Preprocessed_Data_results_DoubleLoss_Dice_CE"
METRICS_CSV = os.path.join(RESULTS_DIR, "TransUNet_preprocessed_metrics_DoubleLoss_Dice_CE.csv")
SAVE_ROOT   = "qualitative_Preprocessed_TransUNet_DoubleLoss_Dice_CE"

DATA_DIR    = 'database_nifti'
SPLIT_DIR   = 'prepared_data'
TRAIN_LIST  = os.path.join(SPLIT_DIR, 'train_samples.npy')
TEST_LIST   = os.path.join(SPLIT_DIR, 'test_ED.npy')   # or test_ES.npy

# ===== Visualization palette (edit colors if you like) =====
PALETTE = {
    0: (0, 0, 0),       # background
    1: (255, 0, 0),     # LV endocardium
    2: (0, 255, 0),     # LV myocardium
    3: (0, 0, 255),     # Left atrium
}

# ======================= Utils =======================
def ensure_dir(p):
    os.makedirs(p, exist_ok=True)
    return p

def tensor_to_uint8_image(img_t):
    """img_t: [C,H,W] -> uint8 RGB [H,W,3]"""
    img = img_t.detach().cpu()
    if img.dim() != 3:
        raise ValueError(f"Expected 3D tensor [C,H,W], got {img.shape}")
    if img.size(0) == 1:
        img = img.repeat(3, 1, 1)
    img = img.numpy().transpose(1, 2, 0)
    mn, mx = img.min(), img.max()
    if mx - mn < 1e-8:
        img = np.zeros_like(img)
    else:
        if mn < 0.0 or mx > 1.0:
            img = (img - mn) / (mx - mn + 1e-8)
    img = (img * 255.0).clip(0, 255).astype(np.uint8)
    return img

def mask_to_color(mask_hw, palette=PALETTE):
    """mask_hw: [H,W] int -> color RGB [H,W,3] uint8"""
    mask = mask_hw.detach().cpu().numpy().astype(np.int64)
    h, w = mask.shape
    color = np.zeros((h, w, 3), dtype=np.uint8)
    for c, rgb in palette.items():
        color[mask == c] = rgb
    return color

def overlay_image(base_rgb, mask_rgb, alpha=0.5):
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
    over_pred = overlay_image(img_rgb, pred_rgb, alpha=alpha)
    Image.fromarray(over_pred).save(os.path.join(out_dir, f"{name}_overlay_pred.png"))

def save_confusion_matrix(cm, out_path):
    np.savetxt(out_path, np.asarray(cm.cpu(), dtype=np.int64), fmt='%d', delimiter=',')

def log_metrics_csv(csv_path, epoch, tr_loss, te_loss, mDice, mIoU, dice_list, iou_list, lr):
    header = (["epoch","train_loss","val_loss","mDice","mIoU","lr"] +
              [f"dice_c{i}" for i in range(len(dice_list))] +
              [f"iou_c{i}"  for i in range(len(iou_list))])
    row = [epoch, f"{tr_loss:.6f}", f"{te_loss:.6f}", f"{mDice:.6f}", f"{mIoU:.6f}", f"{lr:.8f}"] + \
          [f"{d:.6f}" for d in dice_list] + [f"{i:.6f}" for i in iou_list]
    write_header = not os.path.exists(csv_path)
    with open(csv_path, "a", newline="") as f:
        w = csv.writer(f)
        if write_header:
            w.writerow(header)
        w.writerow(row)

# ======================= Losses =======================
def one_hot(target, num_classes, ignore_index=None):
    """[B,H,W] -> [B,C,H,W]"""
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
        self.smooth = smooth
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

class DiceCELoss(nn.Module):
    def __init__(self, dice_w=0.5, ce_w=0.5, class_weights=None, ignore_index=None, smooth=1.0):
        super().__init__()
        self.dice = SoftDiceLoss(smooth=smooth, ignore_index=ignore_index, class_weights=class_weights)
        self.dice_w = float(dice_w)
        self.ce_w   = float(ce_w)
        self.class_weights = class_weights
        self.ignore_index  = ignore_index
    def forward(self, pred, target):
        ld  = self.dice(pred, target)
        weight = self.class_weights.to(pred.device) if self.class_weights is not None else None
        if self.ignore_index is None:
            lce = F.cross_entropy(pred, target, weight=weight)
        else:
            lce = F.cross_entropy(pred, target, weight=weight, ignore_index=self.ignore_index)
        return self.dice_w * ld + self.ce_w * lce

# ================== Size Guards (no dataset edit needed) ==================
def resize_batch_to(imgs, masks, size_hw):
    """Force images & masks to (H,W)=size_hw inside the training loop."""
    Ht, Wt = size_hw
    # images: [B,C,H,W] float
    if imgs.shape[-2:] != (Ht, Wt):
        imgs = F.interpolate(imgs, size=(Ht, Wt), mode='bilinear', align_corners=False)
    # masks: [B,H,W] long -> resize via nearest on a channel
    if masks.shape[-2:] != (Ht, Wt):
        masks_f = masks.unsqueeze(1).float()
        masks_r = F.interpolate(masks_f, size=(Ht, Wt), mode='nearest')
        masks   = masks_r.squeeze(1).long()
    return imgs, masks

# ======================= Train / Eval =======================
def train_one_epoch(model, loader, optimizer, criterion, device, img_size):
    model.train()
    running_loss = 0.0
    for imgs, masks in loader:
        imgs, masks = imgs.to(device), masks.to(device)
        imgs, masks = resize_batch_to(imgs, masks, (img_size, img_size))
        optimizer.zero_grad()
        logits = model(imgs)                   # [B,C,H,W]
        loss = criterion(logits, masks)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    return running_loss / len(loader)

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
    return running_loss / len(loader), per_class_dice, mDice, per_class_iou, mIoU, cm

# ======================= Main =======================
def main():
    device = DEVICE
    ensure_dir(RESULTS_DIR)

    # --- dataset & loaders ---
    # If your CAMUS_loader supports img_size/in_channels, you can pass them here:
    # train_ds = CAMUS_loader(DATA_DIR, TRAIN_LIST, view=VIEW, img_size=IMG_SIZE, in_channels=IN_CHANNELS)
    # test_ds  = CAMUS_loader(DATA_DIR, TEST_LIST,  view=VIEW, img_size=IMG_SIZE, in_channels=IN_CHANNELS)
    # Otherwise we resize inside the loop (see resize_batch_to).
    train_ds = CAMUSPreprocessor(DATA_DIR, TRAIN_LIST, view=VIEW)
    test_ds  = CAMUSPreprocessor(DATA_DIR, TEST_LIST,  view=VIEW)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  num_workers=4, pin_memory=True)
    test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)

    # --- model (TransUNet only) ---
    # If your constructor uses different arg names, adjust them here.
    model = TransUNetLite(
        in_channels=IN_CHANNELS,
        num_classes=NUM_CLASSES,
        img_size=IMG_SIZE
    ).to(device)

    # --- loss, optim, sched ---
    class_weights = None
    criterion = DiceCELoss(dice_w=0.5, ce_w=0.5,
                           class_weights=class_weights,
                           ignore_index=None,
                           smooth=1.0)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=STEP_SIZE, gamma=GAMMA)

    best_mdice = 0.0
    patience, bad = 20, 0

    for epoch in range(EPOCHS):
        tr_loss = train_one_epoch(model, train_loader, optimizer, criterion, device, IMG_SIZE)
        curr_lr = optimizer.param_groups[0]['lr']

        te_loss, per_cls_dice, mDice, per_cls_iou, mIoU, cm = evaluate(
            model, test_loader, criterion, device, NUM_CLASSES,
            save_dir=SAVE_ROOT, epoch=epoch+1, save_n=SAVE_N, img_size=IMG_SIZE
        )

        print(
            f"[TransUNet] Epoch {epoch+1:03d} | "
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
            torch.save(model.state_dict(), f"best_transunet_dicece_mdice_{best_mdice:.4f}.pt")
        else:
            bad += 1
            if bad >= patience:
                print("Early stopping.")
                break

    print(f"\n✅ Metrics logged to: {METRICS_CSV}")
    print(f"   Confusion matrices: {RESULTS_DIR}/confusion_matrix_epoch_XXX.csv")

if __name__ == "__main__":
    main()
