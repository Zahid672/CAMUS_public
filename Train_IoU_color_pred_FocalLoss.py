import os
import numpy as np
from PIL import Image

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.nn.functional as F

import csv

# --- import your classes ---
from dataset import CAMUS_loader
# from Unet import UNet
from Attention_Unet import UNet
# from Trans_Unet import TransUNetLite  # if you want the transformer version

# --------- visualization palette (edit colors if you like) ----------
PALETTE = {
    0: (0, 0, 0),         # background
    1: (255, 0, 0),       # class 1
    2: (0, 255, 0),       # class 2
    3: (0, 0, 255),       # class 3
}

# ---------------- metrics & utils ----------------
def tensor_to_uint8_image(img_t):  # convert tensor to uint8 image
    """img_t: [C,H,W], returns uint8 RGB [H,W,3]"""
    img = img_t.detach().cpu()
    if img.dim() != 3:
        raise ValueError(f"Expected 3D tensor [C,H,W], got {img.shape}")
    if img.size(0) == 1:
        img = img.repeat(3, 1, 1)
    img = img.numpy().transpose(1, 2, 0)  # [H,W,3]
    mn, mx = img.min(), img.max()
    if mx - mn < 1e-8:    # avoid divide by zero
        img = np.zeros_like(img)
    else:
        # robust rescale 0..1 if outside range
        if mn < 0.0 or mx > 1.0:
            img = (img - mn) / (mx - mn + 1e-8)
    img = (img * 255.0).clip(0, 255).astype(np.uint8)
    return img

def mask_to_color(mask_hw, palette=PALETTE):  # convert mask to color image
    """mask_hw: [H,W] int -> color RGB [H,W,3] uint8"""
    mask = mask_hw.detach().cpu().numpy().astype(np.int64)
    h, w = mask.shape
    color = np.zeros((h, w, 3), dtype=np.uint8)
    for c, rgb in palette.items():
        color[mask == c] = rgb
    return color

def overlay_image(base_rgb, mask_rgb, alpha=0.5):  # overlay image with mask
    """alpha blend mask on image; inputs uint8 [H,W,3]"""
    base = base_rgb.astype(np.float32)
    mask = mask_rgb.astype(np.float32)
    out = (1 - alpha) * base + alpha * mask
    return out.clip(0, 255).astype(np.uint8)

def save_visuals(img_t, pred_hw, gt_hw, out_dir, name, alpha=0.45):  # save image, pred, gt, overlay
    os.makedirs(out_dir, exist_ok=True)
    img_rgb  = tensor_to_uint8_image(img_t)             # [H,W,3]
    pred_rgb = mask_to_color(pred_hw)                   # [H,W,3]
    gt_rgb   = mask_to_color(gt_hw)

    Image.fromarray(pred_rgb).save(os.path.join(out_dir, f"{name}_pred.png"))  # prediction
    Image.fromarray(gt_rgb).save(os.path.join(out_dir, f"{name}_gt.png"))      # ground truth
    Image.fromarray(img_rgb).save(os.path.join(out_dir, f"{name}_img.png"))    # original image

    # overlay (pred on image)
    over_pred = overlay_image(img_rgb, pred_rgb, alpha=alpha)
    Image.fromarray(over_pred).save(os.path.join(out_dir, f"{name}_overlay_pred.png"))

def update_confusion_matrix(cm, preds_hw, target_hw, num_classes):
    # preds_hw/target_hw: [B,H,W] Long
    with torch.no_grad():
        k = (target_hw * num_classes + preds_hw).view(-1)
        binc = torch.bincount(k, minlength=num_classes**2)
        cm += binc.reshape(num_classes, num_classes).to(cm.dtype)
    return cm

def dice_iou_from_cm(cm, eps=1e-6):
    # cm: [C,C] on CPU
    tp = cm.diag()  # true positives
    fp = cm.sum(0) - tp
    fn = cm.sum(1) - tp

    dice = (2 * tp + eps) / (2 * tp + fp + fn + eps)
    iou  = (tp + eps) / (tp + fp + fn + eps)

    mdice = dice.mean().item()
    miou  = iou.mean().item()
    return dice.tolist(), mdice, iou.tolist(), miou

# --------------- (optional) keep your dice-by-batch if you want ---------------
def dice_per_class(logits, target, num_classes):
    with torch.no_grad():
        preds = torch.argmax(logits, dim=1)  # [B,H,W]
        dices = []
        eps = 1e-6
        for c in range(num_classes):
            pred_c = (preds == c).float()
            targ_c = (target == c).float()
            inter  = (pred_c * targ_c).sum()
            denom  = pred_c.sum() + targ_c.sum()
            dice   = (2 * inter + eps) / (denom + eps)
            dices.append(dice.item())
        return dices, sum(dices) / num_classes

# ---------------- Multi-class Focal Loss (logits in) ----------------
class FocalLoss(nn.Module):
    """
    Multi-class Focal Loss for segmentation (logits in).
    pred:   [B,C,H,W] logits
    target: [B,H,W]   int in 0..C-1
    """
    def __init__(self, gamma=2.0, alpha=None, ignore_index=None, reduction='mean'):
        super().__init__()
        self.gamma = float(gamma)
        if alpha is not None and not torch.is_tensor(alpha):
            alpha = torch.tensor(alpha, dtype=torch.float32)
        self.alpha = alpha                 # tensor [C] or None
        self.ignore_index = ignore_index   # int or None
        self.reduction = reduction

    def forward(self, pred, target):
        # Ensure class weights (alpha) on the right device/dtype
        weight = None
        if self.alpha is not None:
            weight = self.alpha.to(pred.device, dtype=pred.dtype)

        # Cross-entropy per-pixel (no reduction)
        if self.ignore_index is None:
            ce = F.cross_entropy(pred, target, weight=weight, reduction='none')
        else:
            ce = F.cross_entropy(pred, target, weight=weight,
                                 ignore_index=self.ignore_index, reduction='none')

        # p_t = exp(-CE); focal scaling
        pt = torch.exp(-ce)
        loss = (1.0 - pt) ** self.gamma * ce  # [B,H,W]

        if self.ignore_index is not None:
            valid = (target != self.ignore_index).float()
            loss = loss * valid

        if self.reduction == 'mean':
            if self.ignore_index is not None:
                denom = (target != self.ignore_index).float().sum().clamp_min(1.0)
            else:
                denom = loss.numel()
            return loss.sum() / denom
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss


# ---------------- training / eval loops ----------------
def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    running_loss = 0.0
    for imgs, masks in loader:
        imgs  = imgs.to(device)
        masks = masks.to(device)               # [B,H,W], long
        optimizer.zero_grad()
        logits = model(imgs)                   # [B,C,H,W]
        loss = criterion(logits, masks)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    return running_loss / len(loader)

@torch.no_grad()
def evaluate(model, loader, criterion, device, num_classes, save_dir=None, epoch=None, save_n=8):
    model.eval()
    running_loss = 0.0
    cm = torch.zeros(num_classes, num_classes, dtype=torch.double)

    saved = 0
    out_dir = None
    if save_dir is not None and epoch is not None and save_n > 0:
        out_dir = os.path.join(save_dir, f"epoch_{epoch:03d}")
        os.makedirs(out_dir, exist_ok=True)

    for bidx, (imgs, masks) in enumerate(loader):
        imgs  = imgs.to(device)
        masks = masks.to(device)

        logits = model(imgs)
        loss = criterion(logits, masks)
        running_loss += loss.item()

        preds = torch.argmax(logits, dim=1)
        # update cm
        k = (masks.cpu() * num_classes + preds.cpu()).view(-1)
        binc = torch.bincount(k, minlength=num_classes**2)
        cm += binc.reshape(num_classes, num_classes).to(cm.dtype)

        # (optional) save a few visuals ...
        if out_dir is not None and saved < save_n:
            B = imgs.size(0)
            take = min(B, save_n - saved)
            for i in range(take):
                name = f"b{bidx}_i{i}"
                save_visuals(imgs[i], preds[i], masks[i], out_dir, name, alpha=0.45)
            saved += take

    # compute dataset-level metrics from cm
    tp = cm.diag()
    fp = cm.sum(0) - tp
    fn = cm.sum(1) - tp
    eps = 1e-6

    per_class_dice = ((2*tp + eps) / (2*tp + fp + fn + eps)).tolist()
    mDice = float(cm.new_tensor(per_class_dice).mean().item())
    per_class_iou  = ((tp + eps) / (tp + fp + fn + eps)).tolist()
    mIoU = float(cm.new_tensor(per_class_iou).mean().item())

    return running_loss / len(loader), per_class_dice, mDice, per_class_iou, mIoU, cm

# ----------------------------- helpers to save metrics/CM -----------------------------
def ensure_dir(p):
    os.makedirs(p, exist_ok=True)
    return p

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

# ----------------------------- main -----------------------------
def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    num_classes = 4
    view = '2CH'  # or '4CH'

    # --- dataset & loaders ---
    data_dir = 'database_nifti'
    split_dir = 'prepared_data'
    train_list = os.path.join(split_dir, 'train_samples.npy')
    test_list  = os.path.join(split_dir, 'test_ED.npy')  # or test_ES.npy / combined

    train_ds = CAMUS_loader(data_dir, train_list, view=view)
    test_ds  = CAMUS_loader(data_dir, test_list,  view=view)

    train_loader = DataLoader(train_ds, batch_size=4, shuffle=True,  num_workers=4, pin_memory=True)
    test_loader  = DataLoader(test_ds,  batch_size=4, shuffle=False, num_workers=4, pin_memory=True)

    # -------- model, loss, optim, sched --------
    model = UNet().to(device)  # ensure UNet.in/out channels match (grayscale->1 in, 4 out)

    # Optional per-class alpha weights to counter class imbalance (order: ids 0..C-1)
    alpha = None
    # Example if you know pixel frequencies per class:
    # freq = torch.tensor([f_bg, f_lv_endo, f_lv_myo, f_la], dtype=torch.float32)
    # alpha = (1.0 / (freq + 1e-8)); alpha = alpha / alpha.mean()

    criterion = FocalLoss(gamma=2.0, alpha=None, ignore_index=-100, reduction='mean')

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

    results_dir = ensure_dir("Attention_Unet_results_FocalLoss")  # where metrics + CMs go
    metrics_csv = os.path.join(results_dir, "Attention_Unet_metrics_FocalLoss.csv")
    save_root   = "qualitative_Attention_Unet_FocalLoss"         # where images go

    best_mdice = 0.0
    patience, bad = 20, 0

    for epoch in range(60):
        tr_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)

        # log LR before stepping the scheduler
        curr_lr = optimizer.param_groups[0]['lr']

        te_loss, per_cls_dice, mDice, per_cls_iou, mIoU, cm = evaluate(
            model, test_loader, criterion, device, num_classes,
            save_dir=save_root, epoch=epoch+1, save_n=8
        )

        print(
            f"Epoch {epoch+1:03d} | "
            f"train loss {tr_loss:.4f} | val loss {te_loss:.4f} | "
            f"mDice {mDice:.4f} | mIoU {mIoU:.4f} | "
            f"Dice per-class {[f'{d:.3f}' for d in per_cls_dice]} | "
            f"IoU per-class  {[f'{i:.3f}' for i in per_cls_iou]} | "
            f"LR {curr_lr:.2e}"
        )

        # save quantitative results
        log_metrics_csv(metrics_csv, epoch+1, tr_loss, te_loss, mDice, mIoU,
                        per_cls_dice, per_cls_iou, curr_lr)

        cm_path = os.path.join(results_dir, f"confusion_matrix_epoch_{epoch+1:03d}.csv")
        save_confusion_matrix(cm, cm_path)

        # step scheduler AFTER logging (so LR printed reflects pre-step)
        scheduler.step()

        # early stopping on mean Dice
        if mDice > best_mdice:
            best_mdice = mDice
            bad = 0
            torch.save(model.state_dict(), f"best_unet_focal_mdice_{best_mdice:.4f}.pt")
        else:
            bad += 1
            if bad >= patience:
                print("Early stopping.")
                break

    print(f"\n✅ Quantitative metrics logged to: {metrics_csv}")
    print(f"   Confusion matrices saved under: {results_dir}/confusion_matrix_epoch_XXX.csv")


if __name__ == "__main__":
    main()
