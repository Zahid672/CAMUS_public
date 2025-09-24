# train_from_png_ce_dice.py
import os
import csv
import time
import random
import argparse
from pathlib import Path
from typing import Tuple, Union, List

import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

# ---- local modules ----
from camus_png_dataset import CAMUSPNGSliceDataset
from AttentionUNet_CBAM_SE import UNet  # your model

# ===================== CLI =====================
def get_args():
    ap = argparse.ArgumentParser("Train Attention U-Net (CBAM+SE) on CAMUS PNGs with CE+Dice")
    ap.add_argument("--root", type=str, default="camus_png", help="PNG root folder")
    ap.add_argument("--manifest", type=str, default="manifest.csv", help="manifest filename inside root")
    ap.add_argument("--img", type=int, default=256, help="resize square side (None keeps original)")
    ap.add_argument("--bs", type=int, default=8, help="batch size")
    ap.add_argument("--epochs", type=int, default=60, help="epochs")
    ap.add_argument("--lr", type=float, default=1e-4, help="learning rate")
    ap.add_argument("--wd", type=float, default=1e-4, help="weight decay")
    ap.add_argument("--step", type=int, default=10, help="StepLR step_size")
    ap.add_argument("--gamma", type=float, default=0.1, help="StepLR gamma")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--num_workers", type=int, default=0)
    ap.add_argument("--save_n", type=int, default=8, help="save up to N visuals / epoch")
    ap.add_argument("--results", type=str, default="AttUNet_CBAM_SE_results_CE_DICE")
    ap.add_argument("--qual", type=str, default="qualitative_AttUNet_CBAM_SE_CE_DICE")
    ap.add_argument("--ce_w", type=float, default=0.5, help="weight for CE")
    ap.add_argument("--dice_w", type=float, default=0.5, help="weight for Dice")
    ap.add_argument("--train_frac", type=float, default=0.8, help="train fraction by patient folders")
    return ap.parse_args()

# ===================== Utils =====================
def set_seed(seed=42):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True

def ensure_dir(p):
    os.makedirs(p, exist_ok=True); return p

PALETTE = {0:(0,0,0), 1:(255,0,0), 2:(0,255,0), 3:(0,0,255)}

def tensor_to_uint8_image(img_t):
    img = img_t.detach().cpu()
    if img.dim() != 3: raise ValueError(f"Expected [C,H,W], got {img.shape}")
    if img.size(0) == 1: img = img.repeat(3, 1, 1)
    img = img.numpy().transpose(1, 2, 0)
    mn, mx = img.min(), img.max()
    if mx - mn < 1e-8: img = np.zeros_like(img)
    else:
        if mn < 0.0 or mx > 1.0: img = (img - mn) / (mx - mn + 1e-8)
    return (img * 255.0).clip(0, 255).astype(np.uint8)

def mask_to_color(mask_hw, palette=PALETTE):
    m = mask_hw.detach().cpu().numpy().astype(np.int64)
    h, w = m.shape
    out = np.zeros((h, w, 3), dtype=np.uint8)
    for c, rgb in palette.items(): out[m == c] = rgb
    return out

def overlay_image(base_rgb, mask_rgb, alpha=0.45):
    base = base_rgb.astype(np.float32); mask = mask_rgb.astype(np.float32)
    return ((1 - alpha) * base + alpha * mask).clip(0, 255).astype(np.uint8)

def save_visuals(img_t, pred_hw, gt_hw, out_dir, name, alpha=0.45):
    ensure_dir(out_dir)
    img_rgb  = tensor_to_uint8_image(img_t)
    pred_rgb = mask_to_color(pred_hw)
    gt_rgb   = mask_to_color(gt_hw)
    Image.fromarray(pred_rgb).save(os.path.join(out_dir, f"{name}_pred.png"))
    Image.fromarray(gt_rgb).save(os.path.join(out_dir, f"{name}_gt.png"))
    Image.fromarray(img_rgb).save(os.path.join(out_dir, f"{name}_img.png"))
    Image.fromarray(overlay_image(img_rgb, pred_rgb, alpha)).save(os.path.join(out_dir, f"{name}_overlay_pred.png"))

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

# ===================== Metrics =====================
def evaluate_cm_stats(cm, eps=1e-6):
    tp = cm.diag()
    fp = cm.sum(0) - tp
    fn = cm.sum(1) - tp
    per_class_dice = ((2*tp + eps) / (2*tp + fp + fn + eps)).tolist()
    mDice = float(cm.new_tensor(per_class_dice).mean().item())
    per_class_iou  = ((tp + eps) / (tp + fp + fn + eps)).tolist()
    mIoU = float(cm.new_tensor(per_class_iou).mean().item())
    return per_class_dice, mDice, per_class_iou, mIoU

def main_logits_from(output):
    return output[0] if isinstance(output, (tuple, list)) else output

# ===================== Losses =====================
class SoftDiceLoss(nn.Module):
    def __init__(self, smooth: float = 1.0, ignore_index: int = None):
        super().__init__()
        self.smooth = float(smooth)
        self.ignore_index = ignore_index
    def forward(self, logits, target_hw):
        B, C, H, W = logits.shape
        probs = F.softmax(logits, dim=1)
        tgt = torch.zeros((B, C, H, W), device=logits.device, dtype=probs.dtype)
        tgt.scatter_(1, target_hw.unsqueeze(1), 1.0)
        if self.ignore_index is not None:
            valid = (target_hw != self.ignore_index).float().unsqueeze(1)
            probs = probs * valid
            tgt   = tgt * valid
        dims = (0, 2, 3)
        inter = torch.sum(probs * tgt, dims)
        card  = torch.sum(probs + tgt, dims)
        dice_c = (2. * inter + self.smooth) / (card + self.smooth)
        return 1.0 - dice_c.mean()

class CEDiceLoss(nn.Module):
    def __init__(self, ce_w=0.5, dice_w=0.5, class_weights=None, ignore_index=-100, dice_smooth=1.0):
        super().__init__()
        self.ce_w, self.dice_w = float(ce_w), float(dice_w)
        self.ce   = nn.CrossEntropyLoss(weight=class_weights, ignore_index=ignore_index)
        self.dice = SoftDiceLoss(smooth=dice_smooth, ignore_index=ignore_index)
    def forward(self, logits, targets):
        return self.ce_w * self.ce(logits, targets) + self.dice_w * self.dice(logits, targets)

# ===================== Train / Eval =====================
def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    running = 0.0
    for imgs, masks in loader:
        imgs, masks = imgs.to(device), masks.to(device)
        optimizer.zero_grad(set_to_none=True)
        logits = main_logits_from(model(imgs))
        if logits.shape[-2:] != masks.shape[-2:]:
            logits = F.interpolate(logits, size=masks.shape[-2:], mode='bilinear', align_corners=False)
        loss = criterion(logits, masks)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        running += loss.item()
    return running / max(1, len(loader))

@torch.no_grad()
def evaluate(model, loader, criterion, device, num_classes, save_dir=None, epoch=None, save_n=8):
    model.eval()
    running = 0.0
    cm = torch.zeros(num_classes, num_classes, dtype=torch.double)
    saved = 0
    out_dir = None
    if save_dir is not None and epoch is not None and save_n > 0:
        out_dir = os.path.join(save_dir, f"epoch_{epoch:03d}"); ensure_dir(out_dir)
    for bidx, (imgs, masks) in enumerate(loader):
        imgs, masks = imgs.to(device), masks.to(device)
        logits = main_logits_from(model(imgs))
        if logits.shape[-2:] != masks.shape[-2:]:
            logits = F.interpolate(logits, size=masks.shape[-2:], mode='bilinear', align_corners=False)
        loss = criterion(logits, masks)
        running += loss.item()
        preds = torch.argmax(logits, dim=1)
        k = (masks.cpu() * num_classes + preds.cpu()).view(-1)
        binc = torch.bincount(k, minlength=num_classes**2)
        cm += binc.reshape(num_classes, num_classes).to(cm.dtype)
        if out_dir is not None and saved < save_n:
            B = imgs.size(0); take = min(B, save_n - saved)
            for i in range(take):
                save_visuals(imgs[i], preds[i], masks[i], out_dir, name=f"b{bidx}_i{i}")
            saved += take
    per_class_dice, mDice, per_class_iou, mIoU = evaluate_cm_stats(cm)
    return running / max(1, len(loader)), per_class_dice, mDice, per_class_iou, mIoU, cm

# ===================== Split by patient =====================
def split_indices_by_patient(ds: CAMUSPNGSliceDataset, train_frac=0.8, seed=42):
    """
    Group items by patient folder, split at patient level.
    Assumes each pair lives under root/patientXXXX/...
    """
    rng = random.Random(seed)
    buckets = {}
    for idx, (img_path, mask_path) in enumerate(ds.items):
        patient = Path(img_path).parent.name  # e.g., patient0001
        buckets.setdefault(patient, []).append(idx)

    patients = list(buckets.keys())
    rng.shuffle(patients)
    n_train = int(len(patients) * train_frac)
    train_pat = set(patients[:n_train])

    train_idx, val_idx = [], []
    for p, idxs in buckets.items():
        (train_idx if p in train_pat else val_idx).extend(idxs)
    return train_idx, val_idx

# ===================== Main =====================
def main():
    args = get_args()
    set_seed(args.seed)
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    ensure_dir(args.results); ensure_dir(args.qual)

    # Dataset
    ds_full = CAMUSPNGSliceDataset(root=args.root, manifest=args.manifest,
                                   img_size=(args.img, args.img), verbose=True)

    if len(ds_full) == 0:
        print("No samples found. Aborting."); return

    tr_idx, va_idx = split_indices_by_patient(ds_full, train_frac=args.train_frac, seed=args.seed)
    train_ds = Subset(ds_full, tr_idx)
    val_ds   = Subset(ds_full, va_idx)

    print(f"Patients split → train: {len(tr_idx)} samples, val: {len(va_idx)} samples")

    train_loader = DataLoader(train_ds, batch_size=args.bs, shuffle=True,
                              num_workers=args.num_workers, pin_memory=True)
    val_loader   = DataLoader(val_ds, batch_size=args.bs, shuffle=False,
                              num_workers=args.num_workers, pin_memory=True)

    # Model
    model = UNet(
        in_channels=1,
        num_classes=4,
        enc_cbam_stages=(False, True, True, True, False),
        dec_cbam_stages=(True, True, True, False),
        gate_use_cbam=True,
        bottleneck_se=True,
        cbam_reduction=16,
        se_reduction=16
    ).to(DEVICE)

    # Loss: CE + Dice (no focal)
    criterion = CEDiceLoss(
        ce_w=args.ce_w, dice_w=args.dice_w,
        class_weights=None, ignore_index=-100, dice_smooth=1.0
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.step, gamma=args.gamma)

    best_mdice = 0.0
    patience, bad = 20, 0
    t0 = time.time()
    metrics_csv = os.path.join(args.results, "metrics_ce_dice.csv")

    for epoch in range(args.epochs):
        tr_loss = train_one_epoch(model, train_loader, optimizer, criterion, DEVICE)
        curr_lr = optimizer.param_groups[0]['lr']

        te_loss, per_cls_dice, mDice, per_cls_iou, mIoU, cm = evaluate(
            model, val_loader, criterion, DEVICE, num_classes=4,
            save_dir=args.qual, epoch=epoch+1, save_n=args.save_n
        )

        print(
            f"[AttUNet-CBAM-SE CE+Dice] Epoch {epoch+1:03d} | "
            f"train {tr_loss:.4f} | val {te_loss:.4f} | "
            f"mDice {mDice:.4f} | mIoU {mIoU:.4f} | "
            f"Dice {['%.3f'%d for d in per_cls_dice]} | "
            f"IoU  {['%.3f'%i for i in per_cls_iou]} | "
            f"LR {curr_lr:.2e}"
        )

        log_metrics_csv(metrics_csv, epoch+1, tr_loss, te_loss, mDice, mIoU,
                        per_cls_dice, per_cls_iou, curr_lr)

        cm_path = os.path.join(args.results, f"confusion_matrix_epoch_{epoch+1:03d}.csv")
        np.savetxt(cm_path, np.asarray(cm.cpu(), dtype=np.int64), fmt='%d', delimiter=',')

        scheduler.step()

        if mDice > best_mdice:
            best_mdice = mDice; bad = 0
            torch.save(model.state_dict(), f"best_attunet_cbam_se_ce_dice_mdice_{best_mdice:.4f}.pt")
        else:
            bad += 1
            if bad >= patience:
                print("Early stopping."); break

    dt = time.time() - t0
    print(f"\n✅ Metrics logged to: {metrics_csv}")
    print(f"   Confusion matrices: {args.results}/confusion_matrix_epoch_XXX.csv")
    print(f"   Best mDice: {best_mdice:.4f} | Total time: {dt/60.1:.1f} min")

if __name__ == "__main__":
    main()
