# train_png_unet_ce.py
"""
Train plain U-Net (your Unet.py) on the preprocessed CAMUS PNG dataset
using Cross-Entropy loss only, and save EVERYTHING neatly under:

  UNet_Results_on_Preprocess_PNG_CAMUS_Dataset/
    ├─ metrics_ce.csv
    ├─ checkpoints/
    │    ├─ last.pt
    │    └─ best_unet_ce_mdice_XXXX.pt
    ├─ confusion_matrices/
    │    └─ confusion_matrix_epoch_XXX.csv
    └─ qualitative_UNet_CE/
         └─ epoch_XXX/
              ├─ b0_i0_img.png
              ├─ b0_i0_pred.png
              ├─ b0_i0_gt.png
              └─ b0_i0_overlay_pred.png

Run (example):
  python train_png_unet_ce.py --root camus_png --manifest manifest.csv --img 256 --bs 8 --epochs 60
"""

import os
import csv
import time
import random
import argparse
from pathlib import Path
from typing import List

import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

# --- your local modules ---
from Preprocess_CAMUS_PNG_dataset import CAMUSPNGSliceDataset
from Unet import UNet  # NOTE: your UNet() takes no args and is hard-coded for 1 input ch / 4 classes

# ===================== Fixed result locations =====================
RESULTS_DIR = "UNet_Results_on_Preprocess_PNG_CAMUS_Dataset"
METRICS_CSV = os.path.join(RESULTS_DIR, "metrics_ce.csv")
QUAL_DIR    = os.path.join(RESULTS_DIR, "qualitative_UNet_CE")
CKPT_DIR    = os.path.join(RESULTS_DIR, "checkpoints")
CM_DIR      = os.path.join(RESULTS_DIR, "confusion_matrices")

# ===================== CLI =====================
def get_args():
    ap = argparse.ArgumentParser("Train U-Net on CAMUS PNGs (Cross-Entropy only)")
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
    ap.add_argument("--train_frac", type=float, default=0.8, help="train fraction by patient folders")
    return ap.parse_args()

# ===================== Utils & Viz =====================
PALETTE = {0:(0,0,0), 1:(255,0,0), 2:(0,255,0), 3:(0,0,255)}

def set_seed(seed=42):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True

def ensure_dir(p):
    os.makedirs(p, exist_ok=True); return p

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
        patient = Path(img_path).parent.name
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

    # ensure all output dirs
    ensure_dir(RESULTS_DIR)
    ensure_dir(QUAL_DIR)
    ensure_dir(CKPT_DIR)
    ensure_dir(CM_DIR)

    # dataset
    ds_full = CAMUSPNGSliceDataset(root=args.root, manifest=args.manifest,
                                   img_size=(args.img, args.img), verbose=True)
    if len(ds_full) == 0:
        print("No samples found. Aborting."); return

    # split by patients (hold-out)
    tr_idx, va_idx = split_indices_by_patient(ds_full, train_frac=args.train_frac, seed=args.seed)
    train_ds = Subset(ds_full, tr_idx)
    val_ds   = Subset(ds_full, va_idx)

    print(f"Patients split → train: {len(tr_idx)} samples, val: {len(va_idx)} samples")

    train_loader = DataLoader(train_ds, batch_size=args.bs, shuffle=True,
                              num_workers=args.num_workers, pin_memory=True)
    val_loader   = DataLoader(val_ds, batch_size=args.bs, shuffle=False,
                              num_workers=args.num_workers, pin_memory=True)

    # -------- Model --------
    # Your Unet.UNet() has no args and is fixed to 1 input channel / 4 classes.
    model = UNet().to(DEVICE)

    # -------- Loss: Cross-Entropy ONLY --------
    criterion = nn.CrossEntropyLoss(ignore_index=-100)

    # -------- Optim & Sched --------
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.step, gamma=args.gamma)

    best_mdice = 0.0
    patience, bad = 20, 0
    t0 = time.time()

    for epoch in range(args.epochs):
        tr_loss = train_one_epoch(model, train_loader, optimizer, criterion, DEVICE)
        curr_lr = optimizer.param_groups[0]['lr']

        te_loss, per_cls_dice, mDice, per_cls_iou, mIoU, cm = evaluate(
            model, val_loader, criterion, DEVICE, num_classes=4,
            save_dir=QUAL_DIR, epoch=epoch+1, save_n=args.save_n
        )

        print(
            f"[UNet CE] Epoch {epoch+1:03d} | "
            f"train {tr_loss:.4f} | val {te_loss:.4f} | "
            f"mDice {mDice:.4f} | mIoU {mIoU:.4f} | "
            f"Dice {['%.3f'%d for d in per_cls_dice]} | "
            f"IoU  {['%.3f'%i for i in per_cls_iou]} | "
            f"LR {curr_lr:.2e}"
        )

        # ---- metrics csv ----
        log_metrics_csv(METRICS_CSV, epoch+1, tr_loss, te_loss, mDice, mIoU,
                        per_cls_dice, per_cls_iou, curr_lr)

        # ---- confusion matrix per epoch ----
        cm_path = os.path.join(CM_DIR, f"confusion_matrix_epoch_{epoch+1:03d}.csv")
        np.savetxt(cm_path, np.asarray(cm.cpu(), dtype=np.int64), fmt='%d', delimiter=',')

        # ---- checkpoints ----
        # save "last" every epoch
        torch.save(model.state_dict(), os.path.join(CKPT_DIR, "last.pt"))
        # save "best" by mean Dice
        if mDice > best_mdice:
            best_mdice = mDice; bad = 0
            best_path = os.path.join(CKPT_DIR, f"best_unet_ce_mdice_{best_mdice:.4f}.pt")
            torch.save(model.state_dict(), best_path)
        else:
            bad += 1
            if bad >= patience:
                print("Early stopping."); break

        scheduler.step()

    dt = time.time() - t0
    print(f"\n✅ Metrics logged to: {METRICS_CSV}")
    print(f"   Checkpoints: {CKPT_DIR}\\last.pt and best_*")
    print(f"   Confusion matrices: {CM_DIR}\\confusion_matrix_epoch_XXX.csv")
    print(f"   Visuals: {QUAL_DIR}\\epoch_XXX\\*.png")
    print(f"   Best mDice: {best_mdice:.4f} | Total time: {dt/60.1:.1f} min")

if __name__ == "__main__":
    main()
