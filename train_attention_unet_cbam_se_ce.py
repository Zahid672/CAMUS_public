# train_attention_unet_cbam_se_ce.py
import os
import csv
import time
from typing import Tuple, Union

import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

# --- your modules ---
from dataset import CAMUS_loader               # expects (data_dir, list.npy, view)
from AttentionUNet_CBAM_SE import UNet        # your CBAM+SE Attention U-Net

# ===================== Config =====================
NUM_CLASSES = 4
VIEW        = '2CH'           # or '4CH'
BATCH_SIZE  = 8
EPOCHS      = 60
LR          = 1e-4
WEIGHT_DECAY= 1e-4
STEP_SIZE   = 10
GAMMA       = 0.1
SAVE_N      = 8
NUM_WORKERS = 0 if os.name == "nt" else 4
SEED        = 42
DEVICE      = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Force-resize inside the loop (set to None to keep original)
IMG_SIZE    = 256

DATA_DIR    = 'database_nifti'
SPLIT_DIR   = 'prepared_data'
TRAIN_LIST  = os.path.join(SPLIT_DIR, 'train_samples.npy')
VAL_LIST    = os.path.join(SPLIT_DIR, 'test_ED.npy')  # or test_ES.npy

RESULTS_DIR = "AttUNet_CBAM_SE_results_CE"
METRICS_CSV = os.path.join(RESULTS_DIR, "metrics_ce.csv")
QUAL_DIR    = "qualitative_AttUNet_CBAM_SE_CE"

PALETTE = {0:(0,0,0), 1:(255,0,0), 2:(0,255,0), 3:(0,0,255)}
# ==================================================

# ---------------- utils ----------------
def set_seed(seed=42):
    import random
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True

def ensure_dir(p):
    os.makedirs(p, exist_ok=True); return p

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

# -------------- metrics from confusion matrix --------------
def evaluate_cm_stats(cm, eps=1e-6):
    tp = cm.diag()
    fp = cm.sum(0) - tp
    fn = cm.sum(1) - tp
    per_class_dice = ((2*tp + eps) / (2*tp + fp + fn + eps)).tolist()
    mDice = float(cm.new_tensor(per_class_dice).mean().item())
    per_class_iou  = ((tp + eps) / (tp + fp + fn + eps)).tolist()
    mIoU = float(cm.new_tensor(per_class_iou).mean().item())
    return per_class_dice, mDice, per_class_iou, mIoU

# -------------- size alignment --------------
def resize_imgs_masks(imgs, masks, size_hw):
    if size_hw is None:
        return imgs, masks
    Ht, Wt = size_hw
    if imgs.shape[-2:] != (Ht, Wt):
        imgs = F.interpolate(imgs, size=(Ht, Wt), mode='bilinear', align_corners=False)
    if masks.shape[-2:] != (Ht, Wt):
        masks = F.interpolate(masks.unsqueeze(1).float(), size=(Ht, Wt), mode='nearest').squeeze(1).long()
    return imgs, masks

# -------------- get main logits only --------------
def main_logits_from(output: Union[torch.Tensor, Tuple[torch.Tensor, ...]]) -> torch.Tensor:
    if isinstance(output, (tuple, list)):
        return output[0]
    return output

# -------------- train / eval loops (CE only) --------------
def train_one_epoch(model, loader, optimizer, criterion_ce, device, img_size):
    model.train()
    running = 0.0
    for imgs, masks in loader:
        imgs, masks = imgs.to(device), masks.to(device)
        imgs, masks = resize_imgs_masks(imgs, masks, (img_size, img_size) if img_size else None)
        optimizer.zero_grad(set_to_none=True)
        logits = main_logits_from(model(imgs))
        if logits.shape[-2:] != masks.shape[-2:]:
            logits = F.interpolate(logits, size=masks.shape[-2:], mode='bilinear', align_corners=False)
        loss = criterion_ce(logits, masks)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        running += loss.item()
    return running / max(1, len(loader))

@torch.no_grad()
def evaluate(model, loader, criterion_ce, device, num_classes, img_size,
             save_dir=None, epoch=None, save_n=8):
    model.eval()
    running = 0.0
    cm = torch.zeros(num_classes, num_classes, dtype=torch.double)
    saved = 0
    out_dir = None
    if save_dir is not None and epoch is not None and save_n > 0:
        out_dir = os.path.join(save_dir, f"epoch_{epoch:03d}"); ensure_dir(out_dir)

    for bidx, (imgs, masks) in enumerate(loader):
        imgs, masks = imgs.to(device), masks.to(device)
        imgs, masks = resize_imgs_masks(imgs, masks, (img_size, img_size) if img_size else None)
        logits = main_logits_from(model(imgs))
        if logits.shape[-2:] != masks.shape[-2:]:
            logits = F.interpolate(logits, size=masks.shape[-2:], mode='bilinear', align_corners=False)
        loss = criterion_ce(logits, masks)
        running += loss.item()

        preds = torch.argmax(logits, dim=1)
        k = (masks.cpu() * num_classes + preds.cpu()).view(-1)
        binc = torch.bincount(k, minlength=num_classes**2)
        cm += binc.reshape(num_classes, num_classes).to(cm.dtype)

        if out_dir is not None and saved < save_n:
            B = imgs.size(0); take = min(B, save_n - saved)
            for i in range(take):
                name = f"b{bidx}_i{i}"
                save_visuals(imgs[i], preds[i], masks[i], out_dir, name)
            saved += take

    per_class_dice, mDice, per_class_iou, mIoU = evaluate_cm_stats(cm)
    return running / max(1, len(loader)), per_class_dice, mDice, per_class_iou, mIoU, cm

# ---------------- main ----------------
def main():
    set_seed(SEED)
    ensure_dir(RESULTS_DIR)

    if not (os.path.isfile(TRAIN_LIST) and os.path.isfile(VAL_LIST)):
        raise FileNotFoundError(
            f"Missing split files:\n  {TRAIN_LIST}\n  {VAL_LIST}\n"
            "Create them first (your Prepare_CAMUS step)."
        )

    # Datasets/Loaders
    train_ds = CAMUS_loader(DATA_DIR, TRAIN_LIST, view=VIEW)
    val_ds   = CAMUS_loader(DATA_DIR, VAL_LIST,   view=VIEW)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=NUM_WORKERS, pin_memory=True)
    val_loader   = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False,
                              num_workers=NUM_WORKERS, pin_memory=True)

    # -------- Model: Attention U-Net with CBAM + SE --------
    model = UNet(
        in_channels=1,
        num_classes=NUM_CLASSES,
        # reasonable defaults for CAMUS:
        enc_cbam_stages=(False, True, True, True, False),
        dec_cbam_stages=(True, True, True, False),
        gate_use_cbam=True,
        bottleneck_se=True,
        cbam_reduction=16,
        se_reduction=16
    ).to(DEVICE)

    # -------- Loss: Cross-Entropy ONLY --------
    class_weights = None   # or torch.tensor([...], device=DEVICE)
    criterion_ce = nn.CrossEntropyLoss(weight=class_weights)

    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=STEP_SIZE, gamma=GAMMA)

    best_mdice = 0.0
    patience, bad = 20, 0
    t0 = time.time()

    for epoch in range(EPOCHS):
        tr_loss = train_one_epoch(model, train_loader, optimizer, criterion_ce, DEVICE, IMG_SIZE)
        curr_lr = optimizer.param_groups[0]['lr']

        te_loss, per_cls_dice, mDice, per_cls_iou, mIoU, cm = evaluate(
            model, val_loader, criterion_ce, DEVICE, NUM_CLASSES, IMG_SIZE,
            save_dir=QUAL_DIR, epoch=epoch+1, save_n=SAVE_N
        )

        print(
            f"[AttUNet-CBAM-SE + CE] Epoch {epoch+1:03d} | "
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
            torch.save(model.state_dict(), f"best_attunet_cbam_se_ce_mdice_{best_mdice:.4f}.pt")
        else:
            bad += 1
            if bad >= patience:
                print("Early stopping."); break

    dt = time.time() - t0
    print(f"\n✅ Metrics logged to: {METRICS_CSV}")
    print(f"   Confusion matrices: {RESULTS_DIR}/confusion_matrix_epoch_XXX.csv")
    print(f"   Best mDice: {best_mdice:.4f} | Total time: {dt/60.1:.1f} min")


if __name__ == "__main__":
    main()
