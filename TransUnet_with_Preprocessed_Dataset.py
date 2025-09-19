# camus_all_in_one.py
import os
import time
import csv
from pathlib import Path
import numpy as np
from PIL import Image
import cv2
import SimpleITK as sitk

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from torchvision.transforms import InterpolationMode

# -------------------------------------------------------
# Choose what to do:
#   "preprocess"         -> only build splits and write preprocessed .npz files
#   "train_from_nifti"   -> train directly from CAMUS NIfTI (dataset loader)
#   "train_from_npz"     -> train using .npz files produced by "preprocess"
RUN_MODE = "train_from_nifti"
# -------------------------------------------------------

# =============== PROJECT CONFIG ===============
DATA_DIR   = "database_nifti"
SPLIT_DIR  = "prepared_data"
OUT_DIR    = "preprocessed"     # for preprocess mode (.npz)
VIEW       = "2CH"              # or "4CH"

IMG_SIZE    = 256               # divisible by 16 for TransUNet
IN_CHANNELS = 1
NUM_CLASSES = 4
BATCH_SIZE  = 2

EPOCHS       = 60
LR           = 1e-4
WEIGHT_DECAY = 1e-4
STEP_SIZE    = 10
GAMMA        = 0.1
PATIENCE     = 20
SAVE_N       = 8
USE_AMP      = True

TAG         = "TransUNet"
RESULTS_DIR = f"{TAG}_results_DoubleLoss_Dice_CE"
METRICS_CSV = os.path.join(RESULTS_DIR, f"{TAG}_metrics_DoubleLoss_Dice_CE.csv")
SAVE_ROOT   = f"qualitative_{TAG}_DoubleLoss_Dice_CE"

DEVICE      = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DEFAULT_WORKERS = 0 if os.name == "nt" else 4  # safer default on Windows

PALETTE = {0:(0,0,0), 1:(255,0,0), 2:(0,255,0), 3:(0,0,255)}

# ================== MODEL (import your TransUNet) ==================
# Ensure Trans_Unet.py is in the same folder with class TransUNetLite(in_channels, num_classes, img_size)
from Trans_Unet import TransUNetLite


# ================== SHARED UTILS ==================
def ensure_dir(p):
    os.makedirs(p, exist_ok=True); return p

def tensor_to_uint8_image(img_t):
    img = img_t.detach().cpu()
    if img.dim() != 3: raise ValueError(f"Expected [C,H,W], got {img.shape}")
    if img.size(0) == 1: img = img.repeat(3,1,1)
    img = img.numpy().transpose(1,2,0)
    mn, mx = img.min(), img.max()
    if mx - mn < 1e-8: img = np.zeros_like(img)
    else:
        if mn < 0.0 or mx > 1.0: img = (img - mn) / (mx - mn + 1e-8)
    return (img*255.0).clip(0,255).astype(np.uint8)

def mask_to_color(mask_hw, palette=PALETTE):
    mask = mask_hw.detach().cpu().numpy().astype(np.int64)
    h,w = mask.shape
    out = np.zeros((h,w,3), dtype=np.uint8)
    for c,rgb in palette.items(): out[mask==c] = rgb
    return out

def overlay_image(base_rgb, mask_rgb, alpha=0.5):
    base = base_rgb.astype(np.float32); mask = mask_rgb.astype(np.float32)
    return ((1-alpha)*base + alpha*mask).clip(0,255).astype(np.uint8)

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
              + [f"iou_c{i}" for i in range(len(iou_list))])
    row = [epoch, f"{tr_loss:.6f}", f"{te_loss:.6f}", f"{mDice:.6f}", f"{mIoU:.6f}", f"{lr:.8f}"]
    row += [f"{d:.6f}" for d in dice_list] + [f"{i:.6f}" for i in iou_list]
    write_header = not os.path.exists(csv_path)
    with open(csv_path, "a", newline="") as f:
        w = csv.writer(f)
        if write_header: w.writerow(header)
        w.writerow(row)


# ================== LOSSES ==================
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
        self.smooth = smooth; self.ignore_index = ignore_index; self.class_weights = class_weights
    def forward(self, pred, target):
        C = pred.shape[1]
        prob = F.softmax(pred, dim=1)
        tgt  = one_hot(target, C, self.ignore_index)
        dims = (0,2,3)
        inter = torch.sum(prob * tgt, dims)
        card  = torch.sum(prob + tgt, dims)
        dice_c = (2*inter + self.smooth) / (card + self.smooth)
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
        ld  = self.dice(pred, target)
        weight = self.class_weights.to(pred.device) if self.class_weights is not None else None
        if self.ignore_index is None:
            lce = F.cross_entropy(pred, target, weight=weight)
        else:
            lce = F.cross_entropy(pred, target, weight=weight, ignore_index=self.ignore_index)
        return self.dice_w * ld + self.ce_w * lce


# ================== TRAIN/EVAL LOOPS ==================
def train_one_epoch(model, loader, optimizer, criterion, device, scaler=None):
    model.train(); running = 0.0
    for imgs, masks in loader:
        imgs  = imgs.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        if scaler is not None:
            with torch.amp.autocast(device_type="cuda"):
                logits = model(imgs); loss = criterion(logits, masks)
            scaler.scale(loss).backward(); scaler.step(optimizer); scaler.update()
        else:
            logits = model(imgs); loss = criterion(logits, masks)
            loss.backward(); optimizer.step()
        running += loss.item()
    return running / len(loader)

@torch.no_grad()
def evaluate(model, loader, criterion, device, num_classes, save_dir=None, epoch=None, save_n=8):
    model.eval(); running = 0.0
    cm = torch.zeros(num_classes, num_classes, dtype=torch.double)
    saved = 0; out_dir = None
    if save_dir is not None and epoch is not None and save_n > 0:
        out_dir = os.path.join(save_dir, f"epoch_{epoch:03d}"); os.makedirs(out_dir, exist_ok=True)

    for bidx, (imgs, masks) in enumerate(loader):
        imgs = imgs.to(device, non_blocking=True); masks = masks.to(device, non_blocking=True)
        logits = model(imgs); loss = criterion(logits, masks); running += loss.item()
        preds = torch.argmax(logits, dim=1)
        # update CM
        k = (masks.cpu() * num_classes + preds.cpu()).view(-1)
        binc = torch.bincount(k, minlength=num_classes**2); cm += binc.reshape(num_classes, num_classes).to(cm.dtype)
        # visuals
        if out_dir is not None and saved < save_n:
            B = imgs.size(0); take = min(B, save_n - saved)
            for i in range(take):
                save_visuals(imgs[i], preds[i], masks[i], out_dir, f"b{bidx}_i{i}", alpha=0.45)
            saved += take

    tp = cm.diag(); fp = cm.sum(0) - tp; fn = cm.sum(1) - tp; eps = 1e-6
    per_class_dice = ((2*tp + eps) / (2*tp + fp + fn + eps)).tolist()
    per_class_iou  = ((tp + eps) / (tp + fp + fn + eps)).tolist()
    mDice = float(cm.new_tensor(per_class_dice).mean().item())
    mIoU  = float(cm.new_tensor(per_class_iou).mean().item())
    return running / len(loader), per_class_dice, mDice, per_class_iou, mIoU, cm


# ================== PREPROCESSOR (CLASS) ==================
class CAMUSPreprocessor:
    def __init__(self, data_dir="database_nifti", split_dir="prepared_data", out_dir="preprocessed",
                 view="2CH", img_size=256, do_clahe=True, denoise="median", overwrite=False, seed=1234):
        self.data_dir=data_dir; self.split_dir=split_dir; self.out_dir=out_dir
        self.view=view; self.img_size=int(img_size)
        self.do_clahe=bool(do_clahe); self.denoise=denoise; self.overwrite=bool(overwrite); self.seed=int(seed)
        os.makedirs(self.split_dir, exist_ok=True); os.makedirs(self.out_dir, exist_ok=True)

    def build_splits_if_missing(self):
        train_p = self._p("train_samples.npy"); test_ed = self._p("test_ED.npy"); test_es = self._p("test_ES.npy")
        if all(Path(p).is_file() for p in [train_p, test_ed, test_es]):
            print("----- The split files already exist -----"); return
        high, med, low = self._partition_by_quality(self.data_dir)
        rng = np.random.default_rng(self.seed); rng.shuffle(high); rng.shuffle(med); rng.shuffle(low)
        h_s, m_s, l_s = 20, 20, 10
        htest, htrain = high[:h_s], high[h_s:]; mtest, mtrain = med[:m_s], med[m_s:]; ltest, ltrain = low[:l_s], low[l_s:]
        total_test  = np.array(htest + mtest + ltest, dtype=object)
        total_train = np.array(htrain + mtrain + ltrain, dtype=object)
        test_ED, test_ES = self._expand_rep(total_test, mode="separate")
        train_all        = self._expand_rep(total_train, mode="combined")
        np.save(train_p, np.array(train_all, dtype=object)); np.save(test_ed, np.array(test_ED, dtype=object)); np.save(test_es, np.array(test_ES, dtype=object))
        print("Splits created:", f"train={len(train_all)}, test_ED={len(test_ED)}, test_ES={len(test_ES)}")

    def preprocess_all(self):
        for split in ["train","test_ED","test_ES"]:
            self._preprocess_split(self._p(f"{split if split!='train' else 'train_samples'}.npy") if split=="train" else self._p(f"{split}.npy"), split)
        print("\nSummary:")
        for s in ["train","test_ED","test_ES"]:
            d=os.path.join(self.out_dir,s); n=len(os.listdir(d)) if os.path.isdir(d) else 0; print(f"  {s}: {n} files in {d}")

    # internals
    def _p(self, name): return os.path.join(self.split_dir, name)
    @staticmethod
    def _prepare_cfg(path):
        with open(path,"r") as f: lines=f.read().splitlines()
        out={}; 
        for ln in lines:
            if ": " in ln:
                k,v=ln.split(": ",1); out[k]=v
        return out
    def _partition_by_quality(self, data_dir):
        high,med,low=[],[],[]
        for patient in sorted(os.listdir(data_dir)):
            pdir=os.path.join(data_dir,patient)
            if not os.path.isdir(pdir): continue
            info=os.path.join(pdir,"Info_2CH.cfg")
            if not os.path.isfile(info):
                print(f"warn: missing Info_2CH.cfg for {patient}, skipping"); continue
            q=self._prepare_cfg(info).get("ImageQuality",None)
            (high if q=="Good" else med if q=="Medium" else low if q=="Poor" else []).append(patient) if q in {"Good","Medium","Poor"} else print(f"warn: unknown ImageQuality '{q}' for {patient}")
        print(f"Quality counts -> High:{len(high)} Medium:{len(med)} Low:{len(low)}"); return high,med,low
    @staticmethod
    def _expand_rep(arr, mode="combined"):
        if mode=="combined": return [[a,"ED"] for a in arr] + [[a,"ES"] for a in arr]
        ed,es=[],[]
        for a in arr: ed.append([a,"ED"]); es.append([a,"ES"])
        return ed,es
    @staticmethod
    def _sitk_read(fp): im=sitk.ReadImage(str(fp)); return np.squeeze(sitk.GetArrayFromImage(im))
    @staticmethod
    def _percentile_clip(img,p_low=1.0,p_high=99.0):
        lo,hi=np.percentile(img,[p_low,p_high]); 
        if hi<=lo: return img.astype(np.float32)
        img=np.clip(img,lo,hi); img=(img-lo)/(hi-lo+1e-8); return img.astype(np.float32)
    @staticmethod
    def _apply_clahe_8bit(img01,clip_limit=2.0,tile_grid_size=(8,8)):
        img8=(img01*255.0).astype(np.uint8); clahe=cv2.createCLAHE(clipLimit=clip_limit,tileGridSize=tile_grid_size)
        out8=clahe.apply(img8); return (out8.astype(np.float32)/255.0)
    @staticmethod
    def _despeckle(img01,method="median",ksize=3):
        if method=="median": out=cv2.medianBlur((img01*255).astype(np.uint8),ksize).astype(np.float32)/255.0
        elif method=="bilateral": out=cv2.bilateralFilter(img01.astype(np.float32),d=5,sigmaColor=25,sigmaSpace=5)
        else: out=img01
        return out
    def _robust_preprocess_us(self,img):
        if img.ndim==3: img=np.squeeze(img)
        img=self._percentile_clip(img,1.0,99.0)
        if self.do_clahe: img=self._apply_clahe_8bit(img,2.0,(8,8))
        img=self._despeckle(img,method=self.denoise,ksize=3); return img
    @staticmethod
    def _sanitize_mask(msk):
        msk=msk.astype(np.int64); valid=np.array([0,1,2,3],dtype=np.int64); okay=np.isin(msk,valid)
        return np.where(okay,msk,0).astype(np.int64)
    def _resize_image_mask(self,img01,msk):
        size=self.img_size
        img_r=cv2.resize((img01*255.0).astype(np.uint8),(size,size),interpolation=cv2.INTER_LINEAR).astype(np.float32)/255.0
        msk_r=cv2.resize(msk.astype(np.int32),(size,size),interpolation=cv2.INTER_NEAREST).astype(np.int64)
        return img_r, msk_r
    def _load_pair(self,patient,instant):
        ip=os.path.join(self.data_dir,patient,f"{patient}_{self.view}_{instant}.nii.gz")
        mp=os.path.join(self.data_dir,patient,f"{patient}_{self.view}_{instant}_gt.nii.gz")
        if not (os.path.isfile(ip) and os.path.isfile(mp)): raise FileNotFoundError(f"Missing image/mask for {patient} {self.view} {instant}")
        return self._sitk_read(ip), self._sitk_read(mp)
    @staticmethod
    def _sid(p,i): return f"{p}_{i}"
    def _already_done(self,split,p,i): return os.path.isfile(os.path.join(self.out_dir,split,f"{self._sid(p,i)}.npz"))
    def _save_npz(self,split,p,i,img01,msk):
        os.makedirs(os.path.join(self.out_dir,split),exist_ok=True)
        outp=os.path.join(self.out_dir,split,f"{self._sid(p,i)}.npz")
        np.savez_compressed(outp,image=img01.astype(np.float32),mask=msk.astype(np.uint8),meta=np.array([p,i],dtype=object)); return outp
    def _preprocess_split(self,split_path,split_name):
        items=np.load(split_path,allow_pickle=True); n=len(items)
        print(f"\n==> Preprocessing split '{split_name}' ({n} samples) ..."); ok=fail=0; t0=time.time()
        for k,(p,i) in enumerate(items):
            try:
                if (not self.overwrite) and self._already_done(split_name,p,i):
                    ok+=1; 
                    if (k+1)%50==0: print(f"[{split_name}] {k+1}/{n} skipped ...")
                    continue
                img,msk=self._load_pair(p,i)
                img=self._robust_preprocess_us(img); msk=self._sanitize_mask(msk)
                img,msk=self._resize_image_mask(img,msk)
                self._save_npz(split_name,p,i,img,msk); ok+=1
                if (k+1)%50==0: print(f"[{split_name}] {k+1}/{n} processed ...")
            except Exception as e:
                fail+=1; print(f"[ERROR] {p} {i}: {e}")
        print(f"Done '{split_name}': success={ok}, failed={fail}, time={time.time()-t0:.1f}s")


# ================== DATASETS ==================
class ToFloat32(object):
    def __call__(self, t): return t.to(torch.float32)

class CAMUS_loader(Dataset):
    """Loads CAMUS directly from NIfTI (no .npz) with preprocessing & augment."""
    def __init__(self, data_path, patient_list_path, view, img_size=256, in_channels=1,
                 augment=False, use_clahe=True, denoise="median"):
        super().__init__()
        assert view in ["2CH","4CH"]
        self.data_path=data_path; self.view=view
        self.items=np.load(patient_list_path,allow_pickle=True)
        self.img_size=int(img_size); self.in_channels=int(in_channels)
        self.augment=bool(augment); self.use_clahe=bool(use_clahe); self.denoise=denoise
        target=(self.img_size,self.img_size)
        self.tf = {
            "image": transforms.Compose([
                transforms.ToTensor(),
                transforms.Resize(target, interpolation=InterpolationMode.BILINEAR),
                ToFloat32()  # picklable
            ]),
            "gt": transforms.Compose([
                transforms.ToTensor(),
                transforms.Resize(target, interpolation=InterpolationMode.NEAREST),
            ])
        }
    @staticmethod
    def sitk_read(fp): im=sitk.ReadImage(str(fp)); return np.squeeze(sitk.GetArrayFromImage(im))
    @staticmethod
    def percentile_clip(img,p_low=1.0,p_high=99.0):
        lo,hi=np.percentile(img,[p_low,p_high]); 
        if hi<=lo: return img.astype(np.float32)
        img=np.clip(img,lo,hi); img=(img-lo)/(hi-lo+1e-8); return img.astype(np.float32)
    @staticmethod
    def apply_clahe_8bit(img01,clip_limit=2.0,tile_grid_size=(8,8)):
        img8=(img01*255.0).astype(np.uint8); clahe=cv2.createCLAHE(clipLimit=clip_limit,tileGridSize=tile_grid_size)
        out8=clahe.apply(img8); return (out8.astype(np.float32)/255.0)
    @staticmethod
    def despeckle(img01,method="median",ksize=3):
        if method=="median": out=cv2.medianBlur((img01*255).astype(np.uint8),ksize).astype(np.float32)/255.0
        elif method=="bilateral": out=cv2.bilateralFilter(img01.astype(np.float32),d=5,sigmaColor=25,sigmaSpace=5)
        else: out=img01; return out
        return out
    def robust_preprocess_us(self,img):
        if img.ndim==3: img=np.squeeze(img)
        img=self.percentile_clip(img,1.0,99.0)
        if self.use_clahe: img=self.apply_clahe_8bit(img,2.0,(8,8))
        img=self.despeckle(img,method=self.denoise,ksize=3); return img
    def __len__(self): return len(self.items)
    def __getitem__(self, idx):
        patient,instant=self.items[idx]
        ip=os.path.join(self.data_path,patient,f"{patient}_{self.view}_{instant}.nii.gz")
        mp=os.path.join(self.data_path,patient,f"{patient}_{self.view}_{instant}_gt.nii.gz")
        img=self.sitk_read(ip); msk=self.sitk_read(mp)
        img=self.robust_preprocess_us(img)
        msk=msk.astype(np.int64); 
        if not np.isin(msk,[0,1,2,3]).all(): msk=np.where(np.isin(msk,[0,1,2,3]),msk,0).astype(np.int64)
        img_t=self.tf["image"](img); msk_t=self.tf["gt"](msk).squeeze(0).long()
        if self.in_channels>1: img_t=img_t.repeat(self.in_channels,1,1)
        return img_t, msk_t

class CAMUSNPZDataset(Dataset):
    """Loads preprocessed .npz samples saved by CAMUSPreprocessor."""
    def __init__(self, root_dir):
        super().__init__()
        self.root_dir=root_dir
        self.files=sorted([os.path.join(root_dir,f) for f in os.listdir(root_dir) if f.endswith(".npz")])
        if not self.files: raise FileNotFoundError(f"No .npz files in {root_dir}")
    def __len__(self): return len(self.files)
    def __getitem__(self, idx):
        arr=np.load(self.files[idx], allow_pickle=True)
        img=arr["image"].astype(np.float32) # [H,W] in [0,1]
        msk=arr["mask"].astype(np.int64)    # [H,W]
        img=torch.from_numpy(img)[None,...] # -> [1,H,W]
        msk=torch.from_numpy(msk)
        return img, msk


# ================== MAIN ==================
def main():
    os.makedirs(SPLIT_DIR, exist_ok=True)
    # ---- PREPROCESS MODE ----
    if RUN_MODE == "preprocess":
        pp = CAMUSPreprocessor(data_dir=DATA_DIR, split_dir=SPLIT_DIR, out_dir=OUT_DIR,
                               view=VIEW, img_size=IMG_SIZE, do_clahe=True, denoise="median",
                               overwrite=False, seed=1234)
        pp.build_splits_if_missing()
        pp.preprocess_all()
        return

    # ---- TRAINING SETUP (common) ----
    train_list = os.path.join(SPLIT_DIR, "train_samples.npy")
    val_list   = os.path.join(SPLIT_DIR, "test_ED.npy")

    # Build splits if missing (for train_from_nifti / train_from_npz)
    pp_temp = CAMUSPreprocessor(DATA_DIR, SPLIT_DIR, OUT_DIR, VIEW, IMG_SIZE)
    pp_temp.build_splits_if_missing()

    if RUN_MODE == "train_from_nifti":
        train_ds = CAMUS_loader(DATA_DIR, train_list, view=VIEW, img_size=IMG_SIZE,
                                in_channels=IN_CHANNELS, augment=True, use_clahe=True, denoise="median")
        val_ds   = CAMUS_loader(DATA_DIR, val_list,   view=VIEW, img_size=IMG_SIZE,
                                in_channels=IN_CHANNELS, augment=False, use_clahe=True, denoise="median")
    elif RUN_MODE == "train_from_npz":
        # Ensure preprocessed data exists
        if not (os.path.isdir(os.path.join(OUT_DIR,"train")) and os.path.isdir(os.path.join(OUT_DIR,"test_ED"))):
            print("Preprocessed .npz not found. Running preprocessing once...")
            pp_temp.preprocess_all()
        train_ds = CAMUSNPZDataset(root_dir=os.path.join(OUT_DIR, "train"))
        val_ds   = CAMUSNPZDataset(root_dir=os.path.join(OUT_DIR, "test_ED"))
    else:
        raise ValueError(f"Unknown RUN_MODE '{RUN_MODE}'")

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=DEFAULT_WORKERS, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False,
                              num_workers=DEFAULT_WORKERS, pin_memory=True)

    model = TransUNetLite(in_channels=IN_CHANNELS, num_classes=NUM_CLASSES, img_size=IMG_SIZE).to(DEVICE)
    criterion = DiceCELoss(dice_w=0.5, ce_w=0.5, class_weights=None, ignore_index=None, smooth=1.0)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=STEP_SIZE, gamma=GAMMA)

    ensure_dir(RESULTS_DIR)
    scaler = torch.amp.GradScaler("cuda") if (USE_AMP and DEVICE.type=="cuda") else None
    best_mdice = 0.0; bad = 0

    for epoch in range(EPOCHS):
        tr_loss = train_one_epoch(model, train_loader, optimizer, criterion, DEVICE, scaler=scaler)
        curr_lr = optimizer.param_groups[0]["lr"]
        te_loss, per_dice, mDice, per_iou, mIoU, cm = evaluate(
            model, val_loader, criterion, DEVICE, NUM_CLASSES, save_dir=SAVE_ROOT, epoch=epoch+1, save_n=SAVE_N
        )
        print(f"[{TAG}] Epoch {epoch+1:03d} | train {tr_loss:.4f} | val {te_loss:.4f} | "
              f"mDice {mDice:.4f} | mIoU {mIoU:.4f} | Dice {['%.3f'%d for d in per_dice]} | "
              f"IoU {['%.3f'%i for i in per_iou]} | LR {curr_lr:.2e}")
        log_metrics_csv(METRICS_CSV, epoch+1, tr_loss, te_loss, mDice, mIoU, per_dice, per_iou, curr_lr)
        cm_path = os.path.join(RESULTS_DIR, f"confusion_matrix_epoch_{epoch+1:03d}.csv"); save_confusion_matrix(cm, cm_path)
        scheduler.step()

        if mDice > best_mdice:
            best_mdice = mDice; bad = 0
            torch.save(model.state_dict(), f"best_transunet_dicece_mdice_{best_mdice:.4f}.pt")
        else:
            bad += 1
            if bad >= PATIENCE:
                print("Early stopping."); break

    print(f"\n✅ Metrics logged to: {METRICS_CSV}")
    print(f"   Confusion matrices in: {RESULTS_DIR}/confusion_matrix_epoch_XXX.csv")


if __name__ == "__main__":
    main()
