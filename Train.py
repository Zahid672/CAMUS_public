import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# --- import your classes ---
from dataset import CAMUS_loader
from Unet import UNet

def dice_per_class(logits, target, num_classes):
    # logits: [B,C,H,W], target: [B,H,W]
    with torch.no_grad():
        preds = torch.argmax(logits, dim=1)            # [B,H,W]
        dices = []
        eps = 1e-6
        for c in range(num_classes):
            pred_c = (preds == c).float()
            targ_c = (target == c).float()
            inter = (pred_c * targ_c).sum()
            denom = pred_c.sum() + targ_c.sum()
            dice = (2 * inter + eps) / (denom + eps)
            dices.append(dice.item())
        return dices, sum(dices) / num_classes

def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    running_loss = 0.0
    for imgs, masks in loader:
        imgs  = imgs.to(device)
        masks = masks.to(device)              # [B,H,W], long
        optimizer.zero_grad()
        logits = model(imgs)                  # [B,4,H,W]
        loss = criterion(logits, masks)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    return running_loss / len(loader)

@torch.no_grad()
def evaluate(model, loader, criterion, device, num_classes):
    model.eval()
    running_loss = 0.0
    dices_all = []
    for imgs, masks in loader:
        imgs  = imgs.to(device)
        masks = masks.to(device)
        logits = model(imgs)
        loss = criterion(logits, masks)
        running_loss += loss.item()

        dices, mean_dice = dice_per_class(logits, masks, num_classes)
        dices_all.append((dices, mean_dice))

    # average dice over batches
    per_class = torch.tensor([d for d,_ in dices_all]).mean(dim=0).tolist()
    mean_dice = torch.tensor([m for _,m in dices_all]).mean().item()
    return running_loss / len(loader), per_class, mean_dice

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    num_classes = 4
    view = '2CH'  # or '4CH'

    # --- dataset & loaders ---
    data_dir = 'database_nifti'
    split_dir = 'prepared_data'
    train_list = os.path.join(split_dir, 'train_samples.npy')
    test_list  = os.path.join(split_dir, 'test_ED.npy')  # or combine ED/ES as you like

    train_ds = CAMUS_loader(data_dir, train_list, view=view)
    test_ds  = CAMUS_loader(data_dir, test_list, view=view)

    train_loader = DataLoader(train_ds, batch_size=4, shuffle=True, num_workers=4, pin_memory=True)
    test_loader  = DataLoader(test_ds,  batch_size=4, shuffle=False, num_workers=4, pin_memory=True)

    # --- model, loss, optim, sched ---
    model = UNet().to(device)          # UNet with out_channels=4
    criterion = nn.CrossEntropyLoss()  # target: [B,H,W] long
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

    best_mdice = 0.0
    patience, bad = 20, 0

    for epoch in range(60):
        tr_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
        te_loss, per_class_dice, mean_dice = evaluate(model, test_loader, criterion, device, num_classes)

        print(f"Epoch {epoch+1:03d} | train loss {tr_loss:.4f} | val loss {te_loss:.4f} | mDice {mean_dice:.4f} | per-class {['%.3f'%d for d in per_class_dice]}")
        scheduler.step()  # StepLR doesn't take a metric; if you use ReduceLROnPlateau, pass te_loss.

        # early stopping on mean Dice
        if mean_dice > best_mdice:
            best_mdice = mean_dice
            bad = 0
            torch.save(model.state_dict(), f"best_unet_mdice_{best_mdice:.4f}.pt")
        else:
            bad += 1
            if bad >= patience:
                print("Early stopping.")
                break

if __name__ == "__main__":
    import os
    main()
