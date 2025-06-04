import torch
from typing import Iterable
from sklearn.metrics import confusion_matrix, f1_score, roc_auc_score, accuracy_score
import numpy as np
import torch.nn.functional as F
import ipdb


def train_one_epoch(
    model: torch.nn.Module,
    criterion: torch.nn.Module,
    data_loader: Iterable,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
    logger,
    args,
):
    train_losses = []
    # ipdb.set_trace()

    model.train()
    for idx, (inputs, masks, centroids) in enumerate(data_loader):
        inputs, masks = inputs.to(device), masks.unsqueeze(1).float().to(device)
        outputs = model(inputs)["out"]
        loss = criterion(outputs, masks)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_losses.append(loss.item())

        if idx % args.print_freq == 0:
            logger.info(f"EPOCH {epoch}, STEP [{idx}/{len(data_loader)}], LOSS: {loss:.4f}")

    # return torch.mean(torch.Tensor(train_losses)).item(), torch.mean(torch.Tensor(train_accs)).item()
    return torch.mean(torch.Tensor(train_losses)).item()


@torch.no_grad()
def validate(model, criterion, data_loader, device, logger, args):
    model.eval()
    losses = []
    all_probs, all_preds, all_targets = [], [], []

    for idx, (inputs, masks, _) in enumerate(data_loader):
        inputs = inputs.to(device)  # [B,1,H,W]
        targets = masks.to(device).squeeze(1)  # [B,H,W]

        logits = model(inputs)["out"]  # [B,2,H,W] or [B,1,H,W] w/ BCE
        loss = criterion(logits, targets.unsqueeze(1))
        losses.append(loss.item())

        if idx % args.print_freq == 0:
            logger.info(f"[VALIDATION] STEP [{idx}/{len(data_loader)}], LOSS: {loss:.4f}")

        probs = F.softmax(logits, dim=1)  # [B,2,H,W]
        preds = probs.argmax(dim=1)  # [B,H,W]

        B, C, H, W = probs.shape
        probs_np = probs.permute(0, 2, 3, 1).reshape(-1, C).cpu().numpy()
        preds_np = preds.reshape(-1).cpu().numpy()
        targets_np = targets.reshape(-1).cpu().numpy()

        all_probs.append(probs_np)
        all_preds.append(preds_np)
        all_targets.append(targets_np)

    all_probs = np.concatenate(all_probs, axis=0)
    all_preds = np.concatenate(all_preds, axis=0)
    all_targets = np.concatenate(all_targets, axis=0)

    # Confusion / TP,FP
    cm = confusion_matrix(all_targets, all_preds, labels=[0, 1])
    tp = cm.diagonal()
    fp = cm.sum(axis=0) - tp

    # Pixel accuracy
    acc = accuracy_score(all_targets, all_preds)

    # Macro‑F1: if no positive pixels at all, just set to 0
    if np.all(all_targets == 0):
        f1_macro = 0.0
    else:
        f1_macro = f1_score(all_targets, all_preds, average="macro")

    # AUROC: guard the single‐class case
    unique = np.unique(all_targets)
    if len(unique) < 2:
        auroc = float("nan")  # or 0.5, or skip reporting
    else:
        auroc = roc_auc_score(all_targets, all_probs, multi_class="ovr")

    return {
        "val_loss": np.mean(losses),
        "accuracy": acc,
        "tp_per_class": tp,
        "fp_per_class": fp,
        "f1_macro": f1_macro,
        "auroc": auroc,
    }
