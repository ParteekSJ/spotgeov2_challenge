import torch
from torch import nn


def dice_loss(pred, target, smooth=1e-6):
    pred = torch.sigmoid(pred)  # Convert logits to probabilities
    pred = pred.contiguous().view(-1)
    target = target.contiguous().view(-1)
    intersection = (pred * target).sum()
    return 1 - ((2.0 * intersection + smooth) / (pred.sum() + target.sum() + smooth))


class CombinedLoss(nn.Module):
    def __init__(self, bce_weight=1.0, dice_weight=1.0, pos_weight=None):
        super().__init__()
        self.bce_loss = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        self.dice_weight = dice_weight
        self.bce_weight = bce_weight

    def forward(self, pred, target):
        bce = self.bce_loss(pred, target)
        dice = dice_loss(pred, target)
        return self.bce_weight * bce + self.dice_weight * dice
