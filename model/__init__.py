from .deeplab import create_deeplabv3_model
from .criterion import CombinedLoss
import torch


def create_model(args):
    model = create_deeplabv3_model()
    criterion = CombinedLoss(bce_weight=1.0, dice_weight=1.0, pos_weight=torch.tensor([10.0]))

    return model, criterion
