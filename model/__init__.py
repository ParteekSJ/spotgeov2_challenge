from .deeplab import create_deeplabv3_model
import torch


def create_model(args):
    model = create_deeplabv3_model()
    criterion = torch.nn.MSELoss()

    return model, criterion
