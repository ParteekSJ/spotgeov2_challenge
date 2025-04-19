from .unetplusplus.model import UNetPlusPlus, DiceBCELoss


def create_model(args):
    unetplusplus = UNetPlusPlus()
    criterion = DiceBCELoss()
    return unetplusplus, criterion
