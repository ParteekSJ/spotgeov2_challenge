from torch import nn
import torch
import torch.nn.functional as F
import ipdb


class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class UNetPlusPlus(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, filters=[64, 128, 256, 512, 1024]):
        super().__init__()
        # encoder: one ConvBlock per level
        self.conv0_0 = ConvBlock(in_channels, filters[0])
        self.conv1_0 = ConvBlock(filters[0], filters[1])
        self.conv2_0 = ConvBlock(filters[1], filters[2])
        self.conv3_0 = ConvBlock(filters[2], filters[3])
        self.conv4_0 = ConvBlock(filters[3], filters[4])

        # decoder / nested
        self.conv0_1 = ConvBlock(filters[0] + filters[1], filters[0])
        self.conv1_1 = ConvBlock(filters[1] + filters[2], filters[1])
        self.conv2_1 = ConvBlock(filters[2] + filters[3], filters[2])
        self.conv3_1 = ConvBlock(filters[3] + filters[4], filters[3])

        self.conv0_2 = ConvBlock(filters[0] * 2 + filters[1], filters[0])
        self.conv1_2 = ConvBlock(filters[1] * 2 + filters[2], filters[1])
        self.conv2_2 = ConvBlock(filters[2] * 2 + filters[3], filters[2])

        self.conv0_3 = ConvBlock(filters[0] * 3 + filters[1], filters[0])
        self.conv1_3 = ConvBlock(filters[1] * 3 + filters[2], filters[1])

        self.conv0_4 = ConvBlock(filters[0] * 4 + filters[1], filters[0])

        # pooling & upsampling
        self.pool = nn.MaxPool2d(2, 2)
        self.up = lambda x, size: F.interpolate(x, size=size, mode="bilinear", align_corners=True)

        # final 1×1 conv
        self.final = nn.Conv2d(filters[0], out_channels, kernel_size=1)

    def forward(self, x):
        # -------- encoder --------
        x0_0 = self.conv0_0(x)
        x1_0 = self.conv1_0(self.pool(x0_0))
        x2_0 = self.conv2_0(self.pool(x1_0))
        x3_0 = self.conv3_0(self.pool(x2_0))
        x4_0 = self.conv4_0(self.pool(x3_0))

        # -------- decoder (nested) --------
        x0_1 = self.conv0_1(torch.cat([x0_0, self.up(x1_0, x0_0.shape[2:])], dim=1))

        x1_1 = self.conv1_1(torch.cat([x1_0, self.up(x2_0, x1_0.shape[2:])], dim=1))

        x2_1 = self.conv2_1(torch.cat([x2_0, self.up(x3_0, x2_0.shape[2:])], dim=1))

        x3_1 = self.conv3_1(torch.cat([x3_0, self.up(x4_0, x3_0.shape[2:])], dim=1))

        x0_2 = self.conv0_2(torch.cat([x0_0, x0_1, self.up(x1_1, x0_0.shape[2:])], dim=1))

        x1_2 = self.conv1_2(torch.cat([x1_0, x1_1, self.up(x2_1, x1_0.shape[2:])], dim=1))

        x2_2 = self.conv2_2(torch.cat([x2_0, x2_1, self.up(x3_1, x2_0.shape[2:])], dim=1))

        x0_3 = self.conv0_3(torch.cat([x0_0, x0_1, x0_2, self.up(x1_2, x0_0.shape[2:])], dim=1))

        x1_3 = self.conv1_3(torch.cat([x1_0, x1_1, x1_2, self.up(x2_2, x1_0.shape[2:])], dim=1))

        x0_4 = self.conv0_4(torch.cat([x0_0, x0_1, x0_2, x0_3, self.up(x1_3, x0_0.shape[2:])], dim=1))

        # final head & sigmoid for binary mask
        out = self.final(x0_4)
        return torch.sigmoid(out)


class DiceBCELoss(nn.Module):
    """
    Combination of Dice and Binary Cross Entropy loss for binary segmentation
    """

    def __init__(self, weight_bce=0.5, weight_dice=0.5):
        super(DiceBCELoss, self).__init__()
        self.weight_bce = weight_bce
        self.weight_dice = weight_dice
        self.bce = nn.BCEWithLogitsLoss()

    def forward(self, inputs, targets):

        # Handle deep supervision case
        if isinstance(inputs, list):
            loss = 0
            for inp in inputs:
                loss += self._calculate_loss(inp, targets)
            return loss / len(inputs)
        else:
            return self._calculate_loss(inputs, targets)

    def _calculate_loss(self, inputs, targets):
        # Ensure inputs and targets have same shape
        if inputs.shape != targets.shape:
            targets = F.interpolate(targets, size=inputs.shape[2:], mode="nearest")

        # BCE Loss
        bce_loss = self.bce(inputs, targets)

        # Dice Loss
        # inputs_sigmoid = torch.sigmoid(inputs)
        intersection = (inputs * targets).sum(dim=(2, 3))
        dice_coeff = (2.0 * intersection + 1) / (inputs.sum(dim=(2, 3)) + targets.sum(dim=(2, 3)) + 1)
        dice_loss = 1 - dice_coeff.mean()

        # Combined loss
        return self.weight_bce * bce_loss + self.weight_dice * dice_loss


if __name__ == "__main__":
    gm = UNetPlusPlus()
    ip = torch.randn(1, 1, 480, 640)
    op = gm(ip)
    print(f"{op.shape=}")
