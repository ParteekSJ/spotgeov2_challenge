from torchvision.models.segmentation.deeplabv3 import DeepLabHead, DeepLabV3_ResNet101_Weights
from torchvision import models
import torch
from torch import nn
import ipdb


def modify_deeplabv3_for_grayscale(model):
    """
    Modify the first convolutional layer of DeepLab v3 to accept grayscale (1-channel) inputs.
    """
    # Get the first convolutional layer (expects 3 input channels by default)
    original_conv = model.backbone.conv1
    in_channels = 1  # Grayscale input
    out_channels = original_conv.out_channels
    kernel_size = original_conv.kernel_size
    stride = original_conv.stride
    padding = original_conv.padding

    # Create new convolutional layer with 1 input channel
    new_conv = nn.Conv2d(
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
        bias=original_conv.bias is not None,
    )

    # Copy weights for one channel (e.g., average the weights across the RGB channels)
    with torch.no_grad():
        new_conv.weight = nn.Parameter(original_conv.weight.mean(dim=1, keepdim=True))
        if original_conv.bias is not None:
            new_conv.bias = nn.Parameter(original_conv.bias)

    # Replace the original convolutional layer
    model.backbone.conv1 = new_conv

    return model


def create_deeplabv3_model():
    # loading the pretrained model.
    model = models.segmentation.deeplabv3_resnet101(weights=DeepLabV3_ResNet101_Weights.COCO_WITH_VOC_LABELS_V1)
    model.classifier = DeepLabHead(in_channels=2048, num_classes=1)  # modifying model head.
    return modify_deeplabv3_for_grayscale(model)


if __name__ == "__main__":
    # ipdb.set_trace()
    model = create_deeplabv3_model()
    model.train()

    image = torch.randn(4, 1, 224, 224)
