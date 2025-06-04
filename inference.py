import ipdb
from torchvision import models
from torchvision.models.segmentation.deeplabv3 import DeepLabHead, DeepLabV3_ResNet101_Weights
from model.deeplab import modify_deeplabv3_for_grayscale
from data.data import SpotGeoDataset
import torch
from torchvision import transforms
from torch.utils.data import DataLoader
from data.data import collate_fn
import matplotlib.pyplot as plt

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = models.segmentation.deeplabv3_resnet101(weights=DeepLabV3_ResNet101_Weights.COCO_WITH_VOC_LABELS_V1)
    model.classifier = DeepLabHead(in_channels=2048, num_classes=1)  # modifying model head.
    model = modify_deeplabv3_for_grayscale(model)

    trained_ckpt_dict = torch.load("./checkpoints/deeplab_loss_model_1.pt", map_location="cpu")
    model.load_state_dict(trained_ckpt_dict["model_state_dict"])

    T = transforms.Compose(
        [
            transforms.Resize([224, 224]),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5]),
        ]
    )
    test_dataset = SpotGeoDataset(root_dir="./data/spotGEO", mode="test", transforms=T)
    test_loader = DataLoader(dataset=test_dataset, batch_size=1, shuffle=False, collate_fn=collate_fn)

    for idx, (images, masks, centroids) in enumerate(test_loader):
        model.eval()
        images, masks = images.to(device), masks.unsqueeze(1).float().to(device)
        outputs = model(images)["out"]

        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))

        ax1.imshow(images.squeeze())
        ax2.imshow(masks.squeeze())
        ax3.imshow(outputs.squeeze().detach().numpy())
        plt.show()
        ipdb.set_trace()
