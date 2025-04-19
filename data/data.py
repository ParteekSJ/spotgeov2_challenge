import json
from PIL import Image
from albumentations.pytorch import ToTensorV2
from torchvision import transforms
from torch.utils.data import Dataset
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import ipdb
import numpy as np
import torch


class SpotGeoDataset(Dataset):
    def __init__(self, root_dir: str, mode: str, transforms):
        super().__init__()

        self.root_dir = root_dir
        self.mode = mode
        self.transforms = transforms

        with open(f"{root_dir}/{mode}_anno.json") as json_data:
            annotation_data = json.load(json_data)

        self.organized_data = {}
        for sequence_data in annotation_data:
            seq_id = str(sequence_data["sequence_id"])
            frame = str(sequence_data["frame"])
            self.organized_data[f"{seq_id}_{frame}"] = sequence_data  # already sorted

        self.frame_seq_ids = list(self.organized_data.keys())

    def __len__(self):
        return len(self.organized_data)
        # return 1

    def __getitem__(self, idx):
        sequence, frame = self.frame_seq_ids[idx].split("_")

        image_path = f"{self.root_dir}/{self.mode}/{sequence}/{frame}.png"
        # image = transforms.PILToTensor()(Image.open(image_path))
        image_np = np.array(Image.open(image_path).convert("L"), dtype=np.uint8)

        centroids = self.organized_data[self.frame_seq_ids[idx]]["object_coords"]

        if centroids:
            centroids_tensor = torch.tensor(centroids, dtype=torch.float32)  # [N,2]
        else:
            centroids_tensor = torch.zeros((0, 2), dtype=torch.float32)  # empty

        mask_np = create_mask(image_np, centroids, std=0.2)

        if self.transforms:
            aug = self.transforms(image=image_np, mask=mask_np)
            image_tensor = aug["image"]  # torch.Tensor [1, H', W']
            mask_tensor = aug["mask"]  # torch.Tensor [1, H', W']
        else:
            # fallback to simple conversion
            image_tensor = ToTensorV2()(image=image_np)["image"]
            mask_tensor = ToTensorV2()(image=mask_np)["image"]

        return image_tensor.unsqueeze(0), mask_tensor.unsqueeze(0), centroids_tensor

    def visualize_image_with_centroids(self, idx):
        image, _, centroids = self.__getitem__(idx)

        fig, ax = plt.subplots()
        ax.imshow(image.squeeze(0), cmap="viridis")
        for centroid in centroids:
            plt.scatter(x=centroid[0].item(), y=centroid[1].item(), s=0.5)
            circle = Circle(
                xy=(centroid[0].item(), centroid[1].item()),
                radius=8,
                fill=False,
                color="white",
                linestyle="--",
                linewidth=0.45,
            )
            ax.add_patch(circle)

        plt.grid(True, linestyle="--", alpha=0.7)
        plt.show()


def create_mask(image, centroids, std=0.2):
    """
    Create a binary mask with a Gaussian spread (PSF‑like) for each centroid,
    implemented using NumPy arrays.

    Args:
        image (np.ndarray): Input image array (used only for its H, W).
        centroids (Iterable of (x, y)): Pixel coordinates of each spot.
        std (float): Standard deviation of the Gaussian spread.

    Returns:
        np.ndarray: A mask of shape (H, W) with values 0.0 or 1.0.
    """
    # get W,H from PIL
    h, w = image.shape
    mask = np.zeros((h, w), dtype=np.float32)
    y = np.arange(h, dtype=np.float32)[:, None]
    x = np.arange(w, dtype=np.float32)[None, :]
    for cx, cy in centroids:
        gauss = np.exp(-((x - cx) ** 2 + (y - cy) ** 2) / (2 * std**2))
        mask += gauss
    return (mask > 0).astype(np.float32)


if __name__ == "__main__":
    ds = SpotGeoDataset(
        root_dir="/Users/parteeksj/Desktop/SpotGeoV2_Project/data/SpotGeoV2",
        mode="train",
        transforms=transforms.ToTensor(),
    )

    ds.visualize_image_with_centroids(123)
