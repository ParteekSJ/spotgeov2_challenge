import json
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import numpy as np
import torch
import ipdb
import torch.nn.functional as F


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
        if self.mode == "test":
            return 1000
        return len(self.organized_data)
        # return 10

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
            image_tensor = self.transforms(image_np)
            mask_tensor = self.transforms(mask_np)

        # Binarizing the mask
        # mask_tensor[mask_tensor == -1] = 0.0
        # mask_tensor[mask_tensor < 0] = 1.0

        return image_tensor.unsqueeze(0), mask_tensor.unsqueeze(0), centroids_tensor

    def visualize_image_with_centroids(self, idx):
        image, _, centroids = self.__getitem__(idx)

        fig, ax = plt.subplots()
        ax.imshow(image.squeeze(), cmap="viridis")
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


def collate_fn(batch):
    if len(batch) != 1:
        # ipdb.set_trace()
        centroids_len_batch_max = max([x[-1].shape[0] for x in batch])
        pad_tensor_func = lambda x: (
            (
                x[0],
                x[1],
                F.pad(input=x[-1], pad=(0, 0, 0, centroids_len_batch_max - x[-1].shape[0]), value=-1e-9).unsqueeze(0),
            )
            if x[-1].shape[0] != centroids_len_batch_max
            else (x[0], x[1], x[2].unsqueeze(0))
        )

        _batch = [pad_tensor_func(x) for x in batch]
        images = torch.cat([x[0] for x in _batch], dim=0)
        masks = torch.cat([x[1] for x in _batch], dim=0)
        centroids = torch.cat([x[2] for x in _batch], dim=0)

        # for idx, x in enumerate(batch):
        #     centroid = x[-1]
        #     if centroid.shape[0] != centroids_len_batch_max:
        #         amt_to_pad = centroids_len_batch_max - centroid.shape[0]
        #         batch[idx][-1] = F.pad(input=centroid, pad=(0, 0, 0, amt_to_pad), value=-1e-9)
        return images, masks.squeeze(), centroids
    else:
        return batch[0]


def create_mask(image, centroids, std=0.2, threshold=None):
    """
    Create a binary mask with a Gaussian spread (PSF-like) for each centroid,
    implemented using NumPy arrays and returned as a PyTorch tensor.

    Args:
        image (np.ndarray or PIL.Image): Input image to determine mask shape (H, W).
        centroids (Iterable of (x, y)): Pixel coordinates of each spot (x, y).
        std (float): Standard deviation of the Gaussian spread.
        threshold (float, optional): Threshold for binarizing the Gaussian mask.
                                   If None, defaults to exp(-1/(2*std^2)) (~0.607).

    Returns:
        torch.Tensor: Binary mask of shape (1, H, W) with values 0.0 or 1.0.
    """
    # Handle input image
    if isinstance(image, Image.Image):
        h, w = image.size[1], image.size[0]  # PIL uses (W, H)
    elif isinstance(image, np.ndarray):
        h, w = image.shape[:2]  # Assume (H, W) or (H, W, C)
    else:
        raise ValueError("Image must be a PIL Image or NumPy array")

    # Initialize mask
    mask = np.zeros((h, w), dtype=np.float32)

    # Create coordinate grids
    y = np.arange(h, dtype=np.float32)[:, None]
    x = np.arange(w, dtype=np.float32)[None, :]

    # Validate and process centroids
    for cx, cy in centroids:
        if not (0 <= cx < w and 0 <= cy < h):
            print(f"Warning: Centroid ({cx}, {cy}) is out of bounds for image size ({w}, {h}). Skipping.")
            continue
        gauss = np.exp(-((x - cx) ** 2 + (y - cy) ** 2) / (2 * std**2))
        mask += gauss

    # Set default threshold to limit Gaussian spread (approximately 1 std deviation)
    if threshold is None:
        threshold = np.exp(-1 / (2 * std**2))  # Value at 1 std deviation

    # Binarize mask
    binary_mask = (mask > threshold).astype(np.float32)
    return binary_mask


if __name__ == "__main__":
    ipdb.set_trace()
    ds = SpotGeoDataset(
        root_dir="/Users/parteeksj/Desktop/SpotGeoV2_Project/data/spotGEO",
        mode="train",
        transforms=transforms.ToTensor(),
    )

    image1 = ds.__getitem__(1)
    ds.visualize_image_with_centroids(123)
