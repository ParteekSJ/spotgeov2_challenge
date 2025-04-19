import torch
from .data import SpotGeoDataset
from torch.utils.data import DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2
import ipdb
import torch.nn.functional as F

train_transform = A.Compose(
    [
        # —— Geometric ——
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.2),
        A.RandomRotate90(p=0.5),
        A.RandomResizedCrop(size=(240, 320), scale=(0.8, 1.0), ratio=(0.75, 1.33), p=1.0),
        A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.2, rotate_limit=30, p=0.7),
        A.ElasticTransform(alpha=1, sigma=50, p=0.3),
        A.GridDistortion(num_steps=5, distort_limit=0.3, p=0.3),
        # —— Photometric (image only) ——
        A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=0.5),
        A.GaussNoise(var_limit=(10.0, 50.0), p=0.3),
        A.GaussianBlur(blur_limit=(3, 7), p=0.2),
        # —— Hide patches (image only) ——
        A.CoarseDropout(max_holes=8, max_height=20, max_width=20, fill_value=0, p=0.3),
        # —— Finalise ——
        A.Normalize(mean=[0.5], std=[0.5]),  # for grayscale use single‐value lists
        ToTensorV2(),  # outputs both image & mask as torch.Tensor
    ],
    additional_targets={"mask": "mask"},  # ensures mask gets the same geometric ops
)

val_transform = A.Compose(
    [
        A.Resize(240, 320),  # or whatever your input size is
        A.Normalize(mean=[0.5], std=[0.5]),  # grayscale normalization
        ToTensorV2(),  # → image [1,H,W], mask [1,H,W]
    ],
    additional_targets={"mask": "mask"},
)

infer_transform = A.Compose(
    [
        A.Resize(240, 320),
        A.Normalize(mean=[0.5], std=[0.5]),
        ToTensorV2(),  # → image [1,H,W]
    ]
)


def collate_fn(batch):
    if len(batch) != 1:
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
        return images, masks, centroids
    else:
        return batch[0]


def load_dataset(args):
    train_dataset = SpotGeoDataset(
        root_dir=args.dataset_path,
        mode="train",
        transforms=train_transform,
    )
    val_dataset = SpotGeoDataset(root_dir=args.dataset_path, mode="test", transforms=val_transform)

    trainloader = DataLoader(dataset=train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)
    valloader = DataLoader(dataset=val_dataset, batch_size=1, shuffle=False, collate_fn=collate_fn)

    return trainloader, valloader
