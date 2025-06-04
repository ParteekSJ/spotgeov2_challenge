import torch
from .data import SpotGeoDataset
from torch.utils.data import DataLoader
import torch.nn.functional as F
from torchvision import transforms
import ipdb

T = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Resize([224, 224]),
        transforms.Normalize(
            mean=[0.5],
            std=[0.5],
        ),
    ]
)


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


def load_dataset(args):
    train_dataset = SpotGeoDataset(root_dir=args.dataset_path, mode="train", transforms=T)
    val_dataset = SpotGeoDataset(root_dir=args.dataset_path, mode="test", transforms=T)

    trainloader = DataLoader(dataset=train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)
    valloader = DataLoader(dataset=val_dataset, batch_size=1, shuffle=False, collate_fn=collate_fn)

    return trainloader, valloader
