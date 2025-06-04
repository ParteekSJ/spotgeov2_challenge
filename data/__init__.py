import torch
from .data import SpotGeoDataset, collate_fn
from torch.utils.data import DataLoader
from torchvision import transforms

T = transforms.Compose([transforms.ToTensor(), transforms.Resize([224, 224])])


def load_dataset(args):
    train_dataset = SpotGeoDataset(root_dir=args.dataset_path, mode="train", transforms=T)
    val_dataset = SpotGeoDataset(root_dir=args.dataset_path, mode="test", transforms=T)

    trainloader = DataLoader(dataset=train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)
    valloader = DataLoader(dataset=val_dataset, batch_size=1, shuffle=False, collate_fn=collate_fn)

    return trainloader, valloader
