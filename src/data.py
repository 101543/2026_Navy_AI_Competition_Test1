import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms


def get_transforms(train=True):
    if train:
        return transforms.Compose(
            [
                transforms.RandomRotation(10),
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,)),
            ]
        )

    return transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
        ]
    )


def build_dataloaders(config):
    data_cfg = config["data"]
    train_cfg = config["train"]

    train_source = datasets.MNIST(
        root=data_cfg["root"],
        train=True,
        download=True,
        transform=get_transforms(train=True),
    )
    val_source = datasets.MNIST(
        root=data_cfg["root"],
        train=True,
        download=True,
        transform=get_transforms(train=False),
    )

    val_size = int(len(train_source) * data_cfg["val_ratio"])
    train_size = len(train_source) - val_size
    generator = torch.Generator().manual_seed(config["seed"])
    indices = torch.randperm(len(train_source), generator=generator).tolist()

    train_dataset = Subset(train_source, indices[:train_size])
    val_dataset = Subset(val_source, indices[train_size:])

    test_dataset = datasets.MNIST(
        root=data_cfg["root"],
        train=False,
        download=True,
        transform=get_transforms(train=False),
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=train_cfg["batch_size"],
        shuffle=True,
        num_workers=data_cfg["num_workers"],
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=train_cfg["batch_size"],
        shuffle=False,
        num_workers=data_cfg["num_workers"],
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=train_cfg["batch_size"],
        shuffle=False,
        num_workers=data_cfg["num_workers"],
        pin_memory=True,
    )
    return train_loader, val_loader, test_loader
