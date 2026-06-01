from pathlib import Path

import pandas as pd
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms


class MNISTImageDataset(Dataset):
    def __init__(self, csv_path, transform=None, has_label=True):
        self.csv_path = Path(csv_path)
        if not self.csv_path.exists():
            raise FileNotFoundError(
                f"{self.csv_path} not found. Run: python src/prepare_mnist.py --config configs/default.yaml"
            )
        self.data = pd.read_csv(self.csv_path)
        self.transform = transform
        self.has_label = has_label

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        image_path = self.csv_path.parent / row["image_path"]
        image = Image.open(image_path).convert("L")

        if self.transform:
            image = self.transform(image)

        if self.has_label:
            return image, int(row["label"])

        return image, int(row["id"])


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

    train_dataset = MNISTImageDataset(
        data_cfg["train_csv"],
        transform=get_transforms(train=True),
    )
    val_dataset = MNISTImageDataset(
        data_cfg["val_csv"],
        transform=get_transforms(train=False),
    )
    test_dataset = MNISTImageDataset(
        data_cfg["test_csv"],
        transform=get_transforms(train=False),
        has_label=False,
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
