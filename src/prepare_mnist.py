import argparse
from pathlib import Path

import pandas as pd
import torch
from torchvision import datasets
from tqdm import tqdm

from utils import ensure_dir, load_config, project_root, set_seed


def save_split(dataset, indices, image_dir, csv_path, include_label=True):
    ensure_dir(image_dir)
    rows = []

    for output_id, dataset_idx in enumerate(tqdm(indices, desc=csv_path.stem), start=1):
        image, label = dataset[dataset_idx]
        file_name = f"{output_id:05d}.png"
        image.save(image_dir / file_name)

        row = {
            "id": output_id,
            "image_path": f"{image_dir.name}/{file_name}",
        }
        if include_label:
            row["label"] = int(label)
        rows.append(row)

    pd.DataFrame(rows).to_csv(csv_path, index=False)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    args = parser.parse_args()

    root = project_root()
    config = load_config(root / args.config)
    set_seed(config["seed"])

    data_cfg = config["data"]
    processed_dir = root / data_cfg["processed_dir"]
    ensure_dir(processed_dir)

    train_source = datasets.MNIST(
        root=root / data_cfg["root"],
        train=True,
        download=True,
    )
    test_source = datasets.MNIST(
        root=root / data_cfg["root"],
        train=False,
        download=True,
    )

    val_size = int(len(train_source) * data_cfg["val_ratio"])
    train_size = len(train_source) - val_size
    generator = torch.Generator().manual_seed(config["seed"])
    indices = torch.randperm(len(train_source), generator=generator).tolist()

    train_indices = indices[:train_size]
    val_indices = indices[train_size:]
    test_indices = list(range(len(test_source)))

    save_split(
        train_source,
        train_indices,
        processed_dir / "train_images",
        root / data_cfg["train_csv"],
        include_label=True,
    )
    save_split(
        train_source,
        val_indices,
        processed_dir / "val_images",
        root / data_cfg["val_csv"],
        include_label=True,
    )
    save_split(
        test_source,
        test_indices,
        processed_dir / "test_images",
        root / data_cfg["test_csv"],
        include_label=False,
    )

    print(f"Prepared MNIST image dataset under: {processed_dir}")


if __name__ == "__main__":
    main()
