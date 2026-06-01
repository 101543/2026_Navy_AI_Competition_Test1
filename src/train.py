import argparse
from pathlib import Path

import torch
import torch.nn as nn
from tqdm import tqdm

from data import build_dataloaders
from model import SimpleCNN
from utils import ensure_dir, get_device, load_config, project_root, set_seed


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    for images, labels in tqdm(loader, desc="train", leave=False):
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        logits = model(images)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * images.size(0)
        correct += (logits.argmax(dim=1) == labels).sum().item()
        total += labels.size(0)

    return total_loss / total, correct / total


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    for images, labels in tqdm(loader, desc="valid", leave=False):
        images, labels = images.to(device), labels.to(device)
        logits = model(images)
        loss = criterion(logits, labels)

        total_loss += loss.item() * images.size(0)
        correct += (logits.argmax(dim=1) == labels).sum().item()
        total += labels.size(0)

    return total_loss / total, correct / total


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    args = parser.parse_args()

    root = project_root()
    config = load_config(root / args.config)
    set_seed(config["seed"])

    device = get_device(config["device"])
    print(f"Device: {device}")

    train_loader, val_loader, _ = build_dataloaders(config)
    model = SimpleCNN(num_classes=config["model"]["num_classes"]).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config["train"]["learning_rate"],
    )

    model_dir = root / config["paths"]["model_dir"]
    ensure_dir(model_dir)
    best_path = model_dir / "best_model.pt"

    best_acc = 0.0
    for epoch in range(1, config["train"]["epochs"] + 1):
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device
        )
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)

        print(
            f"Epoch {epoch:02d} | "
            f"train_loss={train_loss:.4f}, train_acc={train_acc:.4f} | "
            f"val_loss={val_loss:.4f}, val_acc={val_acc:.4f}"
        )

        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "best_acc": best_acc,
                    "config": config,
                },
                best_path,
            )
            print(f"Saved best model: {best_path}")

    print(f"Best validation accuracy: {best_acc:.4f}")


if __name__ == "__main__":
    main()
