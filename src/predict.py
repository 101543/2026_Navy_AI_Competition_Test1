import argparse

import pandas as pd
import torch
from tqdm import tqdm

from data import build_dataloaders
from model import SimpleCNN
from utils import ensure_dir, get_device, load_config, project_root


@torch.no_grad()
def predict(model, loader, device):
    model.eval()
    preds = []

    for images, _ in tqdm(loader, desc="predict"):
        images = images.to(device)
        logits = model(images)
        preds.extend(logits.argmax(dim=1).cpu().tolist())

    return preds


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--checkpoint", default="outputs/models/best_model.pt")
    args = parser.parse_args()

    root = project_root()
    config = load_config(root / args.config)
    device = get_device(config["device"])

    _, _, test_loader = build_dataloaders(config)
    model = SimpleCNN(num_classes=config["model"]["num_classes"]).to(device)

    checkpoint = torch.load(root / args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])

    preds = predict(model, test_loader, device)

    submission_dir = root / config["paths"]["submission_dir"]
    ensure_dir(submission_dir)
    submission_path = submission_dir / "submission.csv"

    submission = pd.DataFrame(
        {
            "ImageId": range(1, len(preds) + 1),
            "Label": preds,
        }
    )
    submission.to_csv(submission_path, index=False)
    print(f"Saved submission: {submission_path}")


if __name__ == "__main__":
    main()
