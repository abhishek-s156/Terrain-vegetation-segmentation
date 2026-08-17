import argparse
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from .dataset import LandCoverDataset, CLASS_NAMES
from .metrics import pixel_accuracy, mean_iou
from .models import build_model


def get_logits(model, x):
    output = model(x)
    if isinstance(output, dict):
        return output["out"]
    return output


def run_epoch(model, loader, optimizer, criterion, device, train=True):
    model.train(train)
    total_loss = 0.0
    total_acc = 0.0
    total_iou = 0.0

    context = torch.enable_grad() if train else torch.no_grad()

    with context:
        for images, masks in tqdm(loader, leave=False):
            images = images.to(device)
            masks = masks.to(device)

            logits = get_logits(model, images)
            loss = criterion(logits, masks)

            if train:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()

            total_loss += loss.item()
            total_acc += pixel_accuracy(logits.detach(), masks)
            total_iou += mean_iou(logits.detach(), masks, len(CLASS_NAMES))

    n = max(len(loader), 1)
    return total_loss / n, total_acc / n, total_iou / n


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default="data/processed")
    parser.add_argument("--model", choices=["unet", "deeplabv3"], default="unet")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--image_size", type=int, default=256)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-3)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Classes: {CLASS_NAMES}")

    train_ds = LandCoverDataset(
        args.data_dir, "train", args.image_size, augment=True
    )
    val_ds = LandCoverDataset(
        args.data_dir, "val", args.image_size, augment=False
    )

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=0, pin_memory=torch.cuda.is_available()
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=0, pin_memory=torch.cuda.is_available()
    )

    model = build_model(args.model, len(CLASS_NAMES)).to(device)

    # Unknown class is kept as a normal class for this starter project.
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)

    checkpoint_dir = Path("outputs/checkpoints")
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = checkpoint_dir / f"best_{args.model}.pt"

    best_iou = -1.0

    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc, train_iou = run_epoch(
            model, train_loader, optimizer, criterion, device, train=True
        )
        val_loss, val_acc, val_iou = run_epoch(
            model, val_loader, optimizer, criterion, device, train=False
        )

        print(
            f"Epoch {epoch}/{args.epochs} | "
            f"train loss={train_loss:.4f}, acc={train_acc:.4f}, IoU={train_iou:.4f} | "
            f"val loss={val_loss:.4f}, acc={val_acc:.4f}, IoU={val_iou:.4f}"
        )

        if val_iou > best_iou:
            best_iou = val_iou
            torch.save(
                {
                    "model": args.model,
                    "num_classes": len(CLASS_NAMES),
                    "image_size": args.image_size,
                    "class_names": CLASS_NAMES,
                    "state_dict": model.state_dict(),
                    "val_iou": best_iou,
                },
                checkpoint_path,
            )
            print(f"Saved best checkpoint -> {checkpoint_path}")

    print("Training complete.")
    print(f"Best validation IoU: {best_iou:.4f}")


if __name__ == "__main__":
    main()
