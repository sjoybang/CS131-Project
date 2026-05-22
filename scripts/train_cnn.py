"""Fine-tune ResNet18 on PCam for histopathology cancer detection.

Run from the project root:
    python scripts/train_cnn.py --epochs 5 --num-samples 5000 --download
"""

import argparse
from pathlib import Path
import sys

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.pcam import PCamDataset, get_default_transforms
from src.evaluation.metrics import compute_metrics, plot_confusion_matrix, plot_roc_curve
from src.models.cnn import build_resnet18, evaluate, get_device, train_epoch


RESULTS_DIR = Path("results")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--num-samples", type=int, default=None,
                        help="Cap dataset size (None = full split)")
    parser.add_argument("--data-dir", default="data/raw")
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--save-path", default="results/resnet18_best.pth")
    args = parser.parse_args()

    RESULTS_DIR.mkdir(exist_ok=True)
    device = get_device()
    print(f"Using device: {device}")

    train_dataset = PCamDataset(
        split="train",
        transform=get_default_transforms("train"),
        data_dir=args.data_dir,
        max_samples=args.num_samples,
        download=args.download,
    )
    valid_dataset = PCamDataset(
        split="valid",
        transform=get_default_transforms("valid"),
        data_dir=args.data_dir,
        download=args.download,
    )

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=2)
    valid_loader = DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=False, num_workers=2)

    print(f"Train: {len(train_dataset)} samples | Valid: {len(valid_dataset)} samples")

    model = build_resnet18(pretrained=True).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    best_val_loss = float("inf")
    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_acc, _, _, _ = evaluate(model, valid_loader, criterion, device)

        print(f"Epoch {epoch}/{args.epochs}  "
              f"train loss: {train_loss:.4f}  train acc: {train_acc:.4f}  "
              f"val loss: {val_loss:.4f}  val acc: {val_acc:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), args.save_path)
            print(f"  -> Saved best model to {args.save_path}")

    # Final evaluation on validation set using best checkpoint
    model.load_state_dict(torch.load(args.save_path, map_location=device))
    _, _, y_pred, y_prob, y_true = evaluate(model, valid_loader, criterion, device)

    print("\nFinal validation metrics:")
    compute_metrics(y_true, y_pred, y_prob)

    plot_confusion_matrix(
        y_true, y_pred,
        title="ResNet18 Confusion Matrix",
        save_path=RESULTS_DIR / "figure4_cnn_confusion_matrix.png",
        show=False,
    )
    plot_roc_curve(
        y_true, y_prob,
        title="ResNet18 ROC Curve",
        save_path=RESULTS_DIR / "figure5_cnn_roc_curve.png",
    )

    print("\nSaved:")
    print("  results/figure4_cnn_confusion_matrix.png")
    print("  results/figure5_cnn_roc_curve.png")


if __name__ == "__main__":
    main()
