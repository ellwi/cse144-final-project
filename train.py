"""
Script entry point for running training on the dataset.
"""

from cse144_final_project.dataset import get_dataloaders
from cse144_final_project.model import build_model
from cse144_final_project.train import fit
from cse144_final_project.utils import set_seed

import argparse
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim


def parse_args():
    parser = argparse.ArgumentParser(description="Train a model on the CSE144 dataset")
    parser.add_argument("--datadir", type=Path, default="./data/train", help="Path to training data directory")
    parser.add_argument("--outdir", type=Path, default="./outputs/checkpoints", help="Directory to save model checkpoints")
    return parser.parse_args()


def main():
    args = parse_args()

    set_seed(42)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_loader, val_loader = get_dataloaders(
        data_dir=args.datadir,
        batch_size=32,
        num_workers=2
    )
    
    # Model should be type nn.Module
    model = build_model(num_classes=100, pretrained=True)
    model = model.to(device)

    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=1e-4)

    # History should be a dictionary with this format: {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
    # Each epoch should append entries to each of these lists
    history = fit(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        loss_fn=loss_fn,
        device=device,
        epochs=10,
        save_path=args.outdir
    )

    # Quick summary output of training results. Temporarily here for now, will likley move to a seperate module later.
    best_epoch = max(
        range(len(history["val_acc"])),
        key=lambda i: history["val_acc"][i]
    )

    print("\nTraining complete.")
    print(f"Best epoch:      {best_epoch + 1}")
    print(f"Best val acc:    {history['val_acc'][best_epoch]:.4f}")
    print(f"Best val loss:   {history['val_loss'][best_epoch]:.4f}")
    print(f"Train acc @ best:{history['train_acc'][best_epoch]:.4f}")
    print(f"Checkpoint saved to: {args.outdir}")

if __name__ == "__main__":
    main()