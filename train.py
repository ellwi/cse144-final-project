"""
Script entry point for running training on the dataset.
"""

from cse144_final_project.dataset import get_dataloaders
from cse144_final_project.model import build_model
from cse144_final_project.train import fit, apply_freezing_strategy
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
    # https://docs.pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html
    
    args = parse_args()

    set_seed(42)

    device = torch.device(torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else 'cpu') # type: ignore
    # Assuming that we are on a CUDA machine, this should print a CUDA device:
    print(f'Your device is: {device}')

    # create DataLoaders with get_dataloaders() function from dataset.py
    train_loader, val_loader = get_dataloaders(
        data_dir=args.datadir,
        batch_size=32,
        num_workers=2
        shuffle=True
    )
    
    # build the neural network with build_model() function from model.py
    net = build_model(num_classes=100)
    net = net.to(device)


    # define loss function
    criterion = nn.CrossEntropyLoss()
    
    # define optimizer
    feature_lr = 1e-4
    classifier_lr = 1e-3
    weight_decay = 1e-4 # this is L2 regularization
    optimizer = torch.optim.AdamW([
        {'params': net.features.parameters(), 'lr': feature_lr},
        {'params': net.classifier.parameters(), 'lr': classifier_lr}
    ], weight_decay=weight_decay)

    # This is where the training loop happens.
    # History should be a dictionary with this format: {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
    # Each epoch should append entries to each of these lists

    # use the fit function to train it and you're done!
    print('\nBeginning training now:')

    path = r"C:\Users\eewilson\Documents\University\CSE144\finalproject_data\train"
    history = fit(
        net=net,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        criterion=criterion,
        device=device,
        epochs=10,
        save_path=path
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