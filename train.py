"""
Script entry point for running training on the dataset.
"""

from cse144_final_project.dataset import get_dataloaders
from cse144_final_project.model import build_model
from cse144_final_project.train import fit, apply_unfreezing_strategy
from cse144_final_project.utils import set_seed, make_plots

import argparse
from pathlib import Path

import torch
import torch.nn as nn

import logging
import datetime


def parse_args():
    parser = argparse.ArgumentParser(description="Train a model on the CSE144 dataset")
    parser.add_argument("--datadir", type=Path, default="./data/train", help="Path to training data directory")
    parser.add_argument("--outdir", type=Path, default="./outputs/checkpoints", help="Directory to save model checkpoints")
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs")
    parser.add_argument("--unfreeze-classifier-layers", type=int, default=1, help="Number of classifier head layers to unfreeze for training. Default is 0, which means all layers are frozen.")
    parser.add_argument("--unfreeze-backbone-layers", type=int, default=0, help="Number of backbone layers to unfreeze for training. Default is 0, which means all layers are frozen.")
    return parser.parse_args()


def main():
    # https://docs.pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html

    x = datetime.datetime.now()
    now = x.strftime("%d") + "_" + x.strftime("%m") + "_" + x.strftime("%y") + "_" + x.strftime("%X")
    logfile = now + "_log.txt"
    logging.basicConfig(filename=logfile, level=logging.INFO)
    
    args = parse_args()

    set_seed(42)

    # torch.accelerator is a new API and isn't shipped with all versions of PyTorch. If it's not available, we can fall back to the traditional device selection method.
    if hasattr(torch, 'accelerator') and torch.accelerator.is_available(): # type: ignore
        device = torch.device(torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else 'cpu') # type: ignore
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Assuming that we are on a CUDA machine, this should print a CUDA device:
    logging.info(f'Your device is: {device}')

    # create DataLoaders with get_dataloaders() function from dataset.py
    train_loader, val_loader = get_dataloaders(data_dir=args.datadir, batch_size=32, num_workers=2, shuffle=True)
    
    # build the neural network with build_model() function from model.py
    net = build_model(num_classes=100)
    net = net.to(device)

    # unfreeze layers for training
    apply_unfreezing_strategy(net, classifier_layers=args.unfreeze_classifier_layers, backbone_layers=args.unfreeze_backbone_layers)

    # define loss function
    criterion = nn.CrossEntropyLoss()
    
    # define optimizer
    feature_lr = 1e-4
    classifier_lr = 1e-3
    weight_decay = 1e-4 # this is L2 regularization
    optimizer = torch.optim.AdamW(net.get_parameter_groups(feature_lr, classifier_lr), weight_decay=weight_decay)

    # This is where the training loop happens.
    # History should be a dictionary with this format: {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
    # Each epoch should append entries to each of these lists

    # use the fit function to train it and you're done!
    logging.info('\nBeginning training now:')

    history = fit(
        net=net,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        criterion=criterion,
        device=device,
        epochs=args.epochs,
        save_path=args.outdir
    )

    # Quick summary output of training results. Temporarily here for now, will likley move to a seperate module later.
    best_epoch = max(
        range(len(history["val_acc"])),
        key=lambda i: history["val_acc"][i]
    )

    logging.info("\nTraining complete.")
    logging.info(f"Best epoch:      {best_epoch + 1}")
    logging.info(f"Best val acc:    {history['val_acc'][best_epoch]:.4f}")
    logging.info(f"Best val loss:   {history['val_loss'][best_epoch]:.4f}")
    logging.info(f"Train acc @ best:{history['train_acc'][best_epoch]:.4f}")
    logging.info(f"Checkpoint saved to: {args.outdir}")

    # create loss and accuracy plots for training/validation
    plotfile = now + ".png"
    make_plots(history, plotfile)

if __name__ == "__main__":
    main()