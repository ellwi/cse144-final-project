"""
Script entry point for running training on the dataset.
"""

from cse144_final_project.dataset import get_dataloaders
from cse144_final_project.model import build_model
from cse144_final_project.train import fit, apply_unfreezing_strategy
from cse144_final_project.utils import set_seed, make_plots
from cse144_final_project.config import load_config

import argparse
from pathlib import Path

import torch
import torch.nn as nn
from torch.optim.lr_scheduler import CosineAnnealingLR

import os

import os


def parse_args():
    parser = argparse.ArgumentParser(description="Train a model on the CSE144 dataset")
    parser.add_argument("--config", type=Path, help="Path to YAML config file", required=True)


def main():
    # https://docs.pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html
    
    args = parse_args()
    config = load_config(args.config)

    set_seed(config.training.seed)

    # torch.accelerator is a new API and isn't shipped with all versions of PyTorch. If it's not available, we can fall back to the traditional device selection method.
    if hasattr(torch, 'accelerator') and torch.accelerator.is_available(): # type: ignore
        device = torch.device(torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else 'cpu') # type: ignore
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Assuming that we are on a CUDA machine, this should print a CUDA device:
    print(f'Your device is: {device}')

    # create DataLoaders with get_dataloaders() function from dataset.py
    train_loader, val_loader = get_dataloaders(data_dir=config.data.data_dir, model=config.model.model_name, batch_size=config.data.batch_size, num_workers=config.data.num_workers, shuffle=config.data.shuffle)

    # build the neural network with build_model() function from model.py
    net = build_model(model=config.model.model_name, num_classes=config.model.num_classes)

    if config.model.checkpoint_path is not None:
        net.load_state_dict(torch.load(config.model.checkpoint_path, map_location=device))
        print(f"Loaded checkpoint from: {config.model.checkpoint_path}")
    
    net = net.to(device)

    # unfreeze layers for training
    apply_unfreezing_strategy(net, classifier_layers=config.model.unfreeze_classifier_layers, backbone_layers=config.model.unfreeze_backbone_layers)

    # define loss function
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    
    # define optimizer
    feature_lr = config.optimizer.feature_lr
    classifier_lr = config.optimizer.classifier_lr
    weight_decay = config.optimizer.weight_decay # this is L2 regularization
    optimizer = torch.optim.AdamW(net.get_parameter_groups(feature_lr, classifier_lr), weight_decay=weight_decay)

    # adaptive learning rate scheduler
    
    """scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',        # reduce when val_loss stops decreasing
        factor=0.5,        # halve the LR
        patience=5,        # wait x epochs before reducing
    )"""


    """scheduler = CosineAnnealingLR(
        optimizer,
        T_max=10,    # match your epoch count
        eta_min=1e-7  # floor — don't go to zero
    )"""

    # This is where the training loop happens.
    # History should be a dictionary with this format: {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
    # Each epoch should append entries to each of these lists

    # use the fit function to train it and you're done!
    print(f"optimizer={optimizer}")
    print(f"criterion={criterion}")

    print('\nBeginning training now:\n')
    history = fit(
        net=net,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        criterion=criterion,
        device=device,
        epochs=args.epochs,
        save_path=args.outdir,
        #scheduler=scheduler
    )

    # Quick summary output of training results. Temporarily here for now, will likley move to a seperate module later.
    best_epoch = max(
        range(len(history["val_acc"])),
        key=lambda i: history["val_acc"][i]
    )

    print("\nTraining complete.")
    print(f"Best epoch:      {best_epoch}")
    print(f"Best val acc:    {history['val_acc'][best_epoch]:.4f}")
    print(f"Best val loss:   {history['val_loss'][best_epoch]:.4f}")
    print(f"Train acc @ best:{history['train_acc'][best_epoch]:.4f}")
    print(f"Checkpoint saved to: {args.outdir}")

    # create loss and accuracy plots for training/validation
    make_plots(history, args.outdir)

if __name__ == "__main__":
    main()