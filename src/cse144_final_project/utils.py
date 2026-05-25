"""
Utility functions for training, evaluation, and experiment reproducibility.

This module contains general-purpose helper functions that are shared across
the training and inference pipeline but do not belong to any single core
component such as the model, dataset, or training loop.

Typical functionality includes:
- Setting random seeds for reproducibility across PyTorch, NumPy, and Python
  random number generators.
- Saving and loading model checkpoints (model weights and optionally optimizer
  state) for training resumption and inference.
- Miscellaneous helper functions used across scripts (e.g., device handling,
  metric computation, logging helpers).

This module is intentionally kept framework-agnostic and should not contain:
- Model architecture definitions (see model.py)
- Data loading logic (see dataset.py)
- Training loop logic (see train.py)

The goal is to centralize shared "glue code" that supports experimentation
without coupling it to any specific model or dataset implementation.
"""

from pathlib import Path
import random
import numpy as np
import torch
import os
import matplotlib.pyplot as plt

def set_seed(seed: int = 42) -> None:
    """
    Sets the random seed for experiment reproducibility.

    This ensures deterministic behavior across Python, NumPy, and PyTorch
    (including CUDA where possible).

    Args:
        seed (int): Random seed value to use. Default is 42.
    """

    os.environ["PYTHONHASHSEED"] = str(seed)

    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False


def load_checkpoint(model: torch.nn.Module, checkpoint_path: Path) -> None:
    """
    Loads model weights from a checkpoint file.

    Args:
        model (torch.nn.Module): The model instance to load weights into.
        checkpoint_path (Path): Path to the checkpoint file (.pth or .pt) containing the model state_dict.
    """

    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    model.load_state_dict(checkpoint["model_state_dict"])

def make_plots(history, plotfile):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    # loss plot
    ax1.plot(history['train_loss'], label='train')
    ax1.plot(history['val_loss'], label='val')
    ax1.set_title('Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()

    # accuracy plot
    ax2.plot(history['train_acc'], label='train')
    ax2.plot(history['val_acc'], label='val')
    ax2.set_title('Accuracy')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy (%)')
    ax2.legend()

    plt.tight_layout()
    plt.savefig(plotfile)