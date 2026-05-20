"""
Script entry point for running training on the dataset.
"""

from cse144_final_project.dataset import get_dataloaders
from cse144_final_project.model import build_model
from cse144_final_project.train import fit
from cse144_final_project.utils import set_seed

import torch
import torch.nn as nn
import torch.optim as optim

def main():
    set_seed(42)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_loader, val_loader = get_dataloaders(
        data_dir="data/train",
        batch_size=32,
        num_workers=2
    )

    model = build_model(num_classes=100, pretrained=True)
    model = model.to(device)

    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=1e-4)

    history = fit(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        loss_fn=loss_fn,
        device=device,
        epochs=10,
        save_path="checkpoints/best.pt"
    )

if __name__ == "__main__":
    main()