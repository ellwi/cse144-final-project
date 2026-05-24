"""
Job: Run the training pipeline end-to-end

This is the main engine of the project.

Responsibilities:
    Load config
    Initialize dataset + dataloaders
    Build model
    Define optimizer + scheduler
    Run training loop
    Run validation loop
    Save checkpoints
    Track best model
Core loops:
    train_one_epoch()
    validate()
    main()

Think: “make the model learn”
"""

import torch
import torch.nn as nn
import time
from cse144_final_project.model import BaseTransferModel
from pathlib import Path

def fit(net, train_loader, val_loader, optimizer, criterion, device, epochs, save_path):
    
    # track time
    start = time.time()

    # training and validation loop 
    performance = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
    for epoch in range(epochs):
        # 1. train
        print(f'Entering training epoch {epoch}...')
        train_loss, train_accuracy = train_one_epoch(device, net, train_loader, optimizer, criterion)
        print(f'[epoch {epoch}] loss: {train_loss / len(train_loader):.3f}')

        # 2. validate
        val_loss, val_accuracy = validate(device, net, val_loader, criterion)
        print(f'[epoch {epoch}] accuracy: {val_accuracy} %')

        # 3. record in performance dictionary
        performance["train_loss"].append(train_loss)
        performance["train_acc"].append(train_accuracy)
        performance["val_loss"].append(val_loss)
        performance["val_acc"].append(val_accuracy)

    elapsed = time.time() - start
    print(f'Finished Training in {elapsed/60:.1f} minutes')

    if save_path:
        save_fp = Path(save_path) / "checkpoint.pth"
        print(f'Saving model checkpoint to: {save_fp}')
        torch.save(net.state_dict(), save_fp)
    
    # return the performance dictionary
    return performance

def train_one_epoch(device, net, train_loader, optimizer, criterion):
    """
    Function for training a neural network for specified number of epochs.
    Specify an output path to save the trained model to disk.
    """
    correct = 0
    total = 0

    running_loss = 0.0
    accuracy = 0.0

    for i, data in enumerate(train_loader, 0):
        inputs, labels = data[0].to(device), data[1].to(device)

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net(inputs)

        # predicted label is the output with max weight/energy
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # save loss and accuracy
        running_loss += loss.item()
    
    accuracy = 100 * correct // total
    return running_loss, accuracy


def validate(device, net, val_loader, criterion):
    """
    Function for validating a neural network's performace. 
    """
    correct = 0
    total = 0

    running_loss = 0.0

    # frozen layers because we're not training
    with torch.no_grad():
        for data in val_loader:
            images, labels = data[0].to(device), data[1].to(device)

            # run images through neural network 
            outputs = net(images)
            # predicted label is the output with max weight/energy
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            running_loss += criterion(outputs, labels).item()

    accuracy = 100 * correct // total
    return running_loss, accuracy


def apply_unfreezing_strategy(model: BaseTransferModel, classifier_layers: int = -1, backbone_layers: int = -1) -> None:
    """
    Apply a freezing strategy to the model's parameters.
    """ 

    if classifier_layers > 0:
        # unfreeze classifier head layers
        classifier_blocks = model.get_trainable_classifier_blocks()
        unfreeze_from_end(classifier_blocks, classifier_layers)
    
    if backbone_layers > 0:
        # unfreeze backbone layers
        backbone_blocks = model.get_trainable_backbone_blocks()
        unfreeze_from_end(backbone_blocks, backbone_layers)


def unfreeze_from_end(blocks: list[nn.Module], num_layers_to_unfreeze: int) -> None:
    """
    Unfreeze the last `num_layers_to_unfreeze` layers in the provided list of model blocks.
    """
    len_blocks = len(blocks)
    unfreeze_count = min(num_layers_to_unfreeze, len_blocks)
    for block in blocks[-unfreeze_count:]:
        for param in block.parameters():
            param.requires_grad = True