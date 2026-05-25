"""
src/train.py

Responsibilities:
    Run training loop
    Run validation loop
    Save checkpoints
    Track best model
Functions:
    fit()
    train_one_epoch()
    validate()
    apply_unfreezing_strategy()S
    unfreeze_from_end()
"""

import torch
import torch.nn as nn
import time
from cse144_final_project.model import BaseTransferModel
from pathlib import Path
import logging

def fit(net, train_loader, val_loader, optimizer, criterion, device, epochs, save_path):
    """
    Fits a neural network to input data from train_loader and val_loader.
    Used by train.py wrapper, which defines all parameters. 
    """
    # track time
    start = time.time()
    best_accuracy = 0.0
    
    # training and validation loop 
    performance = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
    for epoch in range(epochs):
        # 1. train
        logging.info(f'Entering training epoch {epoch}...')
        train_loss, train_accuracy = train_one_epoch(device, net, train_loader, optimizer, criterion)
        logging.info(f'[epoch {epoch}] training loss: {train_loss:.3f}')
        logging.info(f'[epoch {epoch}] training accuracy: {train_accuracy} %')

        # 2. validate
        val_loss, val_accuracy = validate(device, net, val_loader, criterion)
        logging.info(f'[epoch {epoch}] validation loss: {val_loss:.3f}')
        logging.info(f'[epoch {epoch}] validation accuracy: {val_accuracy} %')

        # 3. record in performance dictionary
        performance["train_loss"].append(train_loss)
        performance["train_acc"].append(train_accuracy)
        performance["val_loss"].append(val_loss)
        performance["val_acc"].append(val_accuracy)

         # 4 keep track of best model
        if val_accuracy > best_accuracy:
            best_accuracy = val_accuracy
            if save_path:
                save_fp = Path(save_path)
                save_fp.mkdir(parents=True, exist_ok=True)
                save_fp = save_fp / "checkpoint.pth"
                logging.info(f'Saving model checkpoint to: {save_fp}')
                torch.save(net.state_dict(), save_fp)

    elapsed = time.time() - start
    logging.info(f'Finished Training in {elapsed/60:.1f} minutes')        
    
    # return the performance dictionary
    return performance

def train_one_epoch(device, net, train_loader, optimizer, criterion):
    """
    Trains a neural network for one epoch; used by fit().
    """
    net.train()

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
    running_loss = running_loss / len(train_loader)

    return running_loss, accuracy


def validate(device, net, val_loader, criterion):
    """
    Validates a neural network's performance; used by fit(). 
    """
    net.eval()

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
    running_loss = running_loss / len(val_loader)
    
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