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


import dataset
import model
import torch
import torch.nn as nn
import time

def fit(net=net,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        criterion=criterion,
        device=device,
        epochs=10,
        save_path=args.outdir):
    
    # track time
    start = time.time()

    # training and validation loop 
    performance = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
    for epoch in range(epochs):
        # train
        print(f'Entering training epoch {epoch}...')
        train_loss, train_accuracy = train_one_epoch(device, net, train_loader, optimizer, criterion)
        print(f'[epoch {epoch}] loss: {train_loss / len(train_loader):.3f}')

        # validate
        val_loss, val_accuracy = validate(device, net, val_loader, criterion)
        print(f'[epoch {epoch}] accuracy: {val_accuracy} %')

    elapsed = time.time() - start
    print(f'Finished Training in {elapsed/60:.1f} minutes')

    if save_path:
        torch.save(net.state_dict(), save_path)
    
    # return the trained version of the network
    return net

def train_one_epoch(device, net, train_loader, optimizer, criterion):
    """
    Function for training a neural network for specified number of epochs.
    Specify an output path to save the trained model to disk.
    """
    correct = 0
    total = 0

    running_loss = 0.0
    running_accuracy = 

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

def main():
    return

if __name__ == '__main__':
    main()


def apply_freezing_strategy(model: torch.nn.Module) -> None:
    """
    Apply a freezing strategy to the model's parameters.

    This implementation freezes all parameters except the final classification head. Pytorch models typically have an attribute like
    `model.fc` or `model.classifier` that contains the final linear layer(s) responsible for classification. The rest of the model (the "backbone") is frozen to preserve pretrained features.
    """
    
    # freeze everything
    for p in model.parameters():
        p.requires_grad = False

    # unfreeze head only
    if hasattr(model, "fc"):
        for p in model.fc.parameters():
            p.requires_grad = True
    if hasattr(model, "classifier"):
        for p in model.classifier.parameters():
            p.requires_grad = True