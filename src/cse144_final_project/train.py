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
    for epoch in range(epochs):
        # train
        print(f'Entering training epoch {epoch}...')
        net, running_loss = train_one_epoch(device, net, train_loader, optimizer, criterion)
        print(f'[epoch {epoch + 1}] loss: {running_loss / len(train_loader):.3f}')

        # validate
        validate(device, net, val_loader)

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
    running_loss = 0.0
    for i, data in enumerate(train_loader, 0):
        inputs, labels = data[0].to(device), data[1].to(device)

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # save running loss
        running_loss += loss.item()            
    return net, running_loss


def validate(device, net, val_loader):
    """
    Function for validating a neural network's performace. 
    """
    correct = 0
    total = 0

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

    print(f'Accuracy of the network on the {total} validation images: {100 * correct // total} %')

def main():
    # ======================================
    # https://docs.pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html
    # First test: train head only for 10 epochs
    # ======================================

    # build the neural network and give it gpu information
    print('Building model...')
    net = model.build_model()
    device = torch.device(torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else 'cpu') # type: ignore
    net.to(device)
    # Assuming that we are on a CUDA machine, this should print a CUDA device:
    print(f'Your device is: {device}')
    
    # run unit tests
    # dataset_unittest(train_dataset)
    # dataset_unittest(val_dataset)

    # create CSE144Datasets
    print('Generating training and validation datasets...')
    path = r"C:\Users\eewilson\Documents\University\CSE144\finalproject_data\train"
    train_dataset, val_dataset = dataset.get_datasets(path)

    # create DataLoaders
    # https://docs.pytorch.org/tutorials/beginner/basics/data_tutorial.html
    train_dataloader = dataset.DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_dataloader = dataset.DataLoader(val_dataset, batch_size=32, shuffle=True)

    # define optimizer
    feature_lr = 1e-4
    classifier_lr = 1e-3
    weight_decay = 1e-4 # this is L2 regularization
    optimizer = torch.optim.AdamW([
        {'params': net.features.parameters(), 'lr': feature_lr},
        {'params': net.classifier.parameters(), 'lr': classifier_lr}
    ], weight_decay=weight_decay)

    # define loss function
    criterion = nn.CrossEntropyLoss()

    # use the train function to train it and you're done!
    print('\nBeginning training now:')
    net = fit(device, net, train_dataloader, optimizer, criterion, epochs=10)


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