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
import torch

# create training and validation datasets
train_dataset = dataset.CSE144Dataset(dataset.train_images, dataset.train_labels, transform=dataset.train_transform)
val_dataset = dataset.CSE144Dataset(dataset.val_images, dataset.val_labels, transform=dataset.val_transform)

# run unit tests
dataset.dataset_unittest(train_dataset)
dataset.dataset_unittest(val_dataset)

# create DataLoaders
# https://docs.pytorch.org/tutorials/beginner/basics/data_tutorial.html
train_dataloader = dataset.DataLoader(train_dataset, batch_size=32, shuffle=True)
val_dataloader = dataset.DataLoader(val_dataset, batch_size=32, shuffle=True)

# define optimizer
feature_lr = 1e-4
classifier_lr = 1e-3
weight_decay = 1e-4 # this is L2 regularization

optimizer = torch.optim.AdamW([
    {'params': blocks.parameters(), 'lr': feature_lr},
    {'params': blocks.parameters(), 'lr': classifier_lr}
], weight_decay=weight_decay)


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