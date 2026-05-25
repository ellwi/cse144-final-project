"""
Load data from root directory and create Dataset class.
Separate into train and validation Datasets.
Create Dataloaders. 


Job: Turn raw image folders into model-ready batches

This file is responsible for everything related to data access.

Responsibilities:
    Read image paths from train/ and test/
    Map folder names → integer labels (0–99)
    Define Dataset class
    Create train/val splits (or interface with split files later)
    Return (image_tensor, label) pairs
What it SHOULD NOT do:
    No model logic
    No training loops
    No augmentation definitions (those go in transforms.py)

Think: “data loading + labeling only”
"""

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE" 
# "multiple copies of the OpenMP runtime have been linked into the program...As an unsafe, unsupported, undocumented workaround you can set 
# the environment variable KMP_DUPLICATE_LIB_OK=TRUE to allow the program to continue to execute" lol

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from PIL import Image
from sklearn.model_selection import train_test_split
from torchvision.models import efficientnet_v2_s, EfficientNet_V2_S_Weights
import logging

# =========================================
# Data preprocessing/Setup
# =========================================


# custom Dataset class to make sure that samples have correct labels
# https://blog.roboflow.com/pytorch-custom-dataset/
# https://docs.pytorch.org/tutorials/beginner/basics/data_tutorial.html

def collect_samples(root_dir):
    """
    Grabs the images and folder labels from root and creates a list of each
    """
    images = []
    labels = []
    for folder_name in os.listdir(root_dir):
            label = int(folder_name)
            folder_path = os.path.join(root_dir, folder_name)
            # go through all images in folder
            for image_name in os.listdir(folder_path):
                images.append(os.path.join(folder_path, image_name))
                labels.append(label)
    return images, labels


class CSE144Dataset(Dataset):
    """
    Defines a Dataset class, needed for pytorch to handle data loading.
    Applies any needed transforms to images.
    """
    def __init__(self, images, labels, transform=None):
        self.images = images
        self.labels = labels
        self.transform = transform 
        
        # TODO add data augmentation

    def __len__(self):
        return len(self.images) # should be 1079
    
    def __getitem__(self, idx):
        path = self.images[idx] 
        image = Image.open(path).convert("RGB")

        # transform 
        if self.transform:
            image = self.transform(image)

        return image, self.labels[idx]


def dataset_unittest(dataset):
    """
    Unit test for checking dataset processing.
    Print out statistics about a Dataset, as well as a grid of labeled images.
    """
    print(f"Dataset size: {len(dataset)}")
    # Grab a single sample
    sample = dataset[25]
    image, label = sample

    # Check its properties
    print(f"Image type: {type(image)}")
    print(f"Image shape: {image.shape}")   # e.g. torch.Size([3, 224, 224])
    print(f"Image dtype: {image.dtype}")
    print(f"Label: {label}, type: {type(label)}")

    # Sanity checks
    assert image.shape[0] == 3, "Expected 3 channels (RGB)"
    num_classes = 100
    assert label in range(num_classes), f"Label {label} out of range"

    # print out some images and their labels to visually confirm

    loader = DataLoader(dataset, batch_size=16, shuffle=True)
    images, labels = next(iter(loader))

    grid = utils.make_grid(images[:16], normalize=True)
    plt.figure(figsize=(12, 6))
    plt.imshow(grid.permute(1, 2, 0))
    plt.title([str(l.item()) for l in labels[:16]]) # type: ignore
    plt.axis("off")
    plt.show()

def get_datasets(path):

    # collect lists of images and labels from directory 
    images, lables = collect_samples(path)

    # split dataset into train and validate
    # https://www.geeksforgeeks.org/deep-learning/how-to-split-a-dataset-using-pytorch/

    train_images, val_images, train_labels, val_labels = train_test_split(
        images, lables, test_size=0.2, stratify=lables, random_state=7
    )
    logging.info("Number of training samples:", len(train_images))
    logging.info("Number of validation samples:", len(val_images))

    # define transforms for training and validation data.
    # use the transforms that pytorch packages with pre-trained models to get 
    # correct resize and normalization ("preprocess").
    # https://docs.pytorch.org/vision/stable/models.html
    # https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.efficientnet_v2_s.html#torchvision.models.efficientnet_v2_s

    weights = EfficientNet_V2_S_Weights.DEFAULT
    preprocess = weights.transforms()

    train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(),
        preprocess
        ])
    val_transform = transforms.Compose([
        preprocess
        ])
    
    # create training and validation datasets
    train_dataset = CSE144Dataset(train_images, train_labels, transform=train_transform)
    val_dataset = CSE144Dataset(val_images, val_labels, transform=val_transform)

    return train_dataset, val_dataset

def get_dataloaders(data_dir, batch_size=32, num_workers=2, shuffle=True):
    """
    
    """
    train_dataset, val_dataset = get_datasets(data_dir)
    # create DataLoaders
    # https://docs.pytorch.org/tutorials/beginner/basics/data_tutorial.html
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, num_workers=num_workers, shuffle=shuffle)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, num_workers=num_workers, shuffle=False)

    return train_dataloader, val_dataloader
