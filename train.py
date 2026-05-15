import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE" 
# "multiple copies of the OpenMP runtime have been linked into the program...As an unsafe, unsupported, undocumented workaround you can set 
# the environment variable KMP_DUPLICATE_LIB_OK=TRUE to allow the program to continue to execute" lol

import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from PIL import Image

# custom Dataset class to make sure that samples have correct labels
class CSE144Dataset(Dataset):

    def __init__(self, root_dir, transform=None, resize=None):
        self.root_dir = root_dir
        self.transform = transform 
        self.resize = resize

        self.samples = []
        # go through all folders 0-99
        for folder_name in os.listdir(root_dir):
            label = int(folder_name)
            folder_path = os.path.join(root_dir, folder_name)
            # go through all images in folder
            for image_name in os.listdir(folder_path):
                # add tuple of image path and label to samples
                self.samples.append((os.path.join(folder_path, image_name), label))
        
        # TODO add data augmentation

    def __len__(self):
        return len(self.samples) # should be 1079
    
    def __getitem__(self, idx):
        path, label = self.samples[idx] 
        image = Image.open(path).convert("RGB")

        # resize
        if self.resize:
            new_h, new_w = self.resize
            image = transforms.functional.resize(image, (new_h, new_w))
        
        # transform (into tensor)
        if self.transform:
            image = self.transform(image)

        return image, label


def dataset_unittest():
    
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
    assert label in range(num_classes), f"Label {label} out of range"

    # print out some images and their labels to visually confirm

    loader = DataLoader(dataset, batch_size=16, shuffle=True)
    images, labels = next(iter(loader))

    grid = utils.make_grid(images[:16], normalize=True)
    plt.figure(figsize=(12, 6))
    plt.imshow(grid.permute(1, 2, 0))
    plt.title([str(l.item()) for l in labels[:16]])
    plt.axis("off")
    plt.show()


# We'll use this directory on my pc for the actual full-scale training
transform = transforms.ToTensor()
dataset = CSE144Dataset(r"C:\Users\eewilson\Documents\University\CSE144\finalproject_data\train", transform=transform, resize=(224, 224))
num_classes = 100

dataset_unittest()

