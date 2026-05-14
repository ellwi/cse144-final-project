import os
import torch
import pandas as pd
from skimage import io, transform
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils

# custom Dataset class to make sure that samples have correct labels
class CSE144Dataset(Dataset):

    def __init__(self, root_dir):
        self.root_dir = root_dir
        self.samples = []

        # go through all folders 0-99
        for folder_name in os.listdir(root_dir):
            label = int(folder_name)
            folder_path = os.path.join(root_dir, folder_name)
            # add tuple of image path and label to samples
            self.samples.append(os.path.join(root_dir, folder_path), label)

    def __len__(self):
        return len(self.samples) # should be 1000
    
    def __getitem__(self, idx):
        return self.samples[idx] # this returns image PATH and label, need to return image object?

