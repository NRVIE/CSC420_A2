import os
import torch
import pandas as pd
import numpy as np
import torchvision
import torch.nn as nn
from torch.utils.data import Dataset, random_split, DataLoader
import matplotlib.pyplot as plt
import torch.nn.functional as F
from torchvision.utils import make_grid
from torchvision.datasets import ImageFolder
import torchvision.transforms as transforms
import torchvision.models as models
from PIL import Image
from collections import OrderedDict

dataset = ImageFolder('./DBI')

# Define classes
breeds = [n for n in dataset.classes]

random_seed = 45
torch.manual_seed(random_seed)

test_pct = 0.3
test_size = int(len(dataset) * test_pct)
dataset_size = len(dataset) - test_size

val_pct = 0.1
val_size = int(dataset_size * val_pct)
train_size = dataset_size - val_size

train_ds, val_ds, test_ds = random_split(dataset, [train_size, val_size, test_size])


class DogBreedDataset(Dataset):

    def __init__(self, ds, transform=None):
        self.ds = ds
        self.transform = transform

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        img, label = self.ds[idx]
        if self.transform:
            img = self.transform(img)
            return img, label


imagenet_stats = ([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

train_transform = transforms.Compose([
    #    transforms.Resize((224, 224)),
    transforms.Resize((256, 256)),
    transforms.RandomCrop(224, padding=4, padding_mode='reflect'),
    transforms.RandomHorizontalFlip(p=0.3),
    transforms.RandomRotation(degrees=30),
    transforms.ToTensor(),
    #    transforms.Normalize(*imagenet_stats, inplace=True)

])

val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    #    transforms.Normalize(*imagenet_stats, inplace=True)
])

test_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    #    transforms.Normalize(*imagenet_stats, inplace=True)
])

train_dataset = DogBreedDataset(train_ds, train_transform)
val_dataset = DogBreedDataset(val_ds, val_transform)
test_dataset = DogBreedDataset(test_ds, test_transform)

def main():
    img, label = train_ds[6]
    print(dataset.classes[label])
    plt.imshow(img)
    print(type(img))

    img, label = train_dataset[6]
    print(label)
    plt.imshow(img.permute(1, 2, 0))

if __name__ == "__main__":
    main()
