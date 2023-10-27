import os
import torch
# import pandas as pd
# import numpy as np
# import torchvision
# import torch.nn as nn
from torch.utils.data import Dataset, random_split, DataLoader
import matplotlib.pyplot as plt
# import torch.nn.functional as F
# from torchvision.utils import make_grid
from torchvision.datasets import ImageFolder
# import torchvision.transforms as transforms
# import torchvision.models as models
# from PIL import Image
# from collections import OrderedDict

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

def main():
    img, label = train_ds[6]
    print(dataset.classes[label])
    plt.imshow(img)
    print(type(img))

if __name__ == "__main__":
    main()
