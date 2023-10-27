from abc import ABC

from torchvision import transforms, datasets, models
import torch
from torch import optim, cuda
from torch.utils.data import DataLoader, Dataset, random_split
import torch.nn as nn

# Load dataset
dataset = {
    "dbi": datasets.ImageFolder('./DBI'),
    "sdd": datasets.ImageFolder('./SDD')
}

# Define classes
targets = [n for n in dataset['dbi'].classes]

# Image augment
train_trans = transforms.Compose([
    transforms.RandomResizedCrop(size=300, scale=(0.8, 1.0)),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.5),
    transforms.RandomRotation(degrees=15),
    transforms.Resize(size=256),  # Image net standards
    transforms.ToTensor(),
])

val_trans = transforms.Compose([
    transforms.Resize(256),
    transforms.ToTensor(),
])

test_trans = transforms.Compose([
    transforms.Resize(256),
    transforms.ToTensor(),
])

ce_loss = nn.CrossEntropyLoss()

# Training model
class DBI_CNN(nn.Module):
    def __init__(self):
        super(DBI_CNN, self).__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(16),
            nn.Conv2d(16, 16, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=1, padding=0, dilation=1, ceil_mode=False),
            nn.Conv2d(16, 8, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(8),
            nn.Conv2d(8, 8, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=1, padding=0, dilation=1, ceil_mode=False),
            nn.Dropout(p=0.5, inplace=False),
            nn.Linear(254 * 254 * 8, 32),
            nn.Dropout(p=0.5, inplace=False),
            nn.Softmax(1)
        )

    def forward(self, x):
        return self.net(x)

# Custom class Dataset
class CustomDataset(Dataset):
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


def accuracy(outputs, labels):
    _, preds = torch.max(outputs, dim=1)
    return torch.tensor(torch.sum(preds == labels).item() / len(preds))


def train_dbi_model(ds, batch_size=64, train_p=0.6, val_p=0.1, learning_rate=0.01):
    """Part 2: Task 2
    :param
    train_p: the portion of dataset been training set
    val_p: the portion of dataset been valid set
    learning_rate: learning rate for gradient decent
    """
    # Getting train set and validation set and test set
    train_size = int(len(dataset) * train_p)
    val_size = int(len(dataset) * val_p)
    test_size = len(dataset) - train_size - val_size
    train_ds, val_ds, test_ds = random_split(dataset, [train_size, val_size, test_size])

    # Assume we take 256*256 picture
    # Image augmentaion
    train_dataset = CustomDataset(train_ds, train_trans)
    val_dataset = CustomDataset(val_ds, val_trans)
    test_dataset = CustomDataset(test_ds, test_trans)

    # Define DataLoader
    train_dl = DataLoader(train_dataset, batch_size=batch_size)
    val_dl = DataLoader(val_dataset, batch_size=batch_size*2)
    test_dl = DataLoader(test_dataset, batch_size=batch_size*2)


    # training model
    model = DBI_CNN()
    # Classifier
    # TODO: implement classifier
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=learning_rate,
    )
