from abc import ABC

from matplotlib import pyplot as plt
from torchvision import transforms, datasets, models
import torch
from torch import optim, cuda
from torch.utils.data import DataLoader, Dataset, random_split
import torch.nn as nn

random_seed = 78
torch.manual_seed(random_seed);
# Load dataset
dataset = {
    "dbi": datasets.ImageFolder('./DBI'),
    "sdd": datasets.ImageFolder('./SDD')
}

# Define classes
targets = [n for n in dataset['dbi'].classes]

# Whether to train on a gpu
train_on_gpu = cuda.is_available()
print(f'Train on gpu: {train_on_gpu}')

# Number of gpus
if train_on_gpu:
    gpu_count = cuda.device_count()
    print(f'{gpu_count} gpus detected.')
    if gpu_count > 1:
        multi_gpu = True
    else:
        multi_gpu = False

# Image augmentation
train_trans = transforms.Compose([
    transforms.RandomResizedCrop(size=300, scale=(0.8, 1.0)),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.5),
    transforms.RandomRotation(degrees=15),
    transforms.Resize((256, 256)),  # Image net standards
    transforms.ToTensor(),
])

val_trans = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
])

test_trans = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
])

ce_loss = nn.CrossEntropyLoss()


def accuracy(outputs, labels):
    _, preds = torch.max(outputs, dim=1)
    return torch.tensor(torch.sum(preds == labels).item() / len(preds))


class ImageClassificationBase(nn.Module):
    # training step
    def training_step(self, batch):
        img, labels = batch
        out = self(img)
        loss = ce_loss(out, labels)
        return loss

    # validation step
    def validation_step(self, batch):
        img, labels = batch
        out = self(img)
        loss = ce_loss(out, labels)
        acc = accuracy(out, labels)
        return {'val_acc': acc.detach(), 'val_loss': loss.detach()}

    # validation epoch end
    def validation_epoch_end(self, outputs):
        batch_losses = [x['val_loss'] for x in outputs]
        epoch_loss = torch.stack(batch_losses).mean()
        batch_accs = [x['val_acc'] for x in outputs]
        epoch_acc = torch.stack(batch_accs).mean()
        return {'val_loss': epoch_loss.item(), 'val_acc': epoch_acc.item()}

    # print result end epoch
    def epoch_end(self, epoch, result):
        print("Epoch [{}] : train_loss: {:.4f}, val_loss: {:.4f}, val_acc: {:.4f}".format(epoch,
                                                                                          result[
                                                                                              "train_loss"],
                                                                                          result[
                                                                                              "val_loss"],
                                                                                          result[
                                                                                              "val_acc"]))


# Training model
class DBI_CNN(ImageClassificationBase):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(16),
            nn.Conv2d(16, 16, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False),
            nn.Conv2d(16, 8, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(8),
            nn.Conv2d(8, 8, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False),
            nn.Dropout(p=0.5, inplace=False),
            nn.Flatten(),
            nn.Linear(64 * 64 * 8, 32),
            nn.Dropout(p=0.5, inplace=False),
            nn.Linear(32, 7),
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        return self.net(x)


class DBI_Resnet18(ImageClassificationBase):
    def __init__(self):
        super().__init__()
        # Replace last layer
        self.net = models.resnet18()
        num_ftrs = self.net.fc.in_features
        self.net.fc = nn.Sequential(
            nn.Linear(num_ftrs, 7),
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        return self.net(x)


class DBI_PretrainedResnet18(ImageClassificationBase):
    def __init__(self):
        super().__init__()
        # Replace last layer
        self.net = models.resnet18(pretrained=True)
        num_ftrs = self.net.fc.in_features
        self.net.fc = nn.Sequential(
            nn.Linear(num_ftrs, 7),
            nn.Softmax(dim=1)
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


def get_default_device():
    if torch.cuda.is_available():
        return torch.device('cuda')
    else:
        return torch.device('cpu')


def to_device(data, device):
    if isinstance(data, (list, tuple)):
        return [to_device(d, device) for d in data]
    else:
        return data.to(device, non_blocking=True)


class DeviceDataLoader:
    def __init__(self, dl, device):
        self.dl = dl
        self.device = device

    def __len__(self):
        return len(self.dl)

    def __iter__(self):
        for batch in self.dl:
            yield to_device(batch, self.device)


def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']


def train_dbi_model(epoch, ds, model, device, batch_size=64, train_p=0.6, val_p=0.1,
                    max_lr=0.01):
    """Part 2: Task 2
    :param
    train_p: the portion of dataset been training set
    val_p: the portion of dataset been valid set
    learning_rate: learning rate for gradient decent
    """

    # Getting train set and validation set and test set
    train_size = int(len(ds) * train_p)
    val_size = int(len(ds) * val_p)
    test_size = len(ds) - train_size - val_size
    train_ds, val_ds, test_ds = random_split(ds, [train_size, val_size, test_size])

    # Assume we take 256*256 picture
    # Image augmentaion
    train_dataset = CustomDataset(train_ds, train_trans)
    val_dataset = CustomDataset(val_ds, val_trans)
    test_dataset = CustomDataset(test_ds, test_trans)

    # Define DataLoader
    train_dl = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                          pin_memory=True)
    val_dl = DataLoader(val_dataset, batch_size=batch_size, shuffle=True,
                        pin_memory=True)
    test_dl = DataLoader(test_dataset, batch_size=batch_size, shuffle=True,
                         pin_memory=True)
    train_dl = DeviceDataLoader(train_dl, device)
    val_dl = DeviceDataLoader(val_dl, device)
    test_dl = DeviceDataLoader(test_dl, device)

    # training model
    history = []
    # Define Optimizer
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=max_lr,
    )
    # set up one cycle lr scheduler
    sched = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr, epochs=epoch,
                                                steps_per_epoch=len(train_dl))
    for i in range(epoch):
        model.train()
        train_losses = []
        lrs = []
        # Training
        for batch in train_dl:
            # Forward
            loss = model.training_step(batch)
            train_losses.append(loss)

            # Backward
            optimizer.zero_grad()
            loss.backward()

            # Gradient descent
            optimizer.step()

            # record and update lr
            lrs.append(get_lr(optimizer))

            # modifies the lr value
            sched.step()

        # Validation phase
        result = evaluate(model, val_dl)
        result['train_loss'] = torch.stack(train_losses).mean().item()
        result['lrs'] = lrs
        result['train_acc'] = evaluate(model, train_dl)['val_acc']
        result['test_acc'] = evaluate(model, test_dl)['val_acc']
        model.epoch_end(i, result)
        history.append(result)

    return history, test_dl


@torch.no_grad()
def evaluate(model, val_loader):
    model.eval()
    outputs = [model.validation_step(batch) for batch in val_loader]
    return model.validation_epoch_end(outputs)


def main():
    model = DBI_PretrainedResnet18()
    device = get_default_device()
    to_device(model, device);
    history, test_dl = train_dbi_model(10, dataset['dbi'], model, device, max_lr=0.001)

    # Plot preparation
    val_loss = []
    train_loss = []
    val_acc = []
    train_acc = []
    test_acc = []
    time = list(range(len(history)))
    for h in history:
        val_loss.append(h['val_loss'])
        train_loss.append(h['train_loss'])
        val_acc.append(h['val_acc'])
        train_acc.append(h['train_acc'])
        test_acc.append(h['test_acc'])
    # Plotting Accuracy graph
    plt.plot(time, test_acc, c='red', label='test accuracy', marker='x')
    plt.plot(time, train_acc, c='blue', label='train accuracy', marker='x')
    plt.plot(time, val_acc, c='green', label='valid accuracy', marker='x')
    plt.title("accuracy vs. epochs")
    plt.xlabel('epochs')
    plt.ylabel('accuracy')
    plt.show()
    # Print test loss and test accuracy
    result = evaluate(model, test_dl)
    print(f"Test loss: {result['val_loss']}, Test Accuracy: {result['val_acc']}\n")
    # Print accuracy of the model in SDD
    test_dataset = CustomDataset(dataset['sdd'], test_trans)
    test_dl = DataLoader(test_dataset, batch_size=64, shuffle=True,
                         pin_memory=True)
    test_dl = DeviceDataLoader(test_dl, device)
    result = evaluate(model, test_dl)
    print(f"Accuracy on SDD: {result['val_acc']}\n")


if __name__ == "__main__":
    main()
