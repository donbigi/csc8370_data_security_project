import torch
import torchvision
import torchvision.transforms as transforms
import numpy as np
from torch.utils.data import random_split
from torch.utils.data import DataLoader
import torch.nn.functional as F
import torch.nn as nn


def dataloader(train_dataset, test_dataset):

    #TODO: Set the length of the batch (number of samples per batch)
    batch_size = 32

    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=True
    )
   
    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=len(test_dataset),
        shuffle=True
    )
    print(f'training has:{len(train_loader)} batch of data!')
    # print(f'validation has:{len(vali_loader)} batch of data!')
    print(f'testing has:{len(test_loader)} batch of data!')
    return train_loader, test_loader


def load_data():
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])

    train_dataset = torchvision.datasets.MNIST(root="./data/mnist", train=True, download=True, transform=transform)
    test_dataset = torchvision.datasets.MNIST(root="./data/mnist", train=False, download=True, transform=transform)

    print("The number of training data:", len(train_dataset))
    print("The number of testing data:", len(test_dataset))

    return dataloader(train_dataset, test_dataset)


# TODO: Building CNN model
# TODO: The CNN you define should inherit from the nn.Module class and override the forward()
class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()
        # 1 × 28 × 28
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.bn1   = nn.BatchNorm2d(32)

        # 32 × 28 × 28
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2   = nn.BatchNorm2d(64)

        # 64 × 14 × 14 (after first pool)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3   = nn.BatchNorm2d(128)

        self.pool = nn.MaxPool2d(2, 2)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(0.5)

        # After:
        # conv1 -> conv2 -> pool -> conv3 -> pool
        # 28 → 28 → 14 → 14 → 7  (spatial)
        # Channels: 1 → 32 → 64 → 128
        # So features = 128 * 7 * 7 = 6272
        self.fc1 = nn.Linear(128 * 7 * 7, 256)
        self.fc2 = nn.Linear(256, 10)

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))          # 1 → 32
        x = self.relu(self.bn2(self.conv2(x)))          # 32 → 64
        x = self.pool(x)                                # 28×28 → 14×14

        x = self.relu(self.bn3(self.conv3(x)))          # 64 → 128
        x = self.pool(x)                                # 14×14 → 7×7

        x = x.view(x.size(0), -1)                       # flatten (N, 6272)
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

def main():
    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    learning_rate = 0.0005
    epoches = 3

    train_loader, test_loader = load_data()

    model = ConvNet().to(device)  # Instantiate this class
    lossFun = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    best_accuracy = 0.0
    best_model_path = 'best_model.pth'
    for epoch in range(epoches):
        running_loss = 0.0
        running_acc = 0.0
        epoches_loss = []

        model.train()  # The model starts the training step.

        for i, data in enumerate(train_loader):
            features = data[0].to(device)
            labels = data[1].to(device)

            preds = model(features)
            loss = lossFun(preds, labels)

            loss.backward()  # Backpropagation
            optimizer.step()
            optimizer.zero_grad()
            running_loss += loss.item()
            correct = 0
            total = 0
            _, predicted = torch.max(preds, 1)
            total = labels.size(0)  # the lenth of labels

            # accuracy of prediction
            correct = (predicted == labels).sum().item()
            accuracy = correct / total
            running_acc += accuracy

            if i % 100 == 99:
                print(
                    f'epoch:{epoch+1},index of train:{i+1},loss: {(running_loss/100):.6f},acc:{(running_acc/100):.2%}')
                running_loss = 0.0
                running_acc = 0.0

    with torch.no_grad():
        model.eval()
        val_accuracy = 0.0
        num_correct = 0
        num_samples = 0

        for val_features, val_labels in test_loader:

            # Evaluate valset performance
            val_features = val_features.to(device)
            val_labels = val_labels.to(device)
            valiprediction = model(val_features)
            values, val_predicted = torch.max(valiprediction, axis=1)
            num_correct += (val_predicted == val_labels).sum().item()
            num_samples += len(val_labels)
            val_accuracy = num_correct / num_samples

            if val_accuracy > best_accuracy:
                best_accuracy = val_accuracy
                torch.save(model.state_dict(), best_model_path)
                print("Best model saved with accuracy:", best_accuracy)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.load_state_dict(torch.load('best_model.pth'))
    model.eval()
    model.to(device)
    with torch.no_grad():
        num_correct = 0
        num_samples = 0
        for test_features, test_labels in test_loader:
            test_features = test_features.to(device)
            test_labels = test_labels.to(device)
            test_pred = model(test_features)

            values, test_indexes = torch.max(test_pred, axis=1)

            num_correct += (test_indexes == test_labels).sum().item()
            num_samples += len(test_labels)
        print("Accuracy:", num_correct / num_samples)
if __name__ == '__main__':
    main()
