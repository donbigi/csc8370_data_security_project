import torch
import torchvision
import torchvision.transforms as transforms
import numpy as np
from torch.utils.data import random_split, DataLoader
import torch.nn.functional as F
import torch.nn as nn
import copy


# TODO: CNN model definition, same to the model in the level 1
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


# Load data (each client will load its own data in a real FL scenario)
def load_data(transform, datasets='MNIST'):
    if datasets == 'MNIST':
        train_dataset = torchvision.datasets.MNIST(
            root="./data/mnist", train=True, download=True, transform=transform)
        test_dataset = torchvision.datasets.MNIST(
            root="./data/mnist", train=False, download=True, transform=transform)
    else:
        train_dataset = torchvision.datasets.CIFAR10(
            root="./data/cifar-10-python", train=True, download=True, transform=transform)
        test_dataset = torchvision.datasets.CIFAR10(
            root="./data/cifar-10-python", train=False, download=True, transform=transform)

    return train_dataset, test_dataset


# Split the dataset into 'n_clients' partitions
def partition_dataset(dataset, n_clients=10):
    split_size = len(dataset) // n_clients
    return random_split(dataset, [split_size] * n_clients)


# TODO: define the client-side local training here
def client_update(client_model, optimizer, train_loader, device, epochs=1):
    """
    Perform local training on a client's data.
    """
    criterion = nn.CrossEntropyLoss()
    client_model.train()
    client_model.to(device)

    for _ in range(epochs):
        for data, labels in train_loader:
            data, labels = data.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = client_model(data)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
    # no need to return: the model is updated in place


# TODO: define the server-side aggregation of client models here
def server_aggregate(global_model, client_models):
    """
    FedAvg: average client model parameters into the global model,
    then broadcast global weights back to all clients.
    """
    # Get current global state
    global_dict = global_model.state_dict()
    new_state_dict = copy.deepcopy(global_dict)

    with torch.no_grad():
        for key, param in global_dict.items():
            # Collect the corresponding tensors from each client
            client_params = [m.state_dict()[key] for m in client_models]

            # Only average floating-point parameters; for others, just take the first
            if param.dtype.is_floating_point:
                stacked = torch.stack(client_params, dim=0)
                new_state_dict[key] = torch.mean(stacked, dim=0)
            else:
                # e.g., num_batches_tracked for BatchNorm (int64)
                new_state_dict[key] = client_params[0]

        # Load the averaged weights into the global model
        global_model.load_state_dict(new_state_dict)

        # Broadcast the new global weights back to each client
        for m in client_models:
            m.load_state_dict(global_model.state_dict())


# Test model on test dataset
def test_model(model, test_loader, device):
    model.eval()
    model.to(device)
    correct, total = 0, 0
    with torch.no_grad():
        for data, labels in test_loader:
            data, labels = data.to(device), labels.to(device)
            output = model(data)
            _, predicted = torch.max(output, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return correct / total


# Federated Learning process
def federated_learning(n_clients, global_epochs, local_epochs):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Set up the data transformation and load dataset
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    train_dataset, test_dataset = load_data(transform)

    # Partition the dataset for each client
    client_datasets = partition_dataset(train_dataset, n_clients)
    client_loaders = [
        DataLoader(dataset, batch_size=50, shuffle=True)  # TODO: change the batch size here
        for dataset in client_datasets
    ]
    test_loader = DataLoader(test_dataset, batch_size=1000, shuffle=False)  # TODO: change the batch size here

    # Initialize global model and n_clients client models
    global_model = ConvNet().to(device)
    client_models = [copy.deepcopy(global_model) for _ in range(n_clients)]

    # Optimizers for client models
    optimizers = [
        torch.optim.Adam(model.parameters(), lr=0.0005)  # TODO: change the learning rate here for each client model
        for model in client_models
    ]

    # Federated Learning process
    for global_epoch in range(global_epochs):
        print(f'Global Epoch {global_epoch + 1}/{global_epochs}')

        # Each client trains locally
        for client_idx in range(n_clients):
            client_update(
                client_models[client_idx],
                optimizers[client_idx],
                client_loaders[client_idx],
                device,
                local_epochs
            )

        # Server aggregates the models
        server_aggregate(global_model, client_models)

        # Evaluate global model on test dataset
        test_accuracy = test_model(global_model, test_loader, device)
        print(f'Global Model Test Accuracy after round {global_epoch + 1}: {test_accuracy:.4f}')

    # Save the final global model
    torch.save(global_model.state_dict(), 'federated_model.pth')
    print("Federated learning process completed.")


if __name__ == '__main__':
    federated_learning(n_clients=10, global_epochs=10, local_epochs=2)  # TODO: only change the number of global epochs and local epochs here
