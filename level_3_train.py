import torch
import torchvision
import torchvision.transforms as transforms
import numpy as np
from torch.utils.data import random_split, DataLoader
import torch.nn.functional as F
import torch.nn as nn
import copy


#  Config for Level 3

MALICIOUS_CLIENT_ID = 3       # which client becomes malicious
ATTACK_START_ROUND = 3        # from this global epoch onward (1-based)
Z_THRESHOLD = 2.5             # outlier threshold (mean + Z*std)



#  CNN model (same as level 1)

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



#  Data loading

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


def partition_dataset(dataset, n_clients=10):
    split_size = len(dataset) // n_clients
    return random_split(dataset, [split_size] * n_clients)



#  Client-side local training

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
    # model updated in place



#  Malicious client behavior

def make_malicious_update(client_model):
    """
    Overwrite the client's parameters with random values
    to simulate a poisoned / falsified update.
    """
    with torch.no_grad():
        for p in client_model.parameters():
            # Random noise of similar scale to the parameter std (or 1 if std=0)
            scale = p.data.std().item()
            if scale == 0:
                scale = 1.0
            noise = torch.randn_like(p.data) * scale * 5.0   # amplified noise
            p.data.copy_(noise)



#  Helper: flatten model params

def flatten_params(model):
    """
    Flatten model parameters into a single 1D tensor.
    Used for distance-based outlier detection.
    """
    return torch.cat([p.data.view(-1) for p in model.parameters()])



#  Malicious client detection

def detect_malicious_clients(global_model, client_models, z_threshold=Z_THRESHOLD):
    """
    Detect malicious clients by looking for outliers in the
    distance between each client's weights and the previous global model.
    Returns a list of suspicious client indices.
    """
    global_vec = flatten_params(global_model)
    dists = []

    for idx, m in enumerate(client_models):
        client_vec = flatten_params(m)
        dist = torch.norm(client_vec - global_vec).item()
        dists.append(dist)

    dists_tensor = torch.tensor(dists)
    mean = dists_tensor.mean().item()
    std = dists_tensor.std(unbiased=False).item()

    if std == 0:
        # all clients identical → no outliers
        threshold = mean * 1.2
    else:
        threshold = mean + z_threshold * std

    suspicious = [idx for idx, d in enumerate(dists) if d > threshold]

    print(f"  Distances from global: {[round(d, 4) for d in dists]}")
    print(f"  Detection threshold: {threshold:.4f}")
    if suspicious:
        print(f"  [Detection] Suspicious clients this round: {suspicious}")
    else:
        print(f"  [Detection] No suspicious clients this round.")

    return suspicious



#  Server-side aggregation

def server_aggregate(global_model, client_models, malicious_clients=None):
    """
    FedAvg with robustness: ignore clients flagged as malicious.
    """
    if malicious_clients is None:
        malicious_clients = set()
    else:
        malicious_clients = set(malicious_clients)

    all_indices = list(range(len(client_models)))
    used_indices = [i for i in all_indices if i not in malicious_clients]

    # Fallback: if everything got flagged, use all clients (don't break training)
    if len(used_indices) == 0:
        print("  [Warning] All clients flagged malicious; using all clients for this round.")
        used_indices = all_indices

    print(f"  Using clients for aggregation: {used_indices}")

    global_dict = global_model.state_dict()
    new_state_dict = copy.deepcopy(global_dict)

    with torch.no_grad():
        for key, param in global_dict.items():
            client_params = [client_models[i].state_dict()[key] for i in used_indices]

            if param.dtype.is_floating_point:
                stacked = torch.stack(client_params, dim=0)
                new_state_dict[key] = torch.mean(stacked, dim=0)
            else:
                # e.g., num_batches_tracked in BatchNorm
                new_state_dict[key] = client_params[0]

        # Update global model
        global_model.load_state_dict(new_state_dict)

        # Broadcast back to all clients
        for m in client_models:
            m.load_state_dict(global_model.state_dict())



#  Evaluation

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



#  Federated Learning process

def federated_learning(n_clients, global_epochs, local_epochs):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Data
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    train_dataset, test_dataset = load_data(transform)

    client_datasets = partition_dataset(train_dataset, n_clients)
    client_loaders = [
        DataLoader(dataset, batch_size=50, shuffle=True)
        for dataset in client_datasets
    ]
    test_loader = DataLoader(test_dataset, batch_size=1000, shuffle=False)

    # Models
    global_model = ConvNet().to(device)
    client_models = [copy.deepcopy(global_model) for _ in range(n_clients)]

    # Optimizers
    optimizers = [
        torch.optim.Adam(model.parameters(), lr=0.0005)
        for model in client_models
    ]

    confirmed_malicious = set()
    print(f"Malicious client (ground truth): {MALICIOUS_CLIENT_ID}, "
          f"becomes malicious from round {ATTACK_START_ROUND}.")

    # FL rounds
    for global_epoch in range(global_epochs):
        print(f'\n=== Global Epoch {global_epoch + 1}/{global_epochs} ===')

        # Local training on each client
        for client_idx in range(n_clients):
            client_update(
                client_models[client_idx],
                optimizers[client_idx],
                client_loaders[client_idx],
                device,
                local_epochs
            )

            # Simulate attack: from ATTACK_START_ROUND onward,
            # the malicious client overwrites its weights with bogus ones.
            if (global_epoch + 1) >= ATTACK_START_ROUND and client_idx == MALICIOUS_CLIENT_ID:
                make_malicious_update(client_models[client_idx])
                print(f"  [Attack] Client {client_idx} has sent a malicious update this round.")

        # Detect malicious clients based on their updates
        suspicious = detect_malicious_clients(global_model, client_models)
        for idx in suspicious:
            # Keep track of clients flagged over time
            confirmed_malicious.add(idx)
            if idx == MALICIOUS_CLIENT_ID:
                print(f"  --> Correctly flagged the true malicious client: {idx}")

        # Aggregate while ignoring confirmed malicious clients
        server_aggregate(global_model, client_models, malicious_clients=confirmed_malicious)

        # Evaluate global model
        test_accuracy = test_model(global_model, test_loader, device)
        print(f'  Global Model Test Accuracy after round {global_epoch + 1}: {test_accuracy:.4f}')

    # Save final global model
    torch.save(global_model.state_dict(), 'federated_model_level3.pth')
    print("\nFederated learning with malicious client detection completed.")
    print(f"Final flagged malicious clients: {sorted(list(confirmed_malicious))}")


if __name__ == '__main__':
    # IMPORTANT: assignment says training runs for 10 epochs total.
    # Here: 10 global epochs, 1 local epoch per round.
    federated_learning(n_clients=10, global_epochs=10, local_epochs=1)
