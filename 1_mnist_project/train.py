import torch
import torch.nn as nn
import torch.optim as optim

from dataloader4level1 import load_data


class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.relu = nn.ReLU()
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.pool(x)
        x = self.relu(self.conv2(x))
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = self.relu(self.fc1(x))
        return self.fc2(x)


def extract_loaders(loaders):
    """Make the dataloader format flexible."""
    if len(loaders) == 2:
        return loaders[0], loaders[1]
    if len(loaders) == 3:
        return loaders[0], loaders[2]
    return loaders[0], loaders[-1]


def train_model(epochs=5, lr=1e-3, path="mnist_model.pth"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    # Call without batch_size (your error)
    loaders = load_data()
    train_loader, test_loader = extract_loaders(loaders)

    model = SimpleCNN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    for epoch in range(epochs):
        model.train()
        total, correct, running_loss = 0, 0, 0.0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * labels.size(0)
            _, preds = outputs.max(1)
            total += labels.size(0)
            correct += (preds == labels).sum().item()

        print(
            f"Epoch {epoch+1}/{epochs} | "
            f"Loss: {running_loss/total:.4f} | Acc: {correct/total:.4f}"
        )

    torch.save(model.state_dict(), path)
    print(f"Saved model to {path}")


if __name__ == "__main__":
    train_model()
