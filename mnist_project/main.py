import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

# Use data loader
from dataloader4level1 import load_data


# Model: Simple CNN
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(x)
        x = x.view(x.size(0), -1)   # Flatten
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


# Training function

def train(model, train_loader, optimizer, criterion, device):
    model.train()
    running_loss = 0
    correct = 0
    total = 0

    for images, labels in train_loader:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)

        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, preds = torch.max(outputs, 1)
        correct += torch.sum(preds == labels).item()
        total += labels.size(0)

    return running_loss / len(train_loader), correct / total



# Evaluation function

def evaluate(model, test_loader, criterion, device):
    model.eval()
    with torch.no_grad():
        for images, labels in test_loader:  # only one batch (full test set)
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            _, preds = torch.max(outputs, 1)
            accuracy = (preds == labels).float().mean().item()

    return loss.item(), accuracy



# Main Training Loop

def main():

    # ---- Device selection for M2 Mac ----
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    print("Using device:", device)

    # ---- Load dataset using your loader ----
    train_loader, test_loader = load_data()

    # ---- Model, Loss, Optimizer ----
    model = CNN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    epochs = 5

    for epoch in range(1, epochs + 1):
        train_loss, train_acc = train(model, train_loader, optimizer, criterion, device)
        test_loss, test_acc = evaluate(model, test_loader, criterion, device)

        print(f"Epoch {epoch}/{epochs}")
        print(f"  Train Loss: {train_loss:.4f} | Train Acc: {train_acc*100:.2f}%")
        print(f"  Test  Loss: {test_loss:.4f}  | Test Acc: {test_acc*100:.2f}%")


if __name__ == "__main__":
    main()
