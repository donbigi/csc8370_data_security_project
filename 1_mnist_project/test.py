import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torch.nn as nn

# --- Same CNN Architecture Used During Training ---
class ConvNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2,2)

        # After 2 pools on 28×28:
        # 28 → 14 → 7, and 7*7*64 = 3136
        self.fc1 = nn.Linear(3136, 128)
        self.fc2 = nn.Linear(128, 10)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.pool(x)
        x = self.relu(self.conv2(x))
        x = self.pool(x)

        x = x.view(x.size(0), -1)  # flatten
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x


# --- Load MNIST Test Dataset ---
def load_test_data():
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    test_dataset = torchvision.datasets.MNIST(
        root="./data/mnist",
        train=False,
        download=True,
        transform=transform
    )

    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=len(test_dataset),  # all test images at once
        shuffle=False
    )
    
    return test_loader


# --- Evaluate the Trained Model ---
def evaluate_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    model = ConvNet().to(device)

    # Load weights (your new filename)
    model.load_state_dict(torch.load("mnist_model.pth", map_location=device))
    model.eval()

    test_loader = load_test_data()

    with torch.no_grad():
        correct = 0
        total = 0

        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)

            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total += len(labels)

        accuracy = correct / total
        print(f"Final Test Accuracy: {accuracy:.4f}")


if __name__ == "__main__":
    evaluate_model()
