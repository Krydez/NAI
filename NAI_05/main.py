"""
Sieci neuronowe dla klasyfikacji zbiorów:
- Sonar (skała vs mina)
- CIFAR-10
- Fashion MNIST
- FGVC Aircraft
- EuroSAT

pip:
  pip install torch torchvision numpy scikit-learn matplotlib seaborn
  python main.py <sonar|cifar10|fashion|aircraft|eurosat>
uv:
  uv run main.py <sonar|cifar10|fashion|aircraft|eurosat>

Autorzy: Hubert Jóźwiak, Kacper Olejnik
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split, Subset
from torchvision import datasets, transforms
import numpy as np
import argparse
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns


class SonarDataset(Dataset):
    def __init__(self, csv_file):
        data = np.loadtxt(csv_file, delimiter=",", dtype=str)
        self.X = data[:, :-1].astype(np.float32)
        labels = data[:, -1]
        self.y = np.array(
            [0 if label == "R" else 1 for label in labels], dtype=np.int64
        )

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


class SonarNN(nn.Module):
    def __init__(self, input_size=60, hidden_size1=128, hidden_size2=64, num_classes=2):
        super(SonarNN, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size1)
        self.bn1 = nn.BatchNorm1d(hidden_size1)
        self.dropout1 = nn.Dropout(0.3)
        self.fc2 = nn.Linear(hidden_size1, hidden_size2)
        self.bn2 = nn.BatchNorm1d(hidden_size2)
        self.dropout2 = nn.Dropout(0.3)
        self.fc3 = nn.Linear(hidden_size2, num_classes)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.bn1(self.fc1(x)))
        x = self.dropout1(x)
        x = self.relu(self.bn2(self.fc2(x)))
        x = self.dropout2(x)
        x = self.fc3(x)
        return x


class SonarNNSmall(nn.Module):
    def __init__(self, input_size=60, hidden_size1=32, hidden_size2=16, num_classes=2):
        super(SonarNNSmall, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size1)
        self.bn1 = nn.BatchNorm1d(hidden_size1)
        self.dropout1 = nn.Dropout(0.2)
        self.fc2 = nn.Linear(hidden_size1, hidden_size2)
        self.bn2 = nn.BatchNorm1d(hidden_size2)
        self.dropout2 = nn.Dropout(0.2)
        self.fc3 = nn.Linear(hidden_size2, num_classes)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.bn1(self.fc1(x)))
        x = self.dropout1(x)
        x = self.relu(self.bn2(self.fc2(x)))
        x = self.dropout2(x)
        x = self.fc3(x)
        return x


class RemappedDataset(Dataset):
    def __init__(self, subset, class_mapping):
        self.subset = subset
        self.class_mapping = class_mapping

    def __len__(self):
        return len(self.subset)

    def __getitem__(self, idx):
        img, label = self.subset[idx]
        return img, self.class_mapping[label]


class CIFAR10AnimalCNN(nn.Module):
    def __init__(self, num_classes=6):
        super(CIFAR10AnimalCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(128 * 4 * 4, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, num_classes)
        self.dropout = nn.Dropout(0.5)
        self.bn1 = nn.BatchNorm2d(32)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(128)

    def forward(self, x):
        x = self.pool(torch.relu(self.bn1(self.conv1(x))))
        x = self.pool(torch.relu(self.bn2(self.conv2(x))))
        x = self.pool(torch.relu(self.bn3(self.conv3(x))))
        x = x.view(-1, 128 * 4 * 4)
        x = self.dropout(torch.relu(self.fc1(x)))
        x = self.dropout(torch.relu(self.fc2(x)))
        x = self.fc3(x)
        return x


class FashionMNISTCNN(nn.Module):
    def __init__(self, num_classes=10):
        super(FashionMNISTCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(128 * 3 * 3, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, num_classes)
        self.dropout = nn.Dropout(0.5)
        self.bn1 = nn.BatchNorm2d(32)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(128)

    def forward(self, x):
        x = self.pool(torch.relu(self.bn1(self.conv1(x))))
        x = self.pool(torch.relu(self.bn2(self.conv2(x))))
        x = self.pool(torch.relu(self.bn3(self.conv3(x))))
        x = x.view(-1, 128 * 3 * 3)
        x = self.dropout(torch.relu(self.fc1(x)))
        x = self.dropout(torch.relu(self.fc2(x)))
        x = self.fc3(x)
        return x


class EuroSATCNN(nn.Module):
    def __init__(self, num_classes=10):
        super(EuroSATCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, 3, padding=1)
        self.conv2 = nn.Conv2d(64, 128, 3, padding=1)
        self.conv3 = nn.Conv2d(128, 256, 3, padding=1)
        self.conv4 = nn.Conv2d(256, 512, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(512 * 4 * 4, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, num_classes)
        self.dropout = nn.Dropout(0.5)
        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(128)
        self.bn3 = nn.BatchNorm2d(256)
        self.bn4 = nn.BatchNorm2d(512)

    def forward(self, x):
        x = self.pool(torch.relu(self.bn1(self.conv1(x))))
        x = self.pool(torch.relu(self.bn2(self.conv2(x))))
        x = self.pool(torch.relu(self.bn3(self.conv3(x))))
        x = self.pool(torch.relu(self.bn4(self.conv4(x))))
        x = x.view(-1, 512 * 4 * 4)
        x = self.dropout(torch.relu(self.fc1(x)))
        x = self.dropout(torch.relu(self.fc2(x)))
        x = self.fc3(x)
        return x


def train_epoch(model, loader, criterion, optimizer, device):
    """Train the model for one epoch.

    Args:
        model: Neural network model to train
        loader: DataLoader for training data
        criterion: Loss function
        optimizer: Optimizer for updating weights
        device: Device to run training on (CPU or CUDA)

    Returns:
        Tuple of (average_loss, accuracy_percentage)
    """
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    for inputs, labels in loader:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    return total_loss / len(loader), 100 * correct / total


def evaluate(model, loader, criterion, device):
    """Evaluate the model on a dataset.

    Args:
        model: Neural network model to evaluate
        loader: DataLoader for evaluation data
        criterion: Loss function
        device: Device to run evaluation on (CPU or CUDA)

    Returns:
        Tuple of (average_loss, accuracy_percentage)
    """
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, labels in loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            total_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    return total_loss / len(loader), 100 * correct / total


def train_model(
    model, train_loader, test_loader, device, epochs=100, lr=0.001, model_name="model"
):
    """Train a model and save the best version based on test accuracy.

    Args:
        model: Neural network model to train
        train_loader: DataLoader for training data
        test_loader: DataLoader for test data
        device: Device to run training on (CPU or CUDA)
        epochs: Number of training epochs (default: 100)
        lr: Learning rate (default: 0.001)
        model_name: Name for saving model checkpoint (default: "model")

    Returns:
        Best test accuracy achieved during training
    """
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    best_acc = 0.0

    print(f"\nTraining {model_name}")

    for epoch in range(epochs):
        train_loss, train_acc = train_epoch(
            model, train_loader, criterion, optimizer, device
        )
        test_loss, test_acc = evaluate(model, test_loader, criterion, device)

        if test_acc > best_acc:
            best_acc = test_acc
            torch.save(model.state_dict(), f"best_{model_name}.pth")

        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(
                f"Epoch {epoch + 1}/{epochs} - Train: {train_acc:.2f}% - Test: {test_acc:.2f}%"
            )

    model.load_state_dict(torch.load(f"best_{model_name}.pth"))
    final_loss, final_acc = evaluate(model, test_loader, criterion, device)
    print(f"Best accuracy: {best_acc:.2f}%\n")

    return best_acc


def create_confusion_matrix(
    model, loader, device, class_names, save_path="confusion_matrix.png"
):
    """Generate and save confusion matrix for the model.

    Args:
        model: Neural network model to evaluate
        loader: DataLoader for evaluation data
        device: Device to run evaluation on (CPU or CUDA)
        class_names: List of class names for labeling the matrix
        save_path: Path to save the confusion matrix image (default: "confusion_matrix.png")
    """
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for inputs, labels in loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    cm = confusion_matrix(all_labels, all_preds)

    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="viridis",
        xticklabels=class_names,
        yticklabels=class_names,
    )
    plt.title("Confusion Matrix")
    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    plt.tight_layout()
    plt.savefig(save_path)
    print(f"Confusion matrix saved to {save_path}")
    plt.close()


def load_sonar_data(batch_size=16):
    """Load and prepare Sonar dataset for training.

    Args:
        batch_size: Batch size for DataLoaders (default: 16)

    Returns:
        Tuple of (train_loader, test_loader)
    """
    dataset = SonarDataset("sonar_dataset.csv")
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader


def load_cifar10_animals(batch_size=128):
    """Load CIFAR-10 dataset filtered to only animal classes.

    Args:
        batch_size: Batch size for DataLoaders (default: 128)

    Returns:
        Tuple of (train_loader, test_loader) with 6 animal classes
    """
    animal_classes = [2, 3, 4, 5, 6, 7]
    transform = transforms.Compose([transforms.ToTensor()])

    train_full = datasets.CIFAR10(
        root="./data", train=True, download=True, transform=transform
    )
    test_full = datasets.CIFAR10(
        root="./data", train=False, download=True, transform=transform
    )

    train_indices = [
        i for i, (_, label) in enumerate(train_full) if label in animal_classes
    ]
    test_indices = [
        i for i, (_, label) in enumerate(test_full) if label in animal_classes
    ]

    class_mapping = {old: new for new, old in enumerate(animal_classes)}

    train_dataset = RemappedDataset(Subset(train_full, train_indices), class_mapping)
    test_dataset = RemappedDataset(Subset(test_full, test_indices), class_mapping)

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=2
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, num_workers=2
    )

    return train_loader, test_loader


def load_fashion_mnist(batch_size=128):
    """Load Fashion MNIST dataset for training.

    Args:
        batch_size: Batch size for DataLoaders (default: 128)

    Returns:
        Tuple of (train_loader, test_loader)
    """
    transform = transforms.Compose([transforms.ToTensor()])

    train_dataset = datasets.FashionMNIST(
        root="./data", train=True, download=True, transform=transform
    )
    test_dataset = datasets.FashionMNIST(
        root="./data", train=False, download=True, transform=transform
    )

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=2
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, num_workers=2
    )

    return train_loader, test_loader


def load_fgvc_aircraft(batch_size=64):
    """Load FGVC Aircraft dataset for training.

    Args:
        batch_size: Batch size for DataLoaders (default: 64)

    Returns:
        Tuple of (train_loader, test_loader)
    """
    transform = transforms.Compose(
        [
            transforms.Resize((64, 64)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    train_dataset = datasets.FGVCAircraft(
        root="./data", split="train", download=True, transform=transform
    )
    test_dataset = datasets.FGVCAircraft(
        root="./data", split="test", download=True, transform=transform
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=8,
        pin_memory=True,
        prefetch_factor=2,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        prefetch_factor=2,
    )

    return train_loader, test_loader


def load_eurosat(batch_size=64):
    """Load EuroSAT satellite imagery dataset for land use classification.

    Args:
        batch_size: Batch size for DataLoaders (default: 64)

    Returns:
        Tuple of (train_loader, test_loader) with 80-20 train-test split
    """
    transform = transforms.Compose(
        [
            transforms.Resize((64, 64)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    full_dataset = datasets.EuroSAT(root="./data", download=True, transform=transform)

    # Split dataset into train and test (80-20 split)
    train_size = int(0.8 * len(full_dataset))
    test_size = len(full_dataset) - train_size
    train_dataset, test_dataset = random_split(
        full_dataset,
        [train_size, test_size],
        generator=torch.Generator().manual_seed(2137),
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )

    return train_loader, test_loader


def main():
    parser = argparse.ArgumentParser(
        description="Train neural network models for various classification tasks"
    )
    parser.add_argument(
        "model",
        choices=["sonar", "cifar10", "fashion", "aircraft", "eurosat"],
        help="Which model to train: sonar, cifar10, fashion, aircraft, or eurosat",
    )
    args = parser.parse_args()

    torch.manual_seed(2137)
    np.random.seed(2137)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    if args.model == "sonar":
        print("\n[Sonar] Classification (Rock vs Mine)")
        train_loader, test_loader = load_sonar_data(batch_size=16)

        # Train larger model
        model_large = SonarNN().to(device)
        acc_large = train_model(
            model_large,
            train_loader,
            test_loader,
            device,
            epochs=100,
            lr=0.001,
            model_name="sonar",
        )

        # Train smaller model
        model_small = SonarNNSmall().to(device)
        acc_small = train_model(
            model_small,
            train_loader,
            test_loader,
            device,
            epochs=100,
            lr=0.001,
            model_name="sonar_small",
        )

        # Compare results
        print(f"Large Model (128-64):  {acc_large:.2f}%")
        print(f"Small Model (32-16):   {acc_small:.2f}%")
        print(f"Difference:            {acc_large - acc_small:+.2f}%")

    if args.model == "cifar10":
        print("\n[CIFAR-10] Animal Classification")
        train_loader, test_loader = load_cifar10_animals(batch_size=256)
        model = CIFAR10AnimalCNN(num_classes=6).to(device)
        train_model(
            model,
            train_loader,
            test_loader,
            device,
            epochs=50,
            lr=0.001,
            model_name="cifar10_animals",
        )

        # Generate confusion matrix
        class_names = ["bird", "cat", "deer", "dog", "frog", "horse"]
        create_confusion_matrix(
            model, test_loader, device, class_names, "cifar10_confusion_matrix.png"
        )

    if args.model == "fashion":
        print("\n[Fashion MNIST] Classification")
        train_loader, test_loader = load_fashion_mnist(batch_size=256)
        model = FashionMNISTCNN(num_classes=10).to(device)
        train_model(
            model,
            train_loader,
            test_loader,
            device,
            epochs=30,
            lr=0.001,
            model_name="fashion_mnist",
        )

    if args.model == "eurosat":
        print("\n[EuroSAT] Satellite Image Land Use Classification")
        train_loader, test_loader = load_eurosat(batch_size=512)
        model = EuroSATCNN(num_classes=10).to(device)
        train_model(
            model,
            train_loader,
            test_loader,
            device,
            epochs=30,
            lr=0.001,
            model_name="eurosat",
        )


if __name__ == "__main__":
    main()
