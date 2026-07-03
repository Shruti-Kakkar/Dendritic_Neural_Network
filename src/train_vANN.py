import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
from models.vANN import VanillaANN

# ── reproducibility ──────────────────────────────────────────
def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)

# ── data loading ─────────────────────────────────────────────
def get_dataloaders(batch_size=128):
    transform = transforms.Compose([
        transforms.Grayscale(),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    trainset_full = torchvision.datasets.CIFAR10(
        root='./dataset', train=True, download=False, transform=transform
    )
    testset = torchvision.datasets.CIFAR10(
        root='./dataset', train=False, download=False, transform=transform
    )

    # split 90% train / 10% validation
    n_train = int(0.9 * len(trainset_full))
    n_val   = len(trainset_full) - n_train
    trainset, valset = torch.utils.data.random_split(
        trainset_full, [n_train, n_val]
    )

    train_loader = torch.utils.data.DataLoader(
        trainset, batch_size=batch_size, shuffle=True
    )
    val_loader = torch.utils.data.DataLoader(
        valset, batch_size=batch_size, shuffle=False
    )
    test_loader = torch.utils.data.DataLoader(
        testset, batch_size=batch_size, shuffle=False
    )

    return train_loader, val_loader, test_loader

# ── evaluation ────────────────────────────────────────────────
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            total_loss += loss.item() * labels.size(0)
            _, predicted = outputs.max(1)
            correct += predicted.eq(labels).sum().item()
            total += labels.size(0)
    return total_loss / total, 100.0 * correct / total

# ── training loop ─────────────────────────────────────────────
def train(seed=42, epochs=20, batch_size=128, lr=0.001):
    set_seed(seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    print(f"Running seed: {seed}")

    train_loader, val_loader, test_loader = get_dataloaders(batch_size)

    model = VanillaANN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    print(f"Trainable parameters: {model.count_parameters():,}")

    # tracking
    history = {
        'train_loss': [], 'val_loss': [],
        'train_acc':  [], 'val_acc':  []
    }

    for epoch in range(1, epochs + 1):
        # ── train ──
        model.train()
        train_loss, correct, total = 0.0, 0, 0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * labels.size(0)
            _, predicted = outputs.max(1)
            correct += predicted.eq(labels).sum().item()
            total += labels.size(0)

        train_loss /= total
        train_acc   = 100.0 * correct / total

        # ── validate ──
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)

        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_acc'].append(train_acc)
        history['val_acc'].append(val_acc)

        print(f"Epoch {epoch:02d}/{epochs} | "
              f"train_loss: {train_loss:.4f}  train_acc: {train_acc:.2f}%  | "
              f"val_loss: {val_loss:.4f}  val_acc: {val_acc:.2f}%")

    # ── final test evaluation ──
    test_loss, test_acc = evaluate(model, test_loader, criterion, device)
    print(f"\nFinal Test — loss: {test_loss:.4f}  acc: {test_acc:.2f}%")

    return history, test_loss, test_acc

# ── plotting ──────────────────────────────────────────────────
def plot_history(history, seed, save_dir='./outputs'):
    epochs = range(1, len(history['train_loss']) + 1)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # loss plot
    axes[0].plot(epochs, history['train_loss'], label='Train Loss')
    axes[0].plot(epochs, history['val_loss'],   label='Val Loss')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title(f'vANN Loss Curves (seed={seed})')
    axes[0].legend()

    # accuracy plot
    axes[1].plot(epochs, history['train_acc'], label='Train Acc')
    axes[1].plot(epochs, history['val_acc'],   label='Val Acc')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy (%)')
    axes[1].set_title(f'vANN Accuracy Curves (seed={seed})')
    axes[1].legend()

    plt.tight_layout()
    path = os.path.join(save_dir, f'vann_curves_seed{seed}.png')
    plt.savefig(path)
    plt.close()
    print(f"Plot saved to {path}")

# ── main: run 3 seeds ─────────────────────────────────────────
if __name__ == "__main__":
    seeds = [42, 123, 7]
    all_test_acc  = []
    all_test_loss = []

    for seed in seeds:
        print(f"\n{'='*55}")
        history, test_loss, test_acc = train(seed=seed)
        plot_history(history, seed)
        all_test_acc.append(test_acc)
        all_test_loss.append(test_loss)

    print(f"\n{'='*55}")
    print(f"Results across {len(seeds)} seeds:")
    print(f"Test Accuracy: {np.mean(all_test_acc):.2f}% "
          f"± {np.std(all_test_acc):.2f}%")
    print(f"Test Loss:     {np.mean(all_test_loss):.4f} "
          f"± {np.std(all_test_loss):.4f}")