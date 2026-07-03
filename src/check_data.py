import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np

# define transform: grayscale + normalize
transform = transforms.Compose([
    transforms.Grayscale(),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# download and load dataset
trainset = torchvision.datasets.CIFAR10(
    root='./dataset', train=True, download=True, transform=transform
)
testset = torchvision.datasets.CIFAR10(
    root='./dataset', train=False, download=True, transform=transform
)

# check sizes
print(f"Training samples: {len(trainset)}")
print(f"Test samples:     {len(testset)}")

# check one image shape
image, label = trainset[0]
print(f"Image shape: {image.shape}")   # should be torch.Size([1, 32, 32])
print(f"Label: {label}")

# class names
classes = ['airplane','automobile','bird','cat','deer',
           'dog','frog','horse','ship','truck']

# visualize a few samples
fig, axes = plt.subplots(1, 5, figsize=(12, 3))
for i in range(5):
    image, label = trainset[i]
    axes[i].imshow(image.squeeze(), cmap='gray')
    axes[i].set_title(classes[label])
    axes[i].axis('off')
plt.tight_layout()
plt.savefig('../outputs/sample_images.png')
plt.show()
print("Sample images saved to outputs/sample_images.png")