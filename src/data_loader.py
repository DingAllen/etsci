"""
Data loading and preprocessing for CIFAR-10
"""
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, random_split, Dataset
import numpy as np
import matplotlib.pyplot as plt
import os
from PIL import Image

# Set random seeds for reproducibility
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)

# CIFAR-10 classes
CIFAR10_CLASSES = ['airplane', 'automobile', 'bird', 'cat', 'deer',
                   'dog', 'frog', 'horse', 'ship', 'truck']


class CIFAR10BinaryDataset(Dataset):
    """Custom dataset to load CIFAR-10 from binary files"""
    
    def __init__(self, data_files, transform=None):
        """
        Args:
            data_files: List of paths to .bin files
            transform: Optional transform to apply
        """
        self.transform = transform
        self.data = []
        self.labels = []
        
        # Load data from binary files
        for filepath in data_files:
            with open(filepath, 'rb') as f:
                # Each record is 3073 bytes: 1 byte label + 3072 bytes image (32x32x3)
                while True:
                    label_byte = f.read(1)
                    if not label_byte:
                        break
                    label = int.from_bytes(label_byte, 'little')
                    
                    # Read image data (3072 bytes = 32x32x3)
                    img_bytes = f.read(3072)
                    if len(img_bytes) != 3072:
                        break
                    
                    # Convert to numpy array and reshape
                    img_array = np.frombuffer(img_bytes, dtype=np.uint8)
                    img_array = img_array.reshape(3, 32, 32)
                    img_array = img_array.transpose(1, 2, 0)  # CHW to HWC
                    
                    self.data.append(img_array)
                    self.labels.append(label)
        
        self.data = np.array(self.data)
        self.labels = np.array(self.labels)
        
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        img = self.data[idx]
        label = self.labels[idx]
        
        # Convert to PIL Image for transforms
        img = Image.fromarray(img)
        
        if self.transform:
            img = self.transform(img)
        
        return img, label


def get_transforms(train=True):
    """
    Get data transformations for CIFAR-10
    
    Args:
        train: If True, return training transforms with augmentation
    
    Returns:
        torchvision.transforms.Compose object
    """
    if train:
        transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), 
                               (0.2023, 0.1994, 0.2010))
        ])
    else:
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), 
                               (0.2023, 0.1994, 0.2010))
        ])
    return transform


def load_cifar10(data_dir='./data/cifar10-storage/cifar-10-batches-bin', batch_size=128, num_workers=0):
    """
    Load CIFAR-10 dataset with train/val/test splits
    
    Args:
        data_dir: Directory containing .bin files
        batch_size: Batch size for data loaders
        num_workers: Number of worker processes for data loading
    
    Returns:
        train_loader, val_loader, test_loader, class_names
    """
    # Prepare file paths
    train_files = [os.path.join(data_dir, f'data_batch_{i}.bin') for i in range(1, 6)]
    test_file = [os.path.join(data_dir, 'test_batch.bin')]
    
    # Get transforms
    train_transform = get_transforms(train=True)
    test_transform = get_transforms(train=False)
    
    # Load full training set
    full_train_dataset = CIFAR10BinaryDataset(train_files, transform=train_transform)
    
    # Split into train and validation (45000/5000)
    train_size = 45000
    val_size = 5000
    train_dataset, val_dataset = random_split(
        full_train_dataset, 
        [train_size, val_size],
        generator=torch.Generator().manual_seed(SEED)
    )
    
    # Load test set
    test_dataset = CIFAR10BinaryDataset(test_file, transform=test_transform)
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, 
        num_workers=num_workers
    )
    
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers
    )
    
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers
    )
    
    print(f"Dataset loaded:")
    print(f"  Training samples: {len(train_dataset)}")
    print(f"  Validation samples: {len(val_dataset)}")
    print(f"  Test samples: {len(test_dataset)}")
    print(f"  Classes: {len(CIFAR10_CLASSES)}")
    
    return train_loader, val_loader, test_loader, CIFAR10_CLASSES


def visualize_samples(data_loader, num_samples=20, save_path='results/figures/data_samples.png'):
    """
    Visualize sample images from CIFAR-10
    
    Args:
        data_loader: DataLoader to sample from
        num_samples: Number of samples to visualize
        save_path: Path to save the figure
    """
    # Get a batch of images
    dataiter = iter(data_loader)
    images, labels = next(dataiter)
    
    # Denormalize images for visualization
    mean = np.array([0.4914, 0.4822, 0.4465])
    std = np.array([0.2023, 0.1994, 0.2010])
    
    # Select first num_samples
    images = images[:num_samples]
    labels = labels[:num_samples]
    
    # Create figure
    fig, axes = plt.subplots(4, 5, figsize=(12, 10))
    fig.suptitle('CIFAR-10 Sample Images', fontsize=16)
    
    for idx, ax in enumerate(axes.flat):
        if idx < len(images):
            # Denormalize and convert to numpy
            img = images[idx].numpy().transpose((1, 2, 0))
            img = std * img + mean
            img = np.clip(img, 0, 1)
            
            ax.imshow(img)
            ax.set_title(f'{CIFAR10_CLASSES[labels[idx]]}', fontsize=10)
            ax.axis('off')
        else:
            ax.axis('off')
    
    plt.tight_layout()
    
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved sample visualization to {save_path}")
    plt.close()


def get_class_distribution(data_loader):
    """
    Get class distribution in the dataset
    
    Args:
        data_loader: DataLoader to analyze
    
    Returns:
        Dictionary with class counts
    """
    class_counts = {i: 0 for i in range(10)}
    
    for _, labels in data_loader:
        for label in labels:
            class_counts[label.item()] += 1
    
    return class_counts


if __name__ == '__main__':
    # Test the data loader
    print("Testing CIFAR-10 data loader...")
    train_loader, val_loader, test_loader, classes = load_cifar10()
    
    # Visualize samples
    print("\nGenerating sample visualization...")
    visualize_samples(train_loader)
    
    # Show class distribution
    print("\nClass distribution in training set:")
    train_dist = get_class_distribution(train_loader)
    for idx, count in train_dist.items():
        print(f"  {classes[idx]}: {count}")
    
    print("\nData loader test complete!")
