"""
Train baseline CNN models on CIFAR-10
Uses pre-trained models and fine-tunes them
"""
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
from tqdm import tqdm
import os
import json
import numpy as np

from data_loader import load_cifar10, SEED

# Set random seeds
torch.manual_seed(SEED)
np.random.seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed(SEED)

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")


def get_model(model_name, num_classes=10, pretrained=True):
    """
    Get a pre-trained model and modify for CIFAR-10
    
    Args:
        model_name: Name of the model ('resnet18', 'resnet34', 'vgg16', 'mobilenet', 'densenet')
        num_classes: Number of output classes
        pretrained: Whether to load pretrained weights
    
    Returns:
        model: PyTorch model
    """
    if model_name == 'resnet18':
        model = models.resnet18(weights='IMAGENET1K_V1' if pretrained else None)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    
    elif model_name == 'resnet34':
        model = models.resnet34(weights='IMAGENET1K_V1' if pretrained else None)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    
    elif model_name == 'vgg16':
        model = models.vgg16(weights='IMAGENET1K_V1' if pretrained else None)
        model.classifier[6] = nn.Linear(4096, num_classes)
    
    elif model_name == 'mobilenet':
        model = models.mobilenet_v2(weights='IMAGENET1K_V1' if pretrained else None)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
    
    elif model_name == 'densenet':
        model = models.densenet121(weights='IMAGENET1K_V1' if pretrained else None)
        model.classifier = nn.Linear(model.classifier.in_features, num_classes)
    
    else:
        raise ValueError(f"Unknown model: {model_name}")
    
    return model


def train_epoch(model, train_loader, criterion, optimizer, device):
    """Train for one epoch"""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    pbar = tqdm(train_loader, desc='Training')
    for inputs, labels in pbar:
        inputs, labels = inputs.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
        
        pbar.set_postfix({'loss': running_loss/len(train_loader), 'acc': 100.*correct/total})
    
    return running_loss / len(train_loader), 100. * correct / total


def validate(model, val_loader, criterion, device):
    """Validate the model"""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for inputs, labels in tqdm(val_loader, desc='Validation'):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
    
    return running_loss / len(val_loader), 100. * correct / total


def train_model(model_name, train_loader, val_loader, epochs=10, lr=0.001, save_dir='models'):
    """
    Train a model on CIFAR-10
    
    Args:
        model_name: Name of the model
        train_loader: Training data loader
        val_loader: Validation data loader
        epochs: Number of epochs
        lr: Learning rate
        save_dir: Directory to save models
    
    Returns:
        best_acc: Best validation accuracy
        history: Training history
    """
    print(f"\n{'='*60}")
    print(f"Training {model_name}")
    print(f"{'='*60}")
    
    # Get model
    model = get_model(model_name, num_classes=10, pretrained=True)
    model = model.to(device)
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=2, verbose=True)
    
    # Training history
    history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': []
    }
    
    best_acc = 0.0
    
    for epoch in range(epochs):
        print(f"\nEpoch {epoch+1}/{epochs}")
        
        # Train
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        
        # Validate
        val_loss, val_acc = validate(model, val_loader, criterion, device)
        
        # Update scheduler
        scheduler.step(val_acc)
        
        # Save history
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        
        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
        print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
        
        # Save best model
        if val_acc > best_acc:
            best_acc = val_acc
            os.makedirs(save_dir, exist_ok=True)
            save_path = os.path.join(save_dir, f'{model_name}.pth')
            torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'epoch': epoch,
                'best_acc': best_acc,
            }, save_path)
            print(f"Saved best model to {save_path}")
    
    print(f"\nBest validation accuracy for {model_name}: {best_acc:.2f}%")
    
    return best_acc, history


def main():
    """Main training function"""
    # Load data
    print("Loading CIFAR-10 dataset...")
    train_loader, val_loader, test_loader, classes = load_cifar10(batch_size=64)
    
    # Models to train
    models_to_train = ['resnet18', 'resnet34', 'vgg16', 'mobilenet', 'densenet']
    
    # Training configuration
    epochs = 5  # Reduced for faster training
    lr = 0.001
    
    # Track results
    results = {}
    all_histories = {}
    
    # Train each model
    for model_name in models_to_train:
        best_acc, history = train_model(
            model_name, 
            train_loader, 
            val_loader, 
            epochs=epochs, 
            lr=lr
        )
        results[model_name] = best_acc
        all_histories[model_name] = history
    
    # Save results
    os.makedirs('results/tables', exist_ok=True)
    with open('results/tables/baseline_results.json', 'w') as f:
        json.dump({
            'results': results,
            'histories': all_histories
        }, f, indent=2)
    
    # Print summary
    print(f"\n{'='*60}")
    print("Training Summary")
    print(f"{'='*60}")
    for model_name, acc in results.items():
        print(f"{model_name:15s}: {acc:.2f}%")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()
