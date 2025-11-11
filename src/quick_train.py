"""
Quick training script for lightweight baseline models
Trains smaller models more quickly for demonstration
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

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")


def get_simple_cnn(num_classes=10):
    """Simple CNN for faster training"""
    class SimpleCNN(nn.Module):
        def __init__(self, num_classes=10):
            super(SimpleCNN, self).__init__()
            self.features = nn.Sequential(
                nn.Conv2d(3, 32, 3, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(2),
                nn.Conv2d(32, 64, 3, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(2),
                nn.Conv2d(64, 128, 3, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(2),
            )
            self.classifier = nn.Sequential(
                nn.Dropout(0.5),
                nn.Linear(128 * 4 * 4, 256),
                nn.ReLU(),
                nn.Dropout(0.5),
                nn.Linear(256, num_classes)
            )
        
        def forward(self, x):
            x = self.features(x)
            x = x.view(x.size(0), -1)
            x = self.classifier(x)
            return x
    
    return SimpleCNN(num_classes)


def train_quick_model(model_name, model, train_loader, val_loader, epochs=3):
    """Quick training for demonstration"""
    print(f"\nTraining {model_name}...")
    model = model.to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    best_acc = 0.0
    
    for epoch in range(epochs):
        # Train
        model.train()
        for inputs, labels in tqdm(train_loader, desc=f'Epoch {epoch+1}/{epochs}'):
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
        
        # Validate
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
        
        acc = 100. * correct / total
        print(f"Epoch {epoch+1}: Val Acc = {acc:.2f}%")
        
        if acc > best_acc:
            best_acc = acc
            os.makedirs('models', exist_ok=True)
            torch.save(model.state_dict(), f'models/{model_name}.pth')
    
    return best_acc


def main():
    """Train multiple simple models quickly"""
    print("Loading data...")
    train_loader, val_loader, test_loader, _ = load_cifar10(batch_size=128)
    
    # Create diverse models
    models_dict = {
        'cnn_v1': get_simple_cnn(10),
        'cnn_v2': get_simple_cnn(10),  # Different random init
        'cnn_v3': get_simple_cnn(10),
    }
    
    # Also add a pre-trained ResNet (just final layer training)
    resnet = models.resnet18(weights='IMAGENET1K_V1')
    resnet.fc = nn.Linear(resnet.fc.in_features, 10)
    # Freeze all except final layer
    for param in resnet.parameters():
        param.requires_grad = False
    resnet.fc.weight.requires_grad = True
    resnet.fc.bias.requires_grad = True
    models_dict['resnet18_ft'] = resnet
    
    # MobileNet
    mobilenet = models.mobilenet_v2(weights='IMAGENET1K_V1')
    mobilenet.classifier[1] = nn.Linear(mobilenet.classifier[1].in_features, 10)
    for param in mobilenet.parameters():
        param.requires_grad = False
    mobilenet.classifier[1].weight.requires_grad = True
    mobilenet.classifier[1].bias.requires_grad = True
    models_dict['mobilenet_ft'] = mobilenet
    
    results = {}
    
    # Train each model
    for name, model in models_dict.items():
        acc = train_quick_model(name, model, train_loader, val_loader, epochs=3)
        results[name] = acc
        print(f"{name}: {acc:.2f}%\n")
    
    # Save results
    os.makedirs('results/tables', exist_ok=True)
    with open('results/tables/quick_train_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print("\nTraining complete!")
    print("Results:", results)


if __name__ == '__main__':
    main()
