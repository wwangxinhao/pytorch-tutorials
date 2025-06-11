#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Convolutional Neural Networks in PyTorch

This script provides implementations and examples of Convolutional Neural Networks
using PyTorch, from basic concepts to advanced techniques.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
import matplotlib.pyplot as plt
import numpy as np
import time
import copy
from PIL import Image
from torchvision.utils import make_grid
import os

# Set random seed for reproducibility
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed(42)

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# -----------------------------------------------------------------------------
# Section 1: CNN Architecture Components
# -----------------------------------------------------------------------------

def visualize_filters(model, layer_name="conv1"):
    """Visualize the filters of a specific layer in a CNN model."""
    # Get the convolutional layer
    for name, module in model.named_modules():
        if name == layer_name:
            weights = module.weight.data.cpu()
            
            # Normalize the weights for better visualization
            weights = (weights - weights.min()) / (weights.max() - weights.min())
            
            # Plot the filters
            plt.figure(figsize=(12, 6))
            num_filters = weights.shape[0]
            num_cols = 8
            num_rows = (num_filters + num_cols - 1) // num_cols
            
            for i in range(num_filters):
                plt.subplot(num_rows, num_cols, i + 1)
                
                # If there are 3 input channels (RGB)
                if weights.shape[1] == 3:
                    plt.imshow(weights[i].permute(1, 2, 0))
                else:
                    # If there's only 1 input channel
                    plt.imshow(weights[i, 0], cmap='gray')
                    
                plt.axis('off')
            
            plt.tight_layout()
            plt.show()
            return
    
    print(f"Layer '{layer_name}' not found in the model.")

class CNNComponents(nn.Module):
    """A simple CNN demonstrating various architectural components."""
    def __init__(self, in_channels=1, num_classes=10):
        super(CNNComponents, self).__init__()
        
        # Convolutional Layer 1
        self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(32)  # Batch Normalization
        
        # Convolutional Layer 2
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(64)  # Batch Normalization
        
        # Max Pooling
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Average Pooling
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        
        # Dropout for regularization
        self.dropout = nn.Dropout(0.25)
        
        # Fully Connected Layers
        self.fc1 = nn.Linear(64, 128)
        self.fc2 = nn.Linear(128, num_classes)
    
    def forward(self, x):
        # First convolutional block
        x = self.conv1(x)              # Convolution
        x = self.bn1(x)                # Batch Normalization
        x = F.relu(x)                  # ReLU Activation
        x = self.pool(x)               # Max Pooling
        
        # Second convolutional block
        x = self.conv2(x)              # Convolution
        x = self.bn2(x)                # Batch Normalization
        x = F.relu(x)                  # ReLU Activation
        x = self.pool(x)               # Max Pooling
        
        # Global average pooling
        x = self.avgpool(x)            # Average Pooling
        x = torch.flatten(x, 1)        # Flatten
        
        # Fully connected layers with dropout
        x = self.dropout(x)            # Dropout
        x = self.fc1(x)                # Fully Connected
        x = F.relu(x)                  # ReLU Activation
        x = self.dropout(x)            # Dropout
        x = self.fc2(x)                # Fully Connected
        
        return x

def explain_cnn_components():
    """Explain the components of CNNs with visual examples."""
    print("CNN Architecture Components:")
    print("-" * 50)
    
    # Create a sample input
    sample_input = torch.randn(1, 1, 28, 28)  # Batch size, channels, height, width
    
    # Demonstrate convolutional layer
    conv_layer = nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1)
    conv_output = conv_layer(sample_input)
    print(f"Conv2d: Input shape {sample_input.shape} -> Output shape {conv_output.shape}")
    print(f"Parameters: {sum(p.numel() for p in conv_layer.parameters())}")
    
    # Demonstrate batch normalization
    bn_layer = nn.BatchNorm2d(16)
    bn_output = bn_layer(conv_output)
    print(f"BatchNorm2d: Input shape {conv_output.shape} -> Output shape {bn_output.shape}")
    
    # Demonstrate activation function
    relu_output = F.relu(bn_output)
    print(f"ReLU: Input shape {bn_output.shape} -> Output shape {relu_output.shape}")
    
    # Demonstrate max pooling
    pool_layer = nn.MaxPool2d(kernel_size=2, stride=2)
    pool_output = pool_layer(relu_output)
    print(f"MaxPool2d: Input shape {relu_output.shape} -> Output shape {pool_output.shape}")
    
    # Demonstrate average pooling
    avgpool_layer = nn.AvgPool2d(kernel_size=2, stride=2)
    avgpool_output = avgpool_layer(relu_output)
    print(f"AvgPool2d: Input shape {relu_output.shape} -> Output shape {avgpool_output.shape}")
    
    # Demonstrate adaptive average pooling
    adaptive_avgpool = nn.AdaptiveAvgPool2d((1, 1))
    adaptive_output = adaptive_avgpool(pool_output)
    print(f"AdaptiveAvgPool2d: Input shape {pool_output.shape} -> Output shape {adaptive_output.shape}")
    
    # Demonstrate flatten operation
    flatten_output = torch.flatten(adaptive_output, 1)
    print(f"Flatten: Input shape {adaptive_output.shape} -> Output shape {flatten_output.shape}")
    
    # Demonstrate fully connected layer
    fc_layer = nn.Linear(16, 10)
    fc_output = fc_layer(flatten_output)
    print(f"Linear: Input shape {flatten_output.shape} -> Output shape {fc_output.shape}")
    
    # Demonstrate dropout
    dropout_layer = nn.Dropout(0.5)
    dropout_output = dropout_layer(fc_output)
    print(f"Dropout: Input shape {fc_output.shape} -> Output shape {dropout_output.shape}")

# -----------------------------------------------------------------------------
# Section 2: Image Classification with CNNs
# -----------------------------------------------------------------------------

class LeNet5(nn.Module):
    """Implementation of LeNet-5 architecture for MNIST."""
    def __init__(self, num_classes=10):
        super(LeNet5, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, kernel_size=5)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(16*4*4, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, num_classes)
        
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool1(x)
        x = F.relu(self.conv2(x))
        x = self.pool2(x)
        x = x.view(-1, 16*4*4)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

class AlexNetMNIST(nn.Module):
    """Simplified AlexNet architecture for MNIST."""
    def __init__(self, num_classes=10):
        super(AlexNetMNIST, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(64, 192, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
        )
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

def train_model(model, dataloaders, criterion, optimizer, num_epochs=10):
    """Generic function to train a model."""
    since = time.time()
    
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    
    for epoch in range(num_epochs):
        print(f'Epoch {epoch+1}/{num_epochs}')
        print('-' * 10)
        
        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode
                
            running_loss = 0.0
            running_corrects = 0
            
            # Iterate over data
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)
                
                # Zero the parameter gradients
                optimizer.zero_grad()
                
                # Forward pass
                # Enable gradients only in training phase
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)
                    
                    # Backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
                        
                # Statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
                
            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_acc = running_corrects.double() / len(dataloaders[phase].dataset)
            
            print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')
            
            # Deep copy the model if we have a new best validation accuracy
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
        
        print()
    
    time_elapsed = time.time() - since
    print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
    print(f'Best val Acc: {best_acc:.4f}')
    
    # Load best model weights
    model.load_state_dict(best_model_wts)
    return model

def mnist_classification():
    """Train and evaluate CNNs on MNIST dataset."""
    # Define transformations
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    # Load MNIST dataset
    train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
    
    dataloaders = {
        'train': train_loader,
        'val': test_loader
    }
    
    # Initialize LeNet-5 model
    model = LeNet5().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # Train the model
    print("Training LeNet-5 model on MNIST...")
    model = train_model(model, dataloaders, criterion, optimizer, num_epochs=5)
    
    # Test the model
    model.eval()
    with torch.no_grad():
        correct = 0
        total = 0
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
        print(f'Test Accuracy: {100 * correct / total:.2f}%')
    
    return model

# -----------------------------------------------------------------------------
# Section 3: Transfer Learning with Pre-trained Models
# -----------------------------------------------------------------------------

def prepare_cifar10_data():
    """Prepare CIFAR-10 dataset for transfer learning."""
    # Define transformations for training data with augmentation
    train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32, padding=4),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])
    
    # Define transformations for validation data
    val_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])
    
    # Load CIFAR-10 dataset
    train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=train_transform)
    val_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=val_transform)
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=128, shuffle=False, num_workers=2)
    
    dataloaders = {
        'train': train_loader,
        'val': val_loader
    }
    
    class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 
                   'dog', 'frog', 'horse', 'ship', 'truck']
    
    return dataloaders, class_names

def transfer_learning_example():
    """Demonstrate transfer learning using a pre-trained ResNet model on CIFAR-10."""
    print("Transfer Learning with Pre-trained Models")
    print("-" * 50)
    
    # Prepare CIFAR-10 data
    dataloaders, class_names = prepare_cifar10_data()
    
    # Load a pre-trained ResNet model
    model = models.resnet18(pretrained=True)
    
    # Print the original model architecture
    print("Original ResNet-18 Architecture:")
    print(model)
    
    # Modify the final fully connected layer for CIFAR-10 (10 classes)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 10)
    
    # Move model to the appropriate device
    model = model.to(device)
    
    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    
    # Feature extraction: only update the reshaped layer params
    print("\nApproach 1: Feature Extraction - Only train the final layer")
    for param in model.parameters():
        param.requires_grad = False
    for param in model.fc.parameters():
        param.requires_grad = True
    
    # Optimize only the final layer
    optimizer = optim.SGD(model.fc.parameters(), lr=0.001, momentum=0.9)
    
    # Train the model (feature extraction)
    model = train_model(model, dataloaders, criterion, optimizer, num_epochs=5)
    
    # Fine-tuning: update all layers
    print("\nApproach 2: Fine-tuning - Train all layers")
    for param in model.parameters():
        param.requires_grad = True
    
    # Optimize all parameters with a lower learning rate
    optimizer = optim.SGD(model.parameters(), lr=0.0001, momentum=0.9)
    
    # Train the model (fine-tuning)
    model = train_model(model, dataloaders, criterion, optimizer, num_epochs=5)
    
    return model

# -----------------------------------------------------------------------------
# Section 4: Feature Visualization
# -----------------------------------------------------------------------------

def visualize_layer_outputs(model, layer_name, image_tensor):
    """Visualize the outputs of a specific layer given an input image."""
    # Create a hook function
    activation = {}
    def get_activation(name):
        def hook(model, input, output):
            activation[name] = output.detach()
        return hook
    
    # Register the hook
    for name, module in model.named_modules():
        if name == layer_name:
            handle = module.register_forward_hook(get_activation(layer_name))
            break
    
    # Forward pass
    model.eval()
    with torch.no_grad():
        output = model(image_tensor.unsqueeze(0).to(device))
    
    # Remove the hook
    handle.remove()
    
    # Get the activation
    act = activation[layer_name].squeeze().cpu()
    
    # Plot the feature maps
    if len(act.shape) == 3:  # Conv layer outputs: C x H x W
        fig, axs = plt.subplots(4, 4, figsize=(12, 12))
        fig.suptitle(f'Feature Maps for Layer: {layer_name}', fontsize=16)
        axs = axs.flatten()
        
        for i in range(min(16, act.shape[0])):
            axs[i].imshow(act[i], cmap='viridis')
            axs[i].axis('off')
            
        plt.tight_layout()
        plt.subplots_adjust(top=0.9)
        plt.show()
    
    return act

def generate_class_activation_map(model, img_tensor, class_idx=None):
    """Generate a class activation map (CAM) for a given image."""
    # Ensure model is in evaluation mode
    model.eval()
    
    # Move image to device and add batch dimension
    img_tensor = img_tensor.unsqueeze(0).to(device)
    
    # Forward pass
    with torch.no_grad():
        # Get the features and predictions
        features = model.features(img_tensor)
        output = model.classifier(features.view(features.size(0), -1))
        
        # If class_idx is None, use the predicted class
        if class_idx is None:
            _, class_idx = torch.max(output, 1)
            class_idx = class_idx.item()
        
    # Get the weights of the final layer
    params = list(model.parameters())
    weight_softmax = params[-2].data  # Assume penultimate layer is the classifier weight
    
    # Generate CAM
    bz, nc, h, w = features.shape
    cam = torch.matmul(weight_softmax[class_idx], features.reshape((nc, h*w)))
    cam = cam.reshape(h, w)
    cam = cam - cam.min()
    cam = cam / cam.max()
    cam = cam.cpu().numpy()
    
    return cam, class_idx

# -----------------------------------------------------------------------------
# Section 5: Advanced CNN Architectures
# -----------------------------------------------------------------------------

class SimpleCNN(nn.Module):
    """A simple CNN architecture for CIFAR-10."""
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.conv3 = nn.Conv2d(64, 64, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 4 * 4, 512)
        self.fc2 = nn.Linear(512, 10)
        self.dropout = nn.Dropout(0.5)
        
    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = x.view(-1, 64 * 4 * 4)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

class ResidualBlock(nn.Module):
    """Simple residual block implementation."""
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, 
                               stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        # Shortcut connection to match dimensions
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, 
                          stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
            
    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out

class SimpleResNet(nn.Module):
    """A simple ResNet implementation."""
    def __init__(self, num_blocks, num_classes=10):
        super(SimpleResNet, self).__init__()
        self.in_channels = 64
        
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        
        # Create ResNet layers
        self.layer1 = self._make_layer(64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(512, num_blocks[3], stride=2)
        
        self.linear = nn.Linear(512, num_classes)
        
    def _make_layer(self, out_channels, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(ResidualBlock(self.in_channels, out_channels, stride))
            self.in_channels = out_channels
        return nn.Sequential(*layers)
    
    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out

def resnet18_custom():
    """Create a ResNet-18 model."""
    return SimpleResNet([2, 2, 2, 2])

def demonstrate_advanced_architectures():
    """Compare different CNN architectures on CIFAR-10."""
    print("Advanced CNN Architectures")
    print("-" * 50)
    
    # Prepare CIFAR-10 data
    dataloaders, class_names = prepare_cifar10_data()
    
    # Initialize models
    simple_cnn = SimpleCNN().to(device)
    resnet = resnet18_custom().to(device)
    pretrained_resnet = models.resnet18(pretrained=True)
    pretrained_resnet.fc = nn.Linear(512, 10)
    pretrained_resnet = pretrained_resnet.to(device)
    
    # Define loss function
    criterion = nn.CrossEntropyLoss()
    
    # Train and evaluate Simple CNN
    print("\nTraining Simple CNN...")
    optimizer = optim.Adam(simple_cnn.parameters(), lr=0.001)
    simple_cnn_trained = train_model(simple_cnn, dataloaders, criterion, optimizer, num_epochs=5)
    
    # Train and evaluate Custom ResNet
    print("\nTraining Custom ResNet...")
    optimizer = optim.Adam(resnet.parameters(), lr=0.001)
    resnet_trained = train_model(resnet, dataloaders, criterion, optimizer, num_epochs=5)
    
    # Train and evaluate Pre-trained ResNet
    print("\nTraining Pre-trained ResNet (fine-tuned)...")
    optimizer = optim.Adam(pretrained_resnet.parameters(), lr=0.0001)
    pretrained_resnet_trained = train_model(pretrained_resnet, dataloaders, criterion, optimizer, num_epochs=5)
    
    return {
        'simple_cnn': simple_cnn_trained,
        'resnet': resnet_trained,
        'pretrained_resnet': pretrained_resnet_trained
    }

# -----------------------------------------------------------------------------
# Main function to run all sections
# -----------------------------------------------------------------------------

def main():
    print("=" * 80)
    print("PyTorch CNN Tutorial")
    print("=" * 80)
    
    # Section 1: CNN Architecture Components
    print("\n1. CNN Architecture Components")
    explain_cnn_components()
    
    # Create a simple CNN model for components demonstration
    model = CNNComponents()
    print("\nModel Summary:")
    print(model)
    
    # Section 2: Image Classification with CNNs
    print("\n2. Image Classification with CNNs")
    mnist_model = mnist_classification()
    
    # Section 3: Transfer Learning
    print("\n3. Transfer Learning with Pre-trained Models")
    transfer_model = transfer_learning_example()
    
    # Section 4: Advanced CNN Architectures (just print info to avoid long training)
    print("\n5. Advanced CNN Architectures")
    print("This section demonstrates advanced CNN architectures like ResNet.")
    print("To run full training, call the demonstrate_advanced_architectures() function.")
    
    print("\nTutorial complete!")

if __name__ == '__main__':
    main() 