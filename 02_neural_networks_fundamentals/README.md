# Neural Networks Fundamentals with PyTorch

This tutorial covers the fundamental concepts of neural networks using PyTorch, including linear layers, activation functions, loss functions, optimizers, and the basics of building and training neural networks.

## Table of Contents
1. [Introduction to Neural Networks](#introduction-to-neural-networks)
2. [Linear Layers](#linear-layers)
3. [Activation Functions](#activation-functions)
4. [Loss Functions](#loss-functions)
5. [Optimizers](#optimizers)
6. [Building Your First Neural Network](#building-your-first-neural-network)
7. [Forward and Backward Propagation](#forward-and-backward-propagation)
8. [Practical Examples](#practical-examples)

## Introduction to Neural Networks

Neural networks are computational models inspired by the human brain's structure and function. They consist of interconnected nodes (neurons) organized in layers that process information and learn patterns from data.

### Key Components of Neural Networks:

1. **Input Layer**: Receives the initial data
2. **Hidden Layers**: Process the information through weighted connections
3. **Output Layer**: Produces the final prediction or classification
4. **Weights and Biases**: Parameters that are adjusted during training
5. **Activation Functions**: Introduce non-linearity to the model
6. **Loss Function**: Measures how well the network performs
7. **Optimizer**: Updates the weights to minimize the loss

## Linear Layers

Linear layers (also called fully connected or dense layers) are the basic building blocks of neural networks. They perform a linear transformation on the input data.

### Mathematical Representation:

```
y = xW + b
```

Where:
- `x` is the input tensor
- `W` is the weight matrix
- `b` is the bias vector
- `y` is the output tensor

### PyTorch Implementation:

```python
import torch
import torch.nn as nn

# Define a linear layer
linear = nn.Linear(in_features=10, out_features=5)

# Input tensor
x = torch.randn(3, 10)  # Batch size of 3, input dimension of 10

# Forward pass
output = linear(x)  # Shape: [3, 5]

# Access weights and biases
print(f"Weight shape: {linear.weight.shape}")  # Shape: [5, 10]
print(f"Bias shape: {linear.bias.shape}")      # Shape: [5]
```

## Activation Functions

Activation functions introduce non-linearity to neural networks, allowing them to learn complex patterns. Without activation functions, a neural network would be equivalent to a linear regression model, regardless of its depth.

### Common Activation Functions:

1. **ReLU (Rectified Linear Unit)**:
   - `f(x) = max(0, x)`
   - Most commonly used activation function
   - Helps mitigate the vanishing gradient problem

2. **Sigmoid**:
   - `f(x) = 1 / (1 + e^(-x))`
   - Outputs values between 0 and 1
   - Often used in binary classification output layers

3. **Tanh (Hyperbolic Tangent)**:
   - `f(x) = (e^x - e^(-x)) / (e^x + e^(-x))`
   - Outputs values between -1 and 1
   - Zero-centered, which can help with training

4. **Leaky ReLU**:
   - `f(x) = max(αx, x)` where α is a small constant (e.g., 0.01)
   - Addresses the "dying ReLU" problem

5. **Softmax**:
   - `f(x_i) = e^(x_i) / Σ(e^(x_j))` for all j
   - Converts a vector of values to a probability distribution
   - Often used in multi-class classification output layers

### PyTorch Implementation:

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

# Input tensor
x = torch.randn(5)
print(f"Input: {x}")

# ReLU
relu_output = F.relu(x)
print(f"ReLU: {relu_output}")

# Sigmoid
sigmoid_output = torch.sigmoid(x)
print(f"Sigmoid: {sigmoid_output}")

# Tanh
tanh_output = torch.tanh(x)
print(f"Tanh: {tanh_output}")

# Leaky ReLU
leaky_relu_output = F.leaky_relu(x, negative_slope=0.01)
print(f"Leaky ReLU: {leaky_relu_output}")

# Softmax
softmax_output = F.softmax(x, dim=0)
print(f"Softmax: {softmax_output}")
print(f"Sum of Softmax outputs: {softmax_output.sum()}")  # Should be 1
```

## Loss Functions

Loss functions measure how well a neural network performs by comparing its predictions to the actual target values. The goal of training is to minimize this loss.

### Common Loss Functions:

1. **Mean Squared Error (MSE)**:
   - Used for regression problems
   - Measures the average squared difference between predictions and targets

2. **Cross-Entropy Loss**:
   - Used for classification problems
   - Measures the difference between predicted probability distributions and actual class labels

3. **Binary Cross-Entropy**:
   - Special case of cross-entropy for binary classification
   - Measures the performance of a model whose output is a probability value between 0 and 1

4. **Hinge Loss**:
   - Used for maximum-margin classification, like in SVMs
   - Encourages correct classifications to have a score higher than incorrect ones by a margin

### PyTorch Implementation:

```python
import torch
import torch.nn as nn

# MSE Loss
mse_loss = nn.MSELoss()
predictions = torch.tensor([0.5, 1.5, 2.5])
targets = torch.tensor([1.0, 2.0, 3.0])
mse_output = mse_loss(predictions, targets)
print(f"MSE Loss: {mse_output}")

# Cross-Entropy Loss
ce_loss = nn.CrossEntropyLoss()
logits = torch.tensor([[0.1, 0.2, 0.7], [0.3, 0.5, 0.2]])  # Batch of 2, 3 classes
targets = torch.tensor([2, 1])  # Class indices
ce_output = ce_loss(logits, targets)
print(f"Cross-Entropy Loss: {ce_output}")

# Binary Cross-Entropy Loss
bce_loss = nn.BCEWithLogitsLoss()
predictions = torch.tensor([0.7, -0.2, 0.9])
targets = torch.tensor([1.0, 0.0, 1.0])
bce_output = bce_loss(predictions, targets)
print(f"Binary Cross-Entropy Loss: {bce_output}")
```

## Optimizers

Optimizers update the weights and biases of a neural network to minimize the loss function. They implement various algorithms to adjust the parameters based on the gradients computed during backpropagation.

### Common Optimizers:

1. **Stochastic Gradient Descent (SGD)**:
   - The most basic optimizer
   - Updates parameters in the opposite direction of the gradient
   - Can include momentum to accelerate training

2. **Adam (Adaptive Moment Estimation)**:
   - Combines the benefits of AdaGrad and RMSProp
   - Adapts learning rates for each parameter
   - Generally performs well across a wide range of problems

3. **RMSProp**:
   - Adapts learning rates based on the recent magnitude of gradients
   - Helps with non-stationary objectives

4. **Adagrad**:
   - Adapts learning rates based on the historical gradient information
   - Good for sparse data

### PyTorch Implementation:

```python
import torch
import torch.nn as nn
import torch.optim as optim

# Define a simple model
model = nn.Linear(10, 1)

# SGD Optimizer
sgd_optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

# Adam Optimizer
adam_optimizer = optim.Adam(model.parameters(), lr=0.001, betas=(0.9, 0.999))

# RMSProp Optimizer
rmsprop_optimizer = optim.RMSprop(model.parameters(), lr=0.01, alpha=0.99)

# Adagrad Optimizer
adagrad_optimizer = optim.Adagrad(model.parameters(), lr=0.01)

# Example of using an optimizer in a training loop
optimizer = adam_optimizer
for epoch in range(5):
    # Forward pass (example)
    inputs = torch.randn(32, 10)  # Batch of 32, input dimension of 10
    targets = torch.randn(32, 1)  # Batch of 32, output dimension of 1
    outputs = model(inputs)
    loss = nn.MSELoss()(outputs, targets)
    
    # Backward pass and optimization
    optimizer.zero_grad()  # Clear previous gradients
    loss.backward()        # Compute gradients
    optimizer.step()       # Update parameters
    
    print(f"Epoch {epoch+1}, Loss: {loss.item()}")
```

## Building Your First Neural Network

In PyTorch, neural networks are built by creating a class that inherits from `nn.Module`. This class defines the network architecture and the forward pass.

### Example: Multi-Layer Perceptron (MLP)

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class MLP(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, output_size)
        
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# Create an instance of the MLP
input_size = 784  # e.g., for MNIST images (28x28)
hidden_size = 128
output_size = 10  # e.g., for 10 digit classes
model = MLP(input_size, hidden_size, output_size)

# Print the model architecture
print(model)

# Example forward pass
batch_size = 64
x = torch.randn(batch_size, input_size)
output = model(x)
print(f"Output shape: {output.shape}")  # Should be [64, 10]
```

## Forward and Backward Propagation

Neural networks learn through a process called backpropagation, which involves two main steps: forward propagation and backward propagation.

### Forward Propagation:
- Input data is passed through the network layer by layer
- Each layer applies its transformation and activation function
- The final layer produces the output (prediction)

### Backward Propagation:
- The loss is computed by comparing the prediction to the target
- Gradients of the loss with respect to the parameters are calculated
- Parameters are updated using an optimizer to minimize the loss

### PyTorch Implementation:

```python
import torch
import torch.nn as nn
import torch.optim as optim

# Define a simple model
class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(2, 3)
        self.fc2 = nn.Linear(3, 1)
        
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Create model, loss function, and optimizer
model = SimpleNN()
criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)

# Input and target
x = torch.tensor([[0.5, 0.7], [0.1, 0.9], [0.2, 0.3]], dtype=torch.float32)
y = torch.tensor([[1.0], [0.5], [0.7]], dtype=torch.float32)

# Training loop
for epoch in range(100):
    # Forward pass
    outputs = model(x)
    loss = criterion(outputs, y)
    
    # Backward pass and optimization
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    if (epoch+1) % 10 == 0:
        print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}")

# Test the model
with torch.no_grad():
    test_input = torch.tensor([[0.4, 0.6]], dtype=torch.float32)
    prediction = model(test_input)
    print(f"Prediction for input {test_input}: {prediction.item():.4f}")
```

## Practical Examples

Let's implement a complete example of training a neural network on a real dataset.

### Example: MNIST Digit Classification

```python
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Hyperparameters
input_size = 784  # 28x28
hidden_size = 500
num_classes = 10
num_epochs = 5
batch_size = 100
learning_rate = 0.001

# MNIST dataset
train_dataset = torchvision.datasets.MNIST(root='./data', 
                                          train=True, 
                                          transform=transforms.ToTensor(),
                                          download=True)

test_dataset = torchvision.datasets.MNIST(root='./data', 
                                         train=False, 
                                         transform=transforms.ToTensor())

# Data loader
train_loader = DataLoader(dataset=train_dataset, 
                          batch_size=batch_size, 
                          shuffle=True)

test_loader = DataLoader(dataset=test_dataset, 
                         batch_size=batch_size, 
                         shuffle=False)

# Fully connected neural network
class NeuralNet(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(NeuralNet, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size) 
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, num_classes)  
    
    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        return out

model = NeuralNet(input_size, hidden_size, num_classes).to(device)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Train the model
total_step = len(train_loader)
for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(train_loader):
        # Reshape images to (batch_size, input_size)
        images = images.reshape(-1, input_size).to(device)
        labels = labels.to(device)
        
        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if (i+1) % 100 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{total_step}], Loss: {loss.item():.4f}')

# Test the model
with torch.no_grad():
    correct = 0
    total = 0
    for images, labels in test_loader:
        images = images.reshape(-1, input_size).to(device)
        labels = labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    print(f'Accuracy of the network on the 10000 test images: {100 * correct / total}%')
```

## Conclusion

This tutorial covered the fundamental concepts of neural networks using PyTorch, including linear layers, activation functions, loss functions, optimizers, and the basics of building and training neural networks. These concepts form the foundation for more advanced topics in deep learning.

In the next tutorial, we'll explore automatic differentiation in more detail, which is a key component of PyTorch's computational framework.