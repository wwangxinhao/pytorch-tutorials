import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
import matplotlib.pyplot as plt
import numpy as np
import os

def linear_layer_example():
    """
    Demonstrates the usage of linear layers in PyTorch.
    """
    print("\n=== Linear Layer Example ===")
    
    # Define a linear layer
    linear = nn.Linear(in_features=10, out_features=5)
    
    # Input tensor
    x = torch.randn(3, 10)  # Batch size of 3, input dimension of 10
    
    # Forward pass
    output = linear(x)  # Shape: [3, 5]
    
    # Access weights and biases
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Weight shape: {linear.weight.shape}")  # Shape: [5, 10]
    print(f"Bias shape: {linear.bias.shape}")      # Shape: [5]
    
    # Multiple linear layers
    model = nn.Sequential(
        nn.Linear(10, 20),
        nn.ReLU(),
        nn.Linear(20, 15),
        nn.ReLU(),
        nn.Linear(15, 5)
    )
    
    # Forward pass through multiple layers
    output = model(x)
    print(f"Output shape after multiple layers: {output.shape}")  # Shape: [3, 5]

def activation_functions_example():
    """
    Demonstrates various activation functions in PyTorch.
    """
    print("\n=== Activation Functions Example ===")
    
    # Input tensor
    x = torch.tensor([-2.0, -1.0, 0.0, 1.0, 2.0])
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
    
    # Visualize activation functions
    plt.figure(figsize=(12, 8))
    
    # Generate data for plotting
    x_range = torch.linspace(-5, 5, 100)
    
    # ReLU
    plt.subplot(2, 3, 1)
    plt.plot(x_range, F.relu(x_range))
    plt.grid(True)
    plt.title('ReLU')
    
    # Sigmoid
    plt.subplot(2, 3, 2)
    plt.plot(x_range, torch.sigmoid(x_range))
    plt.grid(True)
    plt.title('Sigmoid')
    
    # Tanh
    plt.subplot(2, 3, 3)
    plt.plot(x_range, torch.tanh(x_range))
    plt.grid(True)
    plt.title('Tanh')
    
    # Leaky ReLU
    plt.subplot(2, 3, 4)
    plt.plot(x_range, F.leaky_relu(x_range, negative_slope=0.1))
    plt.grid(True)
    plt.title('Leaky ReLU (slope=0.1)')
    
    # ELU
    plt.subplot(2, 3, 5)
    plt.plot(x_range, F.elu(x_range))
    plt.grid(True)
    plt.title('ELU')
    
    # SELU
    plt.subplot(2, 3, 6)
    plt.plot(x_range, F.selu(x_range))
    plt.grid(True)
    plt.title('SELU')
    
    plt.tight_layout()
    plt.savefig('activation_functions.png')
    print("Activation functions visualization saved as 'activation_functions.png'")

def loss_functions_example():
    """
    Demonstrates various loss functions in PyTorch.
    """
    print("\n=== Loss Functions Example ===")
    
    # MSE Loss
    mse_loss = nn.MSELoss()
    predictions = torch.tensor([0.5, 1.5, 2.5])
    targets = torch.tensor([1.0, 2.0, 3.0])
    mse_output = mse_loss(predictions, targets)
    print(f"MSE Loss: {mse_output.item()}")
    
    # Cross-Entropy Loss
    ce_loss = nn.CrossEntropyLoss()
    logits = torch.tensor([[0.1, 0.2, 0.7], [0.3, 0.5, 0.2]])  # Batch of 2, 3 classes
    targets = torch.tensor([2, 1])  # Class indices
    ce_output = ce_loss(logits, targets)
    print(f"Cross-Entropy Loss: {ce_output.item()}")
    
    # Binary Cross-Entropy Loss
    bce_loss = nn.BCEWithLogitsLoss()
    predictions = torch.tensor([0.7, -0.2, 0.9])
    targets = torch.tensor([1.0, 0.0, 1.0])
    bce_output = bce_loss(predictions, targets)
    print(f"Binary Cross-Entropy Loss: {bce_output.item()}")
    
    # L1 Loss (Mean Absolute Error)
    l1_loss = nn.L1Loss()
    l1_output = l1_loss(predictions, targets)
    print(f"L1 Loss (MAE): {l1_output.item()}")
    
    # Smooth L1 Loss (Huber Loss)
    smooth_l1_loss = nn.SmoothL1Loss()
    smooth_l1_output = smooth_l1_loss(predictions, targets)
    print(f"Smooth L1 Loss (Huber): {smooth_l1_output.item()}")

def optimizers_example():
    """
    Demonstrates various optimizers in PyTorch.
    """
    print("\n=== Optimizers Example ===")
    
    # Define a simple model
    model = nn.Linear(10, 1)
    
    # Print initial parameters
    print("Initial parameters:")
    for name, param in model.named_parameters():
        print(f"{name}: {param.data.mean().item():.4f} (mean)")
    
    # Create different optimizers
    sgd_optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
    adam_optimizer = optim.Adam(model.parameters(), lr=0.01, betas=(0.9, 0.999))
    rmsprop_optimizer = optim.RMSprop(model.parameters(), lr=0.01, alpha=0.99)
    adagrad_optimizer = optim.Adagrad(model.parameters(), lr=0.01)
    
    # Example of using an optimizer in a training loop
    optimizer = adam_optimizer
    print(f"\nTraining with Adam optimizer:")
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
        
        print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}")
    
    # Print updated parameters
    print("\nUpdated parameters:")
    for name, param in model.named_parameters():
        print(f"{name}: {param.data.mean().item():.4f} (mean)")

class MLP(nn.Module):
    """
    Multi-Layer Perceptron (MLP) implementation.
    """
    def __init__(self, input_size, hidden_size, output_size, dropout_rate=0.2):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, output_size)
        self.dropout = nn.Dropout(dropout_rate)
        
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        return x

def build_neural_network_example():
    """
    Demonstrates how to build a neural network in PyTorch.
    """
    print("\n=== Building Neural Network Example ===")
    
    # Create an instance of the MLP
    input_size = 784  # e.g., for MNIST images (28x28)
    hidden_size = 128
    output_size = 10  # e.g., for 10 digit classes
    model = MLP(input_size, hidden_size, output_size)
    
    # Print the model architecture
    print(model)
    
    # Count the number of parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total number of parameters: {total_params}")
    
    # Example forward pass
    batch_size = 64
    x = torch.randn(batch_size, input_size)
    output = model(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")  # Should be [64, 10]

def forward_backward_propagation_example():
    """
    Demonstrates forward and backward propagation in PyTorch.
    """
    print("\n=== Forward and Backward Propagation Example ===")
    
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
    optimizer = optim.SGD(model.parameters(), lr=0.1)
    
    # Input and target
    x = torch.tensor([[0.5, 0.7], [0.1, 0.9], [0.2, 0.3]], dtype=torch.float32)
    y = torch.tensor([[1.0], [0.5], [0.7]], dtype=torch.float32)
    
    print("Initial predictions:")
    with torch.no_grad():
        initial_pred = model(x)
        print(f"Inputs: {x}")
        print(f"Targets: {y}")
        print(f"Predictions: {initial_pred}")
        initial_loss = criterion(initial_pred, y)
        print(f"Initial loss: {initial_loss.item():.4f}")
    
    # Training loop
    print("\nTraining:")
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
    print("\nFinal predictions:")
    with torch.no_grad():
        final_pred = model(x)
        print(f"Predictions: {final_pred}")
        final_loss = criterion(final_pred, y)
        print(f"Final loss: {final_loss.item():.4f}")
        
        test_input = torch.tensor([[0.4, 0.6]], dtype=torch.float32)
        prediction = model(test_input)
        print(f"Prediction for input {test_input}: {prediction.item():.4f}")

class SyntheticDataset(Dataset):
    """
    Synthetic dataset for binary classification.
    """
    def __init__(self, num_samples=1000, input_dim=2):
        self.num_samples = num_samples
        self.input_dim = input_dim
        
        # Generate random data
        self.data = torch.randn(num_samples, input_dim)
        
        # Generate labels: points inside a circle of radius 1 are labeled 1, others 0
        self.labels = torch.zeros(num_samples)
        for i in range(num_samples):
            if torch.norm(self.data[i]) < 1:
                self.labels[i] = 1
    
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]

def binary_classification_example():
    """
    Demonstrates a complete binary classification example.
    """
    print("\n=== Binary Classification Example ===")
    
    # Create synthetic dataset
    train_dataset = SyntheticDataset(num_samples=1000)
    test_dataset = SyntheticDataset(num_samples=200)
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    # Define the model
    class BinaryClassifier(nn.Module):
        def __init__(self):
            super(BinaryClassifier, self).__init__()
            self.fc1 = nn.Linear(2, 10)
            self.fc2 = nn.Linear(10, 10)
            self.fc3 = nn.Linear(10, 1)
            
        def forward(self, x):
            x = F.relu(self.fc1(x))
            x = F.relu(self.fc2(x))
            x = torch.sigmoid(self.fc3(x))
            return x
    
    # Create model, loss function, and optimizer
    model = BinaryClassifier()
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    
    # Training loop
    num_epochs = 10
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for i, (inputs, labels) in enumerate(train_loader):
            # Forward pass
            outputs = model(inputs).squeeze()
            loss = criterion(outputs, labels)
            
            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
        
        # Print statistics
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {running_loss/len(train_loader):.4f}")
    
    # Evaluate the model
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in test_loader:
            outputs = model(inputs).squeeze()
            predicted = (outputs > 0.5).float()
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    accuracy = 100 * correct / total
    print(f"Accuracy on test set: {accuracy:.2f}%")
    
    # Visualize the decision boundary
    plt.figure(figsize=(10, 8))
    
    # Plot the test data
    x_data = test_dataset.data.numpy()
    y_data = test_dataset.labels.numpy()
    
    plt.scatter(x_data[y_data==0, 0], x_data[y_data==0, 1], c='red', label='Class 0')
    plt.scatter(x_data[y_data==1, 0], x_data[y_data==1, 1], c='blue', label='Class 1')
    
    # Create a grid to evaluate the model
    x_min, x_max = x_data[:, 0].min() - 1, x_data[:, 0].max() + 1
    y_min, y_max = x_data[:, 1].min() - 1, x_data[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.01),
                         np.arange(y_min, y_max, 0.01))
    
    # Evaluate the model on the grid
    model.eval()
    with torch.no_grad():
        grid = torch.tensor(np.c_[xx.ravel(), yy.ravel()], dtype=torch.float32)
        outputs = model(grid).squeeze().numpy()
        outputs = outputs.reshape(xx.shape)
    
    # Plot the decision boundary
    plt.contourf(xx, yy, outputs, alpha=0.3, levels=np.linspace(0, 1, 11))
    plt.colorbar()
    plt.contour(xx, yy, outputs, colors='black', levels=[0.5])
    plt.title('Decision Boundary')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.legend()
    plt.grid(True)
    plt.savefig('decision_boundary.png')
    print("Decision boundary visualization saved as 'decision_boundary.png'")

def mnist_example():
    """
    Demonstrates a complete example of training a neural network on MNIST.
    """
    print("\n=== MNIST Classification Example ===")
    
    # Check if CUDA is available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Hyperparameters
    input_size = 784  # 28x28
    hidden_size = 500
    num_classes = 10
    num_epochs = 2  # Reduced for demonstration
    batch_size = 100
    learning_rate = 0.001
    
    # Create data directory if it doesn't exist
    os.makedirs('./data', exist_ok=True)
    
    try:
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
        model.eval()
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
        
        # Visualize some predictions
        dataiter = iter(test_loader)
        images, labels = next(dataiter)
        
        # Get predictions
        images_flat = images.reshape(-1, input_size).to(device)
        outputs = model(images_flat)
        _, predicted = torch.max(outputs, 1)
        
        # Plot the images and predictions
        plt.figure(figsize=(12, 8))
        for i in range(min(25, batch_size)):
            plt.subplot(5, 5, i+1)
            plt.imshow(images[i].squeeze().numpy(), cmap='gray')
            plt.title(f'Pred: {predicted[i].item()}, True: {labels[i].item()}')
            plt.axis('off')
        
        plt.tight_layout()
        plt.savefig('mnist_predictions.png')
        print("MNIST predictions visualization saved as 'mnist_predictions.png'")
    
    except Exception as e:
        print(f"Error in MNIST example: {e}")
        print("Skipping MNIST example due to error.")

def main():
    """
    Main function to run all examples.
    """
    print("PyTorch Neural Networks Fundamentals")
    print(f"PyTorch version: {torch.__version__}")
    
    # Run examples
    linear_layer_example()
    activation_functions_example()
    loss_functions_example()
    optimizers_example()
    build_neural_network_example()
    forward_backward_propagation_example()
    binary_classification_example()
    
    # MNIST example is optional as it requires downloading the dataset
    try:
        mnist_example()
    except Exception as e:
        print(f"Error in MNIST example: {e}")
        print("Skipping MNIST example. You can run it separately if needed.")
    
    print("\nAll examples completed!")

if __name__ == "__main__":
    main()