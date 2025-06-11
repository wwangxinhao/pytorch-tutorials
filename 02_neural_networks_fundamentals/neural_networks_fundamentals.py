#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Neural Networks Fundamentals in PyTorch

This script provides a comprehensive introduction to neural networks, covering basic concepts,
activation functions, multi-layer perceptrons, loss functions, optimizers, and building
and training a simple neural network in PyTorch.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
# torchvision can be used for datasets like MNIST, but for this fundamental script,
# we'll use synthetic data or simple tensors to keep dependencies minimal.
# import torchvision
# import torchvision.transforms as transforms
from torch.utils.data import DataLoader, TensorDataset # Using TensorDataset for simple examples
import matplotlib.pyplot as plt
import numpy as np
import os

# Set random seed for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Create output directory if it doesn't exist
output_dir = "02_neural_networks_fundamentals_outputs"
os.makedirs(output_dir, exist_ok=True)

# -----------------------------------------------------------------------------
# Section 1: Introduction to Neural Networks
# (Conceptual, covered in README.md)
# -----------------------------------------------------------------------------

def intro_to_neural_networks_concepts():
    """Prints a summary of Neural Network concepts (mainly for script flow)."""
    print("\nSection 1: Introduction to Neural Networks")
    print("-" * 70)
    print("Key Concepts (Detailed in README.md):")
    print("  - What is a Neural Network? Biological Inspiration.")
    print("  - Basic Components: Neurons, Weights, Biases, Layers.")
    print("  - Types of Neural Networks (FNNs, CNNs, RNNs). Focus here: FNNs/MLPs.")

# -----------------------------------------------------------------------------
# Section 2: The Perceptron: The Simplest Neural Network
# (Conceptual, covered in README.md, a simple version is part of activation demo)
# -----------------------------------------------------------------------------

def demonstrate_perceptron_concept():
    """Illustrates a simple perceptron with a step-like activation (sigmoid)."""
    print("\nSection 2: The Perceptron: The Simplest Neural Network")
    print("-" * 70)
    print("A single-layer perceptron is the most basic neural network.")
    print("It computes a weighted sum of inputs, adds a bias, and passes it through an activation function.")
    
    # Example: A perceptron with 2 inputs
    perceptron_layer = nn.Linear(2, 1).to(device) # 2 input features, 1 output feature
    sample_input = torch.tensor([0.5, -1.0], device=device) # Example input
    weighted_sum = perceptron_layer(sample_input)
    # Typically, a step function was used historically. Here we use sigmoid for a softer step.
    output = torch.sigmoid(weighted_sum)
    
    print(f"Sample input: {sample_input.cpu().numpy()}")
    print(f"Perceptron layer weights: {perceptron_layer.weight.data.cpu().numpy()}")
    print(f"Perceptron layer bias: {perceptron_layer.bias.data.cpu().numpy()}")
    print(f"Weighted sum + bias: {weighted_sum.item():.4f}")
    print(f"Output after sigmoid (acting as a soft step): {output.item():.4f}")
    print("A single-layer perceptron can only solve linearly separable problems.")

# -----------------------------------------------------------------------------
# Section 3: Activation Functions
# -----------------------------------------------------------------------------

def demonstrate_activation_functions():
    """Demonstrates and plots common activation functions."""
    print("\nSection 3: Activation Functions")
    print("-" * 70)
    print("Activation functions introduce non-linearity, enabling networks to learn complex patterns.")
    
    x_vals = torch.linspace(-6, 6, 100) # Input values for plotting
    
    # Sigmoid
    sigmoid_fn = nn.Sigmoid()
    y_sigmoid = sigmoid_fn(x_vals)
    
    # Tanh
    tanh_fn = nn.Tanh()
    y_tanh = tanh_fn(x_vals)
    
    # ReLU
    relu_fn = nn.ReLU()
    y_relu = relu_fn(x_vals)
    
    # Leaky ReLU
    leaky_relu_fn = nn.LeakyReLU(negative_slope=0.1)
    y_leaky_relu = leaky_relu_fn(x_vals)
    
    # Softmax (applied to a sample batch of logits)
    softmax_fn = nn.Softmax(dim=1)
    sample_logits = torch.tensor([[1.0, -0.5, 2.0], [0.1, 0.5, 0.2]]) # Batch of 2, 3 classes
    y_softmax = softmax_fn(sample_logits)
    print(f"Sample Logits for Softmax:\n{sample_logits}")
    print(f"Softmax Output:\n{y_softmax}")

    # Plotting
    plt.figure(figsize=(15, 10))
    
    plt.subplot(2, 2, 1)
    plt.plot(x_vals.numpy(), y_sigmoid.numpy(), label='Sigmoid')
    plt.title('Sigmoid: 1 / (1 + exp(-x))')
    plt.xlabel('x'); plt.ylabel('f(x)'); plt.grid(True); plt.legend()
    
    plt.subplot(2, 2, 2)
    plt.plot(x_vals.numpy(), y_tanh.numpy(), label='Tanh')
    plt.title('Tanh: (exp(x) - exp(-x)) / (exp(x) + exp(-x))')
    plt.xlabel('x'); plt.ylabel('f(x)'); plt.grid(True); plt.legend()

    plt.subplot(2, 2, 3)
    plt.plot(x_vals.numpy(), y_relu.numpy(), label='ReLU')
    plt.title('ReLU: max(0, x)')
    plt.xlabel('x'); plt.ylabel('f(x)'); plt.grid(True); plt.legend()
    
    plt.subplot(2, 2, 4)
    plt.plot(x_vals.numpy(), y_leaky_relu.numpy(), label='Leaky ReLU (slope=0.1)')
    plt.title('Leaky ReLU: max(0.1*x, x)')
    plt.xlabel('x'); plt.ylabel('f(x)'); plt.grid(True); plt.legend()
    
    plt.tight_layout()
    plot_path = os.path.join(output_dir, 'activation_functions_plot.png')
    plt.savefig(plot_path)
    plt.close()
    print(f"Activation functions plot saved to '{plot_path}'")

# -----------------------------------------------------------------------------
# Section 4: Multi-Layer Perceptrons (MLPs)
# -----------------------------------------------------------------------------

class SimpleMLP(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(SimpleMLP, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size) # Input layer to hidden layer
        self.relu1 = nn.ReLU()                        # Activation function for first hidden layer
        self.fc2 = nn.Linear(hidden_size, hidden_size // 2) # Hidden layer to another smaller hidden layer
        self.relu2 = nn.ReLU()                        # Activation for second hidden layer
        self.fc3 = nn.Linear(hidden_size // 2, num_classes) # Final hidden layer to output layer

    def forward(self, x):
        # x is the input tensor: [batch_size, input_size]
        print(f"  MLP Forward - Input shape: {x.shape}")
        out = self.fc1(x)
        print(f"  MLP Forward - After fc1: {out.shape}")
        out = self.relu1(out)
        print(f"  MLP Forward - After relu1: {out.shape}")
        out = self.fc2(out)
        print(f"  MLP Forward - After fc2: {out.shape}")
        out = self.relu2(out)
        print(f"  MLP Forward - After relu2: {out.shape}")
        out = self.fc3(out)
        print(f"  MLP Forward - Output shape (logits): {out.shape}")
        # Note: Softmax is typically applied outside the model if using nn.CrossEntropyLoss
        return out

def demonstrate_mlp():
    print("\nSection 4: Multi-Layer Perceptrons (MLPs)")
    print("-" * 70)
    print("MLPs consist of an input layer, one or more hidden layers, and an output layer.")
    
    input_dim = 100  # Example input feature dimension
    hidden_dim = 64
    output_dim = 5   # Example number of classes for classification
    
    model = SimpleMLP(input_dim, hidden_dim, output_dim).to(device)
    print("\nMLP Architecture:")
    print(model)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total trainable parameters in MLP: {total_params}")

    print("\nDemonstrating Forward Propagation through MLP:")
    batch_size = 4
    dummy_input = torch.randn(batch_size, input_dim).to(device) # [batch_size, input_features]
    print(f"Dummy input batch shape: {dummy_input.shape}")
    
    # Perform a forward pass
    with torch.no_grad(): # We don't need gradients for this demonstration
        predictions = model(dummy_input)
    print(f"MLP output predictions shape: {predictions.shape}") # Expected: [batch_size, num_classes]
    print("Output from MLP (first sample in batch - raw logits):")
    print(predictions[0].cpu().numpy())

# -----------------------------------------------------------------------------
# Section 5: Defining a Neural Network in PyTorch (nn.Module)
# (Covered by SimpleMLP class definition in Section 4)
# -----------------------------------------------------------------------------

def recap_nn_module():
    print("\nSection 5: Defining a Neural Network in PyTorch (`nn.Module`)")
    print("-" * 70)
    print("PyTorch networks are built by subclassing `nn.Module`.")
    print("  - Layers are defined as attributes in `__init__`.")
    print("  - The `forward` method defines the data flow.")
    print("  - `SimpleMLP` class (shown above) is an example.")

# -----------------------------------------------------------------------------
# Section 6: Loss Functions: Measuring Model Error
# -----------------------------------------------------------------------------

def demonstrate_loss_functions():
    print("\nSection 6: Loss Functions: Measuring Model Error")
    print("-" * 70)
    print("Loss functions quantify the difference between model predictions and true targets.")

    # Mean Squared Error (MSE) - For Regression
    loss_mse_fn = nn.MSELoss()
    predictions_reg = torch.tensor([1.0, 2.5, 3.8], device=device) # Model outputs
    targets_reg = torch.tensor([1.2, 2.3, 4.0], device=device)   # True values
    mse = loss_mse_fn(predictions_reg, targets_reg)
    print(f"\nMSE Loss Example (Regression):")
    print(f"  Predictions: {predictions_reg.cpu().numpy()}")
    print(f"  Targets: {targets_reg.cpu().numpy()}")
    print(f"  Calculated MSE Loss: {mse.item():.4f}")

    # Cross-Entropy Loss - For Multi-class Classification
    loss_ce_fn = nn.CrossEntropyLoss()
    # Raw logits from the model (batch_size, num_classes)
    predictions_mc = torch.tensor([[2.0, 0.5, -1.0], [0.1, 1.5, 0.2]], device=device) # 2 samples, 3 classes
    targets_mc = torch.tensor([0, 1], device=device) # True class indices for each sample
    ce = loss_ce_fn(predictions_mc, targets_mc)
    print(f"\nCross-Entropy Loss Example (Multi-class Classification):")
    print(f"  Predictions (logits):\n{predictions_mc.cpu().numpy()}")
    print(f"  Targets (class indices): {targets_mc.cpu().numpy()}")
    print(f"  Calculated Cross-Entropy Loss: {ce.item():.4f}")
    # Note: nn.CrossEntropyLoss combines LogSoftmax and NLLLoss.

    # Binary Cross-Entropy with Logits Loss - For Binary Classification
    loss_bce_logits_fn = nn.BCEWithLogitsLoss()
    # Raw logits for binary classification (batch_size, 1) or (batch_size,)
    predictions_bc = torch.tensor([-0.5, 1.5, -2.0, 3.0], device=device).unsqueeze(1) # 4 samples, 1 logit each
    targets_bc = torch.tensor([0.0, 1.0, 0.0, 1.0], device=device).unsqueeze(1)     # True binary labels (0 or 1)
    bce_wl = loss_bce_logits_fn(predictions_bc, targets_bc)
    print(f"\nBinary Cross-Entropy with Logits Loss Example (Binary Classification):")
    print(f"  Predictions (logits):\n{predictions_bc.cpu().numpy()}")
    print(f"  Targets (0 or 1):\n{targets_bc.cpu().numpy()}")
    print(f"  Calculated BCEWithLogits Loss: {bce_wl.item():.4f}")

# -----------------------------------------------------------------------------
# Section 7: Optimizers: How Neural Networks Learn
# -----------------------------------------------------------------------------

def demonstrate_optimizers():
    print("\nSection 7: Optimizers: How Neural Networks Learn")
    print("-" * 70)
    print("Optimizers adjust model parameters (weights & biases) to minimize the loss function.")

    # Create a dummy model for optimizer demonstration
    dummy_model = nn.Linear(10, 2).to(device) # 10 input features, 2 output features
    print(f"Dummy model parameters before optimization (first weight): {dummy_model.weight[0,0].item():.4f}")

    # Stochastic Gradient Descent (SGD)
    optimizer_sgd = optim.SGD(dummy_model.parameters(), lr=0.01, momentum=0.9)
    print(f"\nOptimizer: SGD with lr=0.01, momentum=0.9")

    # Adam Optimizer
    optimizer_adam = optim.Adam(dummy_model.parameters(), lr=0.001)
    print(f"Optimizer: Adam with lr=0.001")

    # Conceptual optimization step (requires a loss and .backward() call)
    # Let's simulate a gradient update for SGD
    # Create dummy input, target, and loss
    dummy_input_opt = torch.randn(5, 10).to(device)
    dummy_target_opt = torch.randn(5, 2).to(device)
    criterion_opt = nn.MSELoss()
    
    # --- SGD Example Step ---
    optimizer_sgd.zero_grad()                   # Clear previous gradients
    outputs_opt = dummy_model(dummy_input_opt)  # Forward pass
    loss_opt = criterion_opt(outputs_opt, dummy_target_opt) # Calculate loss
    loss_opt.backward()                         # Backward pass (compute gradients)
    optimizer_sgd.step()                        # Update weights
    print(f"Dummy model parameters after ONE SGD step (first weight): {dummy_model.weight[0,0].item():.4f}")
    
    # Reset model parameters for Adam demo (not perfect, but for illustration)
    dummy_model_adam = nn.Linear(10,2).to(device)
    optimizer_adam = optim.Adam(dummy_model_adam.parameters(), lr=0.001)
    print(f"Dummy Adam model parameters before optimization (first weight): {dummy_model_adam.weight[0,0].item():.4f}")

    # --- Adam Example Step ---
    optimizer_adam.zero_grad()
    outputs_opt_adam = dummy_model_adam(dummy_input_opt) 
    loss_opt_adam = criterion_opt(outputs_opt_adam, dummy_target_opt)
    loss_opt_adam.backward()
    optimizer_adam.step()
    print(f"Dummy Adam model parameters after ONE Adam step (first weight): {dummy_model_adam.weight[0,0].item():.4f}")
    print("Learning rate is a key hyperparameter for optimizers.")

# -----------------------------------------------------------------------------
# Section 8: The Training Loop: Forward and Backward Propagation
# (Conceptual, detailed in README and demonstrated in Section 9)
# -----------------------------------------------------------------------------

def recap_training_loop():
    print("\nSection 8: The Training Loop: Forward and Backward Propagation")
    print("-" * 70)
    print("The training loop is the core of model training:")
    print("  1. Forward Propagation: Get predictions, calculate loss.")
    print("  2. optimizer.zero_grad(): Clear old gradients.")
    print("  3. loss.backward(): Compute current gradients (Backpropagation).")
    print("  4. optimizer.step(): Update model parameters.")
    print("  This process is repeated over epochs and batches of data.")

# -----------------------------------------------------------------------------
# Section 9: Building and Training Your First Neural Network in PyTorch
# -----------------------------------------------------------------------------

def build_and_train_first_nn():
    print("\nSection 9: Building and Training Your First Neural Network in PyTorch")
    print("-" * 70)
    print("Example: Solving the XOR problem, a non-linearly separable task.")

    # --- Step 1: Prepare the Data ---
    print("\nStep 1: Prepare the Data (XOR Problem)")
    # Inputs for XOR: (0,0), (0,1), (1,0), (1,1)
    X_xor = torch.tensor([[0., 0.], [0., 1.], [1., 0.], [1., 1.]], device=device)
    # Outputs for XOR: 0, 1, 1, 0
    y_xor = torch.tensor([[0.], [1.], [1.], [0.]], device=device)
    
    # Create a simple Dataset and DataLoader
    xor_dataset = TensorDataset(X_xor, y_xor)
    # For XOR, batch_size can be the full dataset size (4) as it's very small.
    # If we had a larger dataset, we'd use a smaller batch_size (e.g., 32, 64).
    xor_dataloader = DataLoader(xor_dataset, batch_size=4, shuffle=True)
    print(f"X_xor inputs:\n{X_xor.cpu().numpy()}")
    print(f"y_xor targets:\n{y_xor.cpu().numpy()}")

    # --- Step 2: Define the Model ---
    print("\nStep 2: Define the Model (XORNet)")
    class XORNet(nn.Module):
        def __init__(self):
            super(XORNet, self).__init__()
            self.fc1 = nn.Linear(2, 8)      # 2 input features, 8 neurons in hidden layer
            self.relu = nn.ReLU()
            self.fc2 = nn.Linear(8, 1)      # 8 hidden neurons, 1 output neuron
            # Sigmoid will be applied implicitly by BCEWithLogitsLoss or explicitly after if using BCELoss

        def forward(self, x):
            x = self.fc1(x)
            x = self.relu(x)
            x = self.fc2(x)
            return x # Output raw logits

    xor_model = XORNet().to(device)
    print("XORNet Architecture:")
    print(xor_model)

    # --- Step 3: Define Loss Function and Optimizer ---
    print("\nStep 3: Define Loss Function and Optimizer")
    criterion = nn.BCEWithLogitsLoss() # Handles sigmoid internally, more stable
    optimizer = optim.Adam(xor_model.parameters(), lr=0.05) # Adam with a slightly higher LR for faster convergence on XOR
    print(f"Loss Function: {criterion}")
    print(f"Optimizer: {optimizer}")

    # --- Step 4: Implement the Training Loop ---
    print("\nStep 4: Implement the Training Loop")
    num_epochs = 1000
    losses_history = []
    
    for epoch in range(num_epochs):
        for inputs, labels in xor_dataloader: # Dataloader handles batching
            # Inputs and labels are already on `device` if X_xor, y_xor were created on device
            
            # Forward pass
            outputs = xor_model(inputs) # Model outputs raw logits
            loss = criterion(outputs, labels)
            
            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        losses_history.append(loss.item())
        if (epoch + 1) % 100 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

    # Plot training loss
    plt.figure(figsize=(10, 6))
    plt.plot(losses_history)
    plt.title('Training Loss for XOR Problem')
    plt.xlabel('Epoch')
    plt.ylabel('BCEWithLogitsLoss')
    plt.grid(True)
    loss_plot_path = os.path.join(output_dir, 'xor_training_loss.png')
    plt.savefig(loss_plot_path)
    plt.close()
    print(f"Training loss plot saved to '{loss_plot_path}'")

    # --- Step 5: Evaluate the Model (Conceptual) ---
    print("\nStep 5: Evaluate the Model")
    xor_model.eval() # Set model to evaluation mode (important for layers like dropout, batchnorm)
    with torch.no_grad(): # Disable gradient calculations for inference
        test_predictions_logits = xor_model(X_xor)
        # Apply sigmoid to logits to get probabilities for evaluation
        test_predictions_probs = torch.sigmoid(test_predictions_logits)
        # Convert probabilities to binary classes (0 or 1) based on a 0.5 threshold
        predicted_classes = (test_predictions_probs >= 0.5).float()
        
        accuracy = (predicted_classes == y_xor).float().mean()
        print(f"\nFinal Accuracy on XOR dataset: {accuracy.item()*100:.2f}%")
        print("Input  | True Output | Predicted Prob | Predicted Class")
        print("-----------------------------------------------------")
        for i in range(len(X_xor)):
            print(f"{X_xor[i].cpu().numpy()} | {y_xor[i].item():.0f}           | {test_predictions_probs[i].item():.4f}         | {predicted_classes[i].item():.0f}")
    xor_model.train() # Set model back to training mode if further training is planned

# -----------------------------------------------------------------------------
# Main function to run all sections
# -----------------------------------------------------------------------------

def main():
    """Main function to run all neural networks fundamentals tutorial sections."""
    print("=" * 80)
    print("PyTorch Neural Networks Fundamentals Tutorial")
    print("=" * 80)
    
    intro_to_neural_networks_concepts()
    demonstrate_perceptron_concept()
    demonstrate_activation_functions()
    demonstrate_mlp()
    recap_nn_module()
    demonstrate_loss_functions()
    demonstrate_optimizers()
    recap_training_loop()
    build_and_train_first_nn()
    
    print("\nTutorial complete! Outputs (like plots) are in the '02_neural_networks_fundamentals_outputs' directory.")

if __name__ == '__main__':
    main()