# Neural Networks Fundamentals in PyTorch

This tutorial provides a comprehensive introduction to the fundamental concepts of neural networks and their implementation using PyTorch. We will cover the building blocks of neural networks, how they learn, and how to construct your first neural network.

## Table of Contents
1. [Introduction to Neural Networks](#introduction-to-neural-networks)
   - What is a Neural Network?
   - Biological Inspiration
   - Basic Components: Neurons, Weights, Biases, Layers
   - Types of Neural Networks (Brief Overview)
2. [The Perceptron: The Simplest Neural Network](#the-perceptron-the-simplest-neural-network)
   - Single-Layer Perceptron
   - Linear Separability
3. [Activation Functions](#activation-functions)
   - Purpose: Introducing Non-linearity
   - Common Activation Functions:
     - Sigmoid
     - Tanh (Hyperbolic Tangent)
     - ReLU (Rectified Linear Unit) and its variants (Leaky ReLU, ELU)
     - Softmax (for output layers in classification)
   - Choosing an Activation Function
   - PyTorch Implementation
4. [Multi-Layer Perceptrons (MLPs)](#multi-layer-perceptrons-mlps)
   - Architecture: Input, Hidden, and Output Layers
   - The Power of Hidden Layers: Universal Approximation Theorem (Concept)
   - Forward Propagation in an MLP
5. [Defining a Neural Network in PyTorch (`nn.Module`)](#defining-a-neural-network-in-pytorch-nnmodule)
   - The `nn.Module` Class
   - Defining Layers (`nn.Linear`, etc.)
   - Implementing the `forward` method
   - Example: A Simple MLP for Classification
6. [Loss Functions: Measuring Model Error](#loss-functions-measuring-model-error)
   - Purpose of Loss Functions
   - Common Loss Functions:
     - Mean Squared Error (MSE) (`nn.MSELoss`): For Regression
     - Cross-Entropy Loss (`nn.CrossEntropyLoss`): For Multi-class Classification
     - Binary Cross-Entropy Loss (`nn.BCELoss`, `nn.BCEWithLogitsLoss`): For Binary Classification
   - Choosing the Right Loss Function
7. [Optimizers: How Neural Networks Learn](#optimizers-how-neural-networks-learn)
   - Gradient Descent (Concept)
   - Stochastic Gradient Descent (SGD)
   - SGD with Momentum
   - Adam Optimizer (`torch.optim.Adam`)
   - Learning Rate
   - Linking Optimizers to Model Parameters
8. [The Training Loop: Forward and Backward Propagation](#the-training-loop-forward-and-backward-propagation)
   - Overview of the Training Process
   - **Forward Propagation:** Calculating Predictions and Loss
   - **Backward Propagation (Backpropagation):** Calculating Gradients (`loss.backward()`)
   - **Optimizer Step:** Updating Weights (`optimizer.step()`)
   - Zeroing Gradients (`optimizer.zero_grad()`)
   - Iterating over Data (Epochs and Batches)
9. [Building and Training Your First Neural Network in PyTorch](#building-and-training-your-first-neural-network-in-pytorch)
   - Step 1: Prepare the Data (e.g., a simple synthetic dataset)
   - Step 2: Define the Model (using `nn.Module`)
   - Step 3: Define Loss Function and Optimizer
   - Step 4: Implement the Training Loop
   - Step 5: Evaluate the Model (Conceptual)

## Introduction to Neural Networks

- **What is a Neural Network?**
  An Artificial Neural Network (ANN) is a computational model inspired by the structure and function of biological neural networks in the human brain. It consists of interconnected processing units called neurons (or nodes) organized in layers.
- **Biological Inspiration:** Neurons in the brain receive signals, process them, and transmit signals to other neurons. ANNs attempt to mimic this behavior mathematically.
- **Basic Components:**
  - **Neurons (Nodes):** Basic computational units that receive inputs, perform a calculation (typically a weighted sum followed by an activation function), and produce an output.
  - **Weights:** Parameters associated with each input to a neuron, representing the strength or importance of that input.
  - **Biases:** Additional parameters added to the weighted sum, allowing the neuron to be activated even when all inputs are zero, or shifting the activation function.
  - **Layers:** Neurons are organized into layers: an input layer, one or more hidden layers, and an output layer.
- **Types of Neural Networks:** Feedforward Neural Networks (FNNs), Convolutional Neural Networks (CNNs), Recurrent Neural Networks (RNNs), Transformers, etc. This tutorial focuses on FNNs (specifically MLPs).

## The Perceptron: The Simplest Neural Network

- **Single-Layer Perceptron:** The simplest form of a neural network, consisting of a single layer of output neurons. Inputs are fed directly to the outputs via a series of weights. It performs a weighted sum of inputs and applies an activation function (often a step function).
  `output = activation(sum(weights_i * input_i) + bias)`
- **Linear Separability:** A single-layer perceptron can only solve linearly separable problems.

## Activation Functions

- **Purpose:** Activation functions introduce non-linearity into the network. Without non-linearity, a multi-layer network would behave like a single-layer linear network, severely limiting its ability to model complex relationships.
- **Common Activation Functions:**
  - **Sigmoid:** `f(x) = 1 / (1 + exp(-x))`. Squashes values between 0 and 1. Used in older networks, can suffer from vanishing gradients.
  - **Tanh (Hyperbolic Tangent):** `f(x) = (exp(x) - exp(-x)) / (exp(x) + exp(-x))`. Squashes values between -1 and 1. Also prone to vanishing gradients but often preferred over sigmoid in hidden layers as it's zero-centered.
  - **ReLU (Rectified Linear Unit):** `f(x) = max(0, x)`. Computationally efficient, helps alleviate vanishing gradients. Most popular choice for hidden layers.
  - **Leaky ReLU:** `f(x) = max(0.01*x, x)`. Addresses the "dying ReLU" problem by allowing a small, non-zero gradient when the unit is not active.
  - **Softmax:** `f(x_i) = exp(x_i) / sum(exp(x_j))`. Used in the output layer of multi-class classification tasks to convert raw scores (logits) into probabilities that sum to 1.

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

# Examples of activation functions
sigmoid = nn.Sigmoid()
relu = nn.ReLU()
tanh = nn.Tanh()
softmax = nn.Softmax(dim=1) # Apply softmax across a specific dimension

input_tensor = torch.randn(2, 3) # Batch of 2, 3 features each
print("Input:\n", input_tensor)
print("Sigmoid output:\n", sigmoid(input_tensor))
print("ReLU output:\n", relu(input_tensor))
print("Tanh output:\n", tanh(input_tensor))
# For softmax, let's assume these are logits for 2 samples, 3 classes
print("Softmax output:\n", softmax(input_tensor))
```

## Multi-Layer Perceptrons (MLPs)

MLPs are feedforward neural networks with one or more hidden layers between the input and output layers. Each layer is fully connected to the next.

- **Architecture:**
  - **Input Layer:** Receives the raw input data.
  - **Hidden Layer(s):** Perform intermediate computations. The number of hidden layers and neurons per layer are hyperparameters.
  - **Output Layer:** Produces the final prediction.
- **Universal Approximation Theorem:** (Conceptual) An MLP with at least one hidden layer and a non-linear activation function can approximate any continuous function to an arbitrary degree of accuracy, given enough neurons.
- **Forward Propagation:** The process of passing input data through the network layer by layer to compute the output.
  `h1 = activation1(W1*x + b1)`
  `h2 = activation2(W2*h1 + b2)`
  `output = activation_out(W_out*h2 + b_out)`

## Defining a Neural Network in PyTorch (`nn.Module`)

PyTorch provides the `nn.Module` class as a base for all neural network modules.

- **The `nn.Module` Class:**
  - Your custom network should inherit from `nn.Module`.
  - Layers are defined as attributes in the `__init__` method.
  - The `forward` method defines how input data flows through the network.
- **Defining Layers:** PyTorch offers various predefined layers in `torch.nn`:
  - `nn.Linear(in_features, out_features)`: Applies a linear transformation (fully connected layer).
  - `nn.Conv2d`, `nn.RNN`, etc., for other network types.

```python
class SimpleMLP(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(SimpleMLP, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size) # Input layer to hidden layer
        self.relu = nn.ReLU()                         # Activation function
        self.fc2 = nn.Linear(hidden_size, num_classes) # Hidden layer to output layer

    def forward(self, x):
        # x is the input tensor
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        # No softmax here if using nn.CrossEntropyLoss, as it combines Softmax and NLLLoss
        return out

# Example usage
input_dim = 784 # e.g., for flattened 28x28 MNIST images
hidden_dim = 128
output_dim = 10   # e.g., for 10 digit classes
model_mlp = SimpleMLP(input_dim, hidden_dim, output_dim)
print(model_mlp)
```

## Loss Functions: Measuring Model Error

Loss functions (or cost functions) quantify how far the model's predictions are from the actual target values.

- **Common Loss Functions:**
  - **`nn.MSELoss` (Mean Squared Error):** For regression tasks. `loss = (1/N) * sum((y_true - y_pred)^2)`.
  - **`nn.CrossEntropyLoss`:** For multi-class classification. It conveniently combines `nn.LogSoftmax` and `nn.NLLLoss`. Expects raw logits as model output.
  - **`nn.BCELoss` (Binary Cross-Entropy Loss):** For binary classification. Expects model output to be probabilities (after a Sigmoid activation).
  - **`nn.BCEWithLogitsLoss`:** For binary classification. More numerically stable than `nn.BCELoss` as it combines Sigmoid and BCE. Expects raw logits.

```python
# Example Loss Functions
loss_mse = nn.MSELoss()
loss_ce = nn.CrossEntropyLoss()
loss_bce_logits = nn.BCEWithLogitsLoss()

# For MSE (Regression)
predictions_reg = torch.randn(5, 1) # 5 samples, 1 output value
targets_reg = torch.randn(5, 1)
mse = loss_mse(predictions_reg, targets_reg)
print(f"MSE Loss: {mse.item()}")

# For CrossEntropy (Multi-class classification)
predictions_mc = torch.randn(5, 3) # 5 samples, 3 classes (logits)
targets_mc = torch.tensor([0, 1, 2, 0, 1]) # True class indices
ce = loss_ce(predictions_mc, targets_mc)
print(f"CrossEntropy Loss: {ce.item()}")

# For BCEWithLogits (Binary classification)
predictions_bc = torch.randn(5, 1) # 5 samples, 1 output logit
targets_bc = torch.rand(5, 1)      # True probabilities (or 0s and 1s)
bce_wl = loss_bce_logits(predictions_bc, targets_bc)
print(f"BCEWithLogits Loss: {bce_wl.item()}")
```

## Optimizers: How Neural Networks Learn

Optimizers implement algorithms to update the model's weights based on the gradients computed during backpropagation, aiming to minimize the loss function.

- **Gradient Descent:** Iteratively moves in the direction opposite to the gradient of the loss function.
- **Stochastic Gradient Descent (SGD):** Uses a single training example or a small batch to compute the gradient and update weights, making it faster and often able to escape local minima.
  `optimizer = torch.optim.SGD(model.parameters(), lr=0.01)`
- **SGD with Momentum:** Adds a fraction of the previous update vector to the current one, helping accelerate SGD in the relevant direction and dampening oscillations.
  `optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)`
- **Adam (Adaptive Moment Estimation):** An adaptive learning rate optimization algorithm that computes individual learning rates for different parameters. Often a good default choice.
  `optimizer = torch.optim.Adam(model.parameters(), lr=0.001)`
- **Learning Rate (lr):** A crucial hyperparameter that controls the step size during weight updates.

## The Training Loop: Forward and Backward Propagation

The core process of training a neural network involves repeatedly feeding data to the model and adjusting its weights.

- **Forward Propagation:** Input data is passed through the network to generate predictions. The loss function then compares these predictions to the true targets to compute the loss.
  `outputs = model(inputs)`
  `loss = criterion(outputs, labels)`
- **Backward Propagation (Backpropagation):** The `loss.backward()` call computes the gradients of the loss with respect to all model parameters (weights and biases) that have `requires_grad=True`.
- **Optimizer Step:** The `optimizer.step()` call updates the model parameters using the computed gradients and the optimizer's update rule (e.g., SGD, Adam).
- **Zeroing Gradients:** Before each `loss.backward()` call in a new iteration, it's crucial to clear old gradients using `optimizer.zero_grad()`. Otherwise, gradients would accumulate across iterations.
- **Epochs and Batches:**
  - **Epoch:** One complete pass through the entire training dataset.
  - **Batch:** A subset of the training dataset processed in one iteration of the training loop.

## Building and Training Your First Neural Network in PyTorch

This section will be detailed in the accompanying Python script (`neural_networks_fundamentals.py`) and Jupyter Notebook, showing a complete end-to-end example.

**Conceptual Steps:**
1.  **Prepare Data:** Load and preprocess your dataset. PyTorch uses `Dataset` and `DataLoader` classes.
2.  **Define Model:** Create your neural network class inheriting from `nn.Module`.
3.  **Define Loss and Optimizer:** Instantiate your chosen loss function and optimizer, linking the optimizer to your model's parameters.
4.  **Training Loop:**
    ```python
    # num_epochs = ...
    # for epoch in range(num_epochs):
    #     for i, (inputs, labels) in enumerate(train_loader):
    #         # Move tensors to the configured device (CPU/GPU)
    #         inputs = inputs.to(device)
    #         labels = labels.to(device)
    #
    #         # Forward pass
    #         outputs = model(inputs)
    #         loss = criterion(outputs, labels)
    #
    #         # Backward and optimize
    #         optimizer.zero_grad()
    #         loss.backward()
    #         optimizer.step()
    #
    #         if (i+1) % 100 == 0:
    #             print (f'Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{len(train_loader)}], Loss: {loss.item():.4f}')
    ```
5.  **Evaluate Model:** Assess performance on a separate test dataset.

## Running the Tutorial

To run the Python script associated with this tutorial:
```bash
python neural_networks_fundamentals.py
```
Alternatively, you can follow along with the Jupyter notebook `neural_networks_fundamentals.ipynb` for an interactive experience. We recommend manually creating the notebook and copying code from the script if direct creation fails.

## Prerequisites
- Python 3.7+
- PyTorch 1.10+
- NumPy
- Matplotlib (for visualization)
- Scikit-learn (for generating sample data or splitting)

## Related Tutorials
1. [PyTorch Basics](../01_pytorch_basics/README.md)
2. [Automatic Differentiation](../03_automatic_differentiation/README.md)
3. [Training Neural Networks](../04_training_neural_networks/README.md)