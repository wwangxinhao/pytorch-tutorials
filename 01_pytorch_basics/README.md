# PyTorch Basics

This tutorial covers the fundamental concepts of PyTorch, providing a foundation for deep learning applications.

## Table of Contents
1. [Introduction to PyTorch](#introduction-to-pytorch)
2. [Tensors](#tensors)
3. [Tensor Operations](#tensor-operations)
4. [NumPy Integration](#numpy-integration)
5. [GPU Acceleration](#gpu-acceleration)
6. [Computational Graphs](#computational-graphs)

## Introduction to PyTorch

- Overview of PyTorch as a deep learning framework
- Key features and advantages
- Installation and setup instructions

## Tensors

- Creating tensors
- Tensor types and shapes
- Tensor initialization methods
- Converting between data types

## Tensor Operations

- Element-wise operations
- Matrix operations
- Reshaping and indexing
- Broadcasting

## NumPy Integration

- Converting between PyTorch tensors and NumPy arrays
- Shared memory considerations
- Practical examples of integration

## GPU Acceleration

- Checking GPU availability
- Moving tensors to GPU
- Basic operations on GPU
- Performance considerations

## Computational Graphs

- Understanding dynamic computational graphs
- Graph visualization
- Basic autograd operations

## Running the Tutorial

To run this tutorial:

```bash
python pytorch_basics.py
```

Alternatively, you can follow along with the Jupyter notebook `pytorch_basics.ipynb` for an interactive experience.

## Prerequisites

- Python 3.7+
- PyTorch 1.10+

## Related Tutorials

1. [Neural Networks Fundamentals](../02_neural_networks_fundamentals/README.md)
2. [Automatic Differentiation](../03_automatic_differentiation/README.md)

## Introduction to PyTorch

PyTorch is an open-source machine learning library developed by Facebook's AI Research lab. It provides a flexible and intuitive interface for building and training neural networks. PyTorch is known for its dynamic computational graph, which allows for more flexible model development compared to static graph frameworks.

Key features of PyTorch include:
- Dynamic computational graph (define-by-run)
- Intuitive Python interface
- Seamless integration with Python data science stack
- GPU acceleration
- Rich ecosystem of tools and libraries

## Tensors

Tensors are the fundamental data structure in PyTorch, similar to NumPy arrays but with additional capabilities like GPU acceleration and automatic differentiation. Tensors can represent scalars, vectors, matrices, and higher-dimensional data.

### Creating Tensors

```python
import torch

# Create a tensor from a Python list
x = torch.tensor([1, 2, 3, 4])
print(x)  # tensor([1, 2, 3, 4])

# Create a 2D tensor (matrix)
matrix = torch.tensor([[1, 2], [3, 4]])
print(matrix)
# tensor([[1, 2],
#         [3, 4]])

# Create tensors with specific data types
float_tensor = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float32)
int_tensor = torch.tensor([1, 2, 3], dtype=torch.int64)

# Create tensors with specific shapes
zeros = torch.zeros(3, 4)  # 3x4 tensor of zeros
ones = torch.ones(2, 3)    # 2x3 tensor of ones
rand = torch.rand(2, 2)    # 2x2 tensor with random values from uniform distribution [0, 1)
randn = torch.randn(2, 2)  # 2x2 tensor with random values from standard normal distribution

# Create a tensor with a specific range
range_tensor = torch.arange(0, 10, step=1)  # tensor([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
linspace = torch.linspace(0, 1, steps=5)    # tensor([0.0000, 0.2500, 0.5000, 0.7500, 1.0000])

# Create an identity matrix
eye = torch.eye(3)  # 3x3 identity matrix
```

### Tensor Attributes

```python
x = torch.randn(3, 4, 5)

print(x.shape)      # torch.Size([3, 4, 5])
print(x.size())     # torch.Size([3, 4, 5])
print(x.dim())      # 3 (number of dimensions)
print(x.dtype)      # torch.float32
print(x.device)     # device(type='cpu')
```

### Tensor Indexing and Slicing

```python
x = torch.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]])

# Indexing
print(x[0, 0])      # tensor(1)
print(x[1, 2])      # tensor(6)

# Slicing
print(x[:, 0])      # First column: tensor([1, 4, 7])
print(x[1, :])      # Second row: tensor([4, 5, 6])
print(x[0:2, 1:3])  # Sub-matrix: tensor([[2, 3], [5, 6]])

# Advanced indexing
indices = torch.tensor([0, 2])
print(x[indices])   # tensor([[1, 2, 3], [7, 8, 9]])

# Boolean indexing
mask = x > 5
print(mask)
# tensor([[False, False, False],
#         [False, False,  True],
#         [ True,  True,  True]])
print(x[mask])      # tensor([6, 7, 8, 9])
```

## Tensor Operations

PyTorch provides a wide range of operations for manipulating tensors.

### Arithmetic Operations

```python
a = torch.tensor([1, 2, 3])
b = torch.tensor([4, 5, 6])

# Addition
print(a + b)                # tensor([5, 7, 9])
print(torch.add(a, b))      # tensor([5, 7, 9])

# Subtraction
print(a - b)                # tensor([-3, -3, -3])
print(torch.sub(a, b))      # tensor([-3, -3, -3])

# Multiplication (element-wise)
print(a * b)                # tensor([4, 10, 18])
print(torch.mul(a, b))      # tensor([4, 10, 18])

# Division (element-wise)
print(a / b)                # tensor([0.2500, 0.4000, 0.5000])
print(torch.div(a, b))      # tensor([0.2500, 0.4000, 0.5000])

# In-place operations (modifies the tensor)
a.add_(b)                   # a becomes tensor([5, 7, 9])
```

### Matrix Operations

```python
a = torch.tensor([[1, 2], [3, 4]])
b = torch.tensor([[5, 6], [7, 8]])

# Matrix multiplication
print(torch.matmul(a, b))
# tensor([[19, 22],
#         [43, 50]])

print(a @ b)  # @ operator for matrix multiplication
# tensor([[19, 22],
#         [43, 50]])

# Element-wise multiplication
print(a * b)
# tensor([[ 5, 12],
#         [21, 32]])

# Transpose
print(a.t())
# tensor([[1, 3],
#         [2, 4]])

# Determinant
print(torch.det(a))  # tensor(-2.)

# Inverse
print(torch.inverse(a))
# tensor([[-2.0000,  1.0000],
#         [ 1.5000, -0.5000]])
```

### Reduction Operations

```python
x = torch.tensor([[1, 2, 3], [4, 5, 6]])

# Sum
print(torch.sum(x))         # tensor(21)
print(x.sum())              # tensor(21)
print(x.sum(dim=0))         # Sum along rows: tensor([5, 7, 9])
print(x.sum(dim=1))         # Sum along columns: tensor([6, 15])

# Mean
print(torch.mean(x.float()))  # tensor(3.5000)
print(x.float().mean())       # tensor(3.5000)

# Max and Min
print(torch.max(x))         # tensor(6)
print(x.max())              # tensor(6)
print(x.max(dim=0))         # Max along rows: (values=tensor([4, 5, 6]), indices=tensor([1, 1, 1]))
print(x.min())              # tensor(1)

# Product
print(torch.prod(x))        # tensor(720)
```

### Reshaping Operations

```python
x = torch.tensor([[1, 2, 3], [4, 5, 6]])

# Reshape
print(x.reshape(3, 2))
# tensor([[1, 2],
#         [3, 4],
#         [5, 6]])

# View (shares the same data with the original tensor)
print(x.view(6, 1))
# tensor([[1],
#         [2],
#         [3],
#         [4],
#         [5],
#         [6]])

# Flatten
print(x.flatten())          # tensor([1, 2, 3, 4, 5, 6])

# Permute dimensions
y = torch.tensor([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])  # Shape: (2, 2, 2)
print(y.permute(2, 0, 1))   # Permute to shape (2, 2, 2)

# Squeeze and Unsqueeze
z = torch.tensor([[[1], [2]]])  # Shape: (1, 2, 1)
print(z.squeeze())          # Remove dimensions of size 1: tensor([1, 2])
print(z.squeeze(0))         # Remove dimension 0 if it's size 1: tensor([[1], [2]])
print(torch.unsqueeze(x, 0))  # Add dimension at position 0: shape becomes (1, 2, 3)
```

## NumPy Integration

PyTorch provides seamless integration with NumPy, allowing you to convert between PyTorch tensors and NumPy arrays.

```python
import numpy as np

# Convert NumPy array to PyTorch tensor
np_array = np.array([1, 2, 3])
tensor = torch.from_numpy(np_array)
print(tensor)  # tensor([1, 2, 3])

# Convert PyTorch tensor to NumPy array
tensor = torch.tensor([4, 5, 6])
np_array = tensor.numpy()
print(np_array)  # array([4, 5, 6])

# Note: If the tensor is on CPU, the tensor and the NumPy array share the same memory
# Changes to one will affect the other
np_array = np.array([1, 2, 3])
tensor = torch.from_numpy(np_array)
np_array[0] = 5
print(tensor)  # tensor([5, 2, 3])

# This doesn't work for tensors on GPU
```

## GPU Acceleration

One of the key features of PyTorch is its ability to leverage GPU acceleration for faster computations.

```python
# Check if CUDA (NVIDIA GPU) is available
print(torch.cuda.is_available())  # True if CUDA is available

# Create a tensor on GPU
if torch.cuda.is_available():
    device = torch.device("cuda")
    x = torch.tensor([1, 2, 3], device=device)
    # or
    y = torch.tensor([4, 5, 6]).to(device)
    
    # Move tensor back to CPU
    z = y.cpu()
else:
    device = torch.device("cpu")
    x = torch.tensor([1, 2, 3])  # Default is CPU

# Check which device a tensor is on
print(x.device)  # device(type='cuda') or device(type='cpu')

# Perform operations on GPU
if torch.cuda.is_available():
    a = torch.tensor([1, 2, 3], device=device)
    b = torch.tensor([4, 5, 6], device=device)
    c = a + b  # Operation happens on GPU
    print(c)   # tensor([5, 7, 9], device='cuda:0')
```

## Computational Graphs

PyTorch uses a dynamic computational graph, which means the graph is built on-the-fly as operations are executed. This is different from static graph frameworks where the graph is defined before execution.

```python
# Create tensors with requires_grad=True to track operations
x = torch.tensor(2.0, requires_grad=True)
y = torch.tensor(3.0, requires_grad=True)

# Build a computational graph
z = x**2 + y**3

# Compute gradients
z.backward()

# Access gradients
print(x.grad)  # tensor(4.) (dz/dx = 2*x = 2*2 = 4)
print(y.grad)  # tensor(27.) (dz/dy = 3*y^2 = 3*3^2 = 27)

# Detach a tensor from the graph
a = x.detach()  # Creates a new tensor that shares data but doesn't require gradients
```

### Gradient Accumulation

By default, PyTorch accumulates gradients when `backward()` is called multiple times.

```python
# Reset gradients
x.grad.zero_()
y.grad.zero_()

# Compute gradients multiple times
z = x**2 + y**3
z.backward()
print(x.grad)  # tensor(4.)

z = x**2 + y**3
z.backward()
print(x.grad)  # tensor(8.) (gradients are accumulated)

# To avoid accumulation, reset gradients before each backward pass
x.grad.zero_()
y.grad.zero_()
```

## Conclusion

This tutorial covered the basics of PyTorch, including tensors, operations, NumPy integration, GPU acceleration, and computational graphs. These concepts form the foundation for building and training neural networks with PyTorch.

In the next tutorial, we'll explore automatic differentiation and optimization in more detail.