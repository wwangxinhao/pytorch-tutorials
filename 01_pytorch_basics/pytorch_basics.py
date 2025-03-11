#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
PyTorch Basics
--------------
This script demonstrates the fundamental concepts of PyTorch, including tensors,
operations, NumPy integration, GPU acceleration, and computational graphs.
"""

import torch
import numpy as np


def tensor_creation_examples():
    """Examples of creating tensors in PyTorch."""
    print("\n" + "="*50)
    print("TENSOR CREATION EXAMPLES")
    print("="*50)
    
    # Create a tensor from a Python list
    x = torch.tensor([1, 2, 3, 4])
    print("Tensor from list:", x)
    
    # Create a 2D tensor (matrix)
    matrix = torch.tensor([[1, 2], [3, 4]])
    print("\nMatrix:")
    print(matrix)
    
    # Create tensors with specific data types
    float_tensor = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float32)
    int_tensor = torch.tensor([1, 2, 3], dtype=torch.int64)
    print("\nFloat tensor:", float_tensor)
    print("Integer tensor:", int_tensor)
    
    # Create tensors with specific shapes
    zeros = torch.zeros(3, 4)
    ones = torch.ones(2, 3)
    rand = torch.rand(2, 2)
    randn = torch.randn(2, 2)
    
    print("\nZeros tensor:")
    print(zeros)
    print("\nOnes tensor:")
    print(ones)
    print("\nRandom uniform tensor:")
    print(rand)
    print("\nRandom normal tensor:")
    print(randn)
    
    # Create a tensor with a specific range
    range_tensor = torch.arange(0, 10, step=1)
    linspace = torch.linspace(0, 1, steps=5)
    print("\nRange tensor:", range_tensor)
    print("Linspace tensor:", linspace)
    
    # Create an identity matrix
    eye = torch.eye(3)
    print("\nIdentity matrix:")
    print(eye)


def tensor_attributes_examples():
    """Examples of tensor attributes in PyTorch."""
    print("\n" + "="*50)
    print("TENSOR ATTRIBUTES EXAMPLES")
    print("="*50)
    
    x = torch.randn(3, 4, 5)
    
    print("Tensor shape:", x.shape)
    print("Tensor size:", x.size())
    print("Number of dimensions:", x.dim())
    print("Data type:", x.dtype)
    print("Device:", x.device)


def tensor_indexing_examples():
    """Examples of tensor indexing and slicing in PyTorch."""
    print("\n" + "="*50)
    print("TENSOR INDEXING AND SLICING EXAMPLES")
    print("="*50)
    
    x = torch.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    print("Original tensor:")
    print(x)
    
    # Indexing
    print("\nIndexing:")
    print("x[0, 0] =", x[0, 0])
    print("x[1, 2] =", x[1, 2])
    
    # Slicing
    print("\nSlicing:")
    print("First column:")
    print(x[:, 0])
    print("Second row:")
    print(x[1, :])
    print("Sub-matrix (top-right 2x2):")
    print(x[0:2, 1:3])
    
    # Advanced indexing
    indices = torch.tensor([0, 2])
    print("\nAdvanced indexing with indices [0, 2]:")
    print(x[indices])
    
    # Boolean indexing
    mask = x > 5
    print("\nBoolean mask (x > 5):")
    print(mask)
    print("Elements where x > 5:")
    print(x[mask])


def tensor_operations_examples():
    """Examples of tensor operations in PyTorch."""
    print("\n" + "="*50)
    print("TENSOR OPERATIONS EXAMPLES")
    print("="*50)
    
    # Arithmetic operations
    print("Arithmetic Operations:")
    a = torch.tensor([1, 2, 3])
    b = torch.tensor([4, 5, 6])
    
    print("a =", a)
    print("b =", b)
    print("a + b =", a + b)
    print("a - b =", a - b)
    print("a * b =", a * b)
    print("a / b =", a / b)
    
    # Matrix operations
    print("\nMatrix Operations:")
    a = torch.tensor([[1, 2], [3, 4]])
    b = torch.tensor([[5, 6], [7, 8]])
    
    print("Matrix a:")
    print(a)
    print("Matrix b:")
    print(b)
    print("Matrix multiplication (a @ b):")
    print(a @ b)
    print("Element-wise multiplication (a * b):")
    print(a * b)
    print("Transpose of a:")
    print(a.t())
    
    # Reduction operations
    print("\nReduction Operations:")
    x = torch.tensor([[1, 2, 3], [4, 5, 6]])
    print("Tensor x:")
    print(x)
    print("Sum of all elements:", torch.sum(x))
    print("Sum along rows (dim=0):", x.sum(dim=0))
    print("Sum along columns (dim=1):", x.sum(dim=1))
    print("Mean of all elements:", torch.mean(x.float()))
    print("Max of all elements:", torch.max(x))
    
    # Reshaping operations
    print("\nReshaping Operations:")
    print("Original tensor x:")
    print(x)
    print("Reshape to (3, 2):")
    print(x.reshape(3, 2))
    print("View as (6, 1):")
    print(x.view(6, 1))
    print("Flatten:")
    print(x.flatten())


def numpy_integration_examples():
    """Examples of NumPy integration with PyTorch."""
    print("\n" + "="*50)
    print("NUMPY INTEGRATION EXAMPLES")
    print("="*50)
    
    # NumPy array to PyTorch tensor
    np_array = np.array([1, 2, 3])
    tensor = torch.from_numpy(np_array)
    print("NumPy array:", np_array)
    print("PyTorch tensor from NumPy:", tensor)
    
    # PyTorch tensor to NumPy array
    tensor = torch.tensor([4, 5, 6])
    np_array = tensor.numpy()
    print("\nPyTorch tensor:", tensor)
    print("NumPy array from tensor:", np_array)
    
    # Shared memory demonstration
    print("\nShared memory demonstration:")
    np_array = np.array([1, 2, 3])
    tensor = torch.from_numpy(np_array)
    print("Original NumPy array:", np_array)
    print("Original tensor:", tensor)
    
    np_array[0] = 5
    print("Modified NumPy array:", np_array)
    print("Tensor after NumPy modification:", tensor)


def gpu_acceleration_examples():
    """Examples of GPU acceleration in PyTorch."""
    print("\n" + "="*50)
    print("GPU ACCELERATION EXAMPLES")
    print("="*50)
    
    # Check if CUDA is available
    cuda_available = torch.cuda.is_available()
    print("CUDA available:", cuda_available)
    
    # Create tensors on CPU or GPU
    if cuda_available:
        device = torch.device("cuda")
        print("Using CUDA device")
        
        # Create tensor directly on GPU
        x_gpu = torch.tensor([1, 2, 3], device=device)
        print("Tensor created on GPU:", x_gpu)
        
        # Move tensor from CPU to GPU
        x_cpu = torch.tensor([4, 5, 6])
        x_gpu = x_cpu.to(device)
        print("Tensor moved from CPU to GPU:", x_gpu)
        
        # Move tensor back to CPU
        x_cpu_again = x_gpu.cpu()
        print("Tensor moved back to CPU:", x_cpu_again)
    else:
        print("CUDA not available. Using CPU only.")
        device = torch.device("cpu")
        x = torch.tensor([1, 2, 3])
        print("Tensor on CPU:", x)


def computational_graph_examples():
    """Examples of computational graphs in PyTorch."""
    print("\n" + "="*50)
    print("COMPUTATIONAL GRAPH EXAMPLES")
    print("="*50)
    
    # Create tensors with requires_grad=True to track operations
    x = torch.tensor(2.0, requires_grad=True)
    y = torch.tensor(3.0, requires_grad=True)
    
    print("x =", x)
    print("y =", y)
    
    # Build a computational graph
    z = x**2 + y**3
    print("z = x^2 + y^3 =", z)
    
    # Compute gradients
    z.backward()
    
    # Access gradients
    print("Gradient of z with respect to x (dz/dx):", x.grad)
    print("Gradient of z with respect to y (dz/dy):", y.grad)
    
    # Gradient accumulation
    print("\nGradient accumulation:")
    
    # Reset gradients
    x.grad.zero_()
    y.grad.zero_()
    print("After zeroing gradients:")
    print("x.grad =", x.grad)
    print("y.grad =", y.grad)
    
    # Compute gradients multiple times
    z = x**2 + y**3
    z.backward()
    print("After first backward pass:")
    print("x.grad =", x.grad)
    
    z = x**2 + y**3
    z.backward()
    print("After second backward pass (gradients are accumulated):")
    print("x.grad =", x.grad)
    
    # Detach a tensor from the graph
    print("\nDetaching tensors:")
    a = x.detach()
    print("Detached tensor a =", a)
    print("a.requires_grad =", a.requires_grad)


def main():
    """Main function to run all examples."""
    print("PyTorch Basics Tutorial")
    print("Version:", torch.__version__)
    
    tensor_creation_examples()
    tensor_attributes_examples()
    tensor_indexing_examples()
    tensor_operations_examples()
    numpy_integration_examples()
    gpu_acceleration_examples()
    computational_graph_examples()


if __name__ == "__main__":
    main()