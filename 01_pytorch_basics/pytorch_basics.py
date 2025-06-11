#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
PyTorch Basics

This script provides an introduction to PyTorch, covering tensors, operations,
and basic computational graphs.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt

# Set random seed for reproducibility
torch.manual_seed(42)

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# -----------------------------------------------------------------------------
# Section 1: Introduction to PyTorch
# -----------------------------------------------------------------------------

def intro_to_pytorch():
    """Introduce basic PyTorch concepts and features."""
    print("Introduction to PyTorch")
    print("-" * 50)
    print("PyTorch is an open-source machine learning library for Python.")
    print("Key features include:")
    print("  - Tensor computation with strong GPU acceleration")
    print("  - Dynamic neural networks")
    print("  - Automatic differentiation for deep learning")
    print(f"PyTorch version: {torch.__version__}")
    if torch.cuda.is_available():
        print(f"CUDA version: {torch.version.cuda}")

# -----------------------------------------------------------------------------
# Section 2: Tensors
# -----------------------------------------------------------------------------

def demonstrate_tensors():
    """Demonstrate tensor creation and properties."""
    print("\nTensors in PyTorch")
    print("-" * 50)
    
    # Creating tensors
    tensor_1d = torch.tensor([1, 2, 3, 4, 5])
    tensor_2d = torch.tensor([[1, 2, 3], [4, 5, 6]])
    print("1D Tensor:", tensor_1d)
    print("2D Tensor:\n", tensor_2d)
    
    # Tensor properties
    print("\nTensor Properties:")
    print(f"Shape of 1D tensor: {tensor_1d.shape}")
    print(f"Shape of 2D tensor: {tensor_2d.shape}")
    print(f"Data type of 1D tensor: {tensor_1d.dtype}")
    print(f"Device of 1D tensor: {tensor_1d.device}")
    
    # Different initialization methods
    zeros_tensor = torch.zeros(3, 3)
    ones_tensor = torch.ones(2, 4)
    random_tensor = torch.randn(2, 3)
    print("\nInitialization Methods:")
    print("Zeros Tensor:\n", zeros_tensor)
    print("Ones Tensor:\n", ones_tensor)
    print("Random Tensor:\n", random_tensor)
    
    # Converting data types
    float_tensor = tensor_1d.float()
    int_tensor = tensor_1d.int()
    print("\nType Conversion:")
    print(f"Original dtype: {tensor_1d.dtype}")
    print(f"Float tensor dtype: {float_tensor.dtype}")
    print(f"Int tensor dtype: {int_tensor.dtype}")

# -----------------------------------------------------------------------------
# Section 3: Tensor Operations
# -----------------------------------------------------------------------------

def demonstrate_tensor_operations():
    """Demonstrate various tensor operations."""
    print("\nTensor Operations")
    print("-" * 50)
    
    # Create sample tensors
    a = torch.tensor([[1, 2], [3, 4]], dtype=torch.float)
    b = torch.tensor([[5, 6], [7, 8]], dtype=torch.float)
    
    # Element-wise operations
    add_result = a + b
    mul_result = a * b
    print("Element-wise Operations:")
    print("Addition:\n", add_result)
    print("Multiplication:\n", mul_result)
    
    # Matrix operations
    matmul_result = torch.matmul(a, b)
    transpose = a.t()
    print("\nMatrix Operations:")
    print("Matrix Multiplication:\n", matmul_result)
    print("Transpose of a:\n", transpose)
    
    # Reshaping
    reshape_result = a.view(4, 1)
    print("\nReshaping:")
    print("Original shape:", a.shape)
    print("Reshaped tensor shape:", reshape_result.shape)
    print("Reshaped tensor:\n", reshape_result)
    
    # Indexing
    element = a[0, 1]
    row = a[0, :]
    print("\nIndexing:")
    print("Element at [0,1]:", element)
    print("First row:", row)
    
    # Broadcasting
    scalar = torch.tensor(2.0)
    broadcast_result = a + scalar
    print("\nBroadcasting:")
    print("Original tensor:\n", a)
    print("After adding scalar 2.0:\n", broadcast_result)

# -----------------------------------------------------------------------------
# Section 4: NumPy Integration
# -----------------------------------------------------------------------------

def demonstrate_numpy_integration():
    """Demonstrate integration between PyTorch tensors and NumPy arrays."""
    print("\nNumPy Integration")
    print("-" * 50)
    
    # Convert NumPy array to tensor
    np_array = np.array([[1, 2], [3, 4]])
    tensor_from_np = torch.from_numpy(np_array)
    print("NumPy array to Tensor:")
    print("NumPy array:\n", np_array)
    print("Tensor:\n", tensor_from_np)
    
    # Convert tensor to NumPy array
    tensor = torch.tensor([[5, 6], [7, 8]])
    np_from_tensor = tensor.numpy()
    print("\nTensor to NumPy array:")
    print("Tensor:\n", tensor)
    print("NumPy array:\n", np_from_tensor)
    
    # Shared memory demonstration
    print("\nShared Memory Demonstration:")
    np_array[0, 0] = 99
    print("Modified NumPy array:\n", np_array)
    print("Tensor (shares memory):\n", tensor_from_np)

# -----------------------------------------------------------------------------
# Section 5: GPU Acceleration
# -----------------------------------------------------------------------------

def demonstrate_gpu_acceleration():
    """Demonstrate GPU usage with PyTorch."""
    print("\nGPU Acceleration")
    print("-" * 50)
    
    if torch.cuda.is_available():
        # Create tensor on CPU
        cpu_tensor = torch.randn(1000, 1000)
        print("CPU tensor device:", cpu_tensor.device)
        
        # Move tensor to GPU
        gpu_tensor = cpu_tensor.to(device)
        print("GPU tensor device:", gpu_tensor.device)
        
        # Perform operation on GPU
        start_time = time.time()
        result_gpu = torch.matmul(gpu_tensor, gpu_tensor)
        gpu_time = time.time() - start_time
        
        # Perform same operation on CPU
        start_time = time.time()
        result_cpu = torch.matmul(cpu_tensor, cpu_tensor)
        cpu_time = time.time() - start_time
        
        print(f"Matrix multiplication time on CPU: {cpu_time:.4f} seconds")
        print(f"Matrix multiplication time on GPU: {gpu_time:.4f} seconds")
        print(f"Speedup: {cpu_time/gpu_time:.2f}x")
    else:
        print("CUDA is not available. GPU demonstration skipped.")
        print("To enable GPU acceleration, install CUDA and cuDNN.")

# -----------------------------------------------------------------------------
# Section 6: Computational Graphs
# -----------------------------------------------------------------------------

def demonstrate_computational_graphs():
    """Demonstrate dynamic computational graphs and autograd."""
    print("\nComputational Graphs")
    print("-" * 50)
    
    # Create tensors with gradient tracking
    x = torch.tensor(2.0, requires_grad=True)
    y = torch.tensor(3.0, requires_grad=True)
    
    # Define a simple computation
    z = x * y + x**2
    print("Forward computation: z = x * y + x^2")
    print(f"x = {x.item()}, y = {y.item()}")
    print(f"z = {z.item()}")
    
    # Compute gradients
    z.backward()
    print("\nGradients:")
    print(f"dz/dx = {x.grad.item()} (should be y + 2x = {y.item() + 2*x.item()})")
    print(f"dz/dy = {y.grad.item()} (should be x = {x.item()})")
    
    # Demonstrate a more complex graph
    a = torch.tensor(1.0, requires_grad=True)
    b = torch.tensor(2.0, requires_grad=True)
    c = a + b
    d = c * a
    e = d + b**2
    print("\nMore complex graph: e = (a + b) * a + b^2")
    e.backward()
    print("Gradients:")
    print(f"de/da = {a.grad.item()}")
    print(f"de/db = {b.grad.item()}")

# -----------------------------------------------------------------------------
# Main function to run all sections
# -----------------------------------------------------------------------------

import time

def main():
    """Main function to run all PyTorch basics tutorial sections."""
    print("=" * 80)
    print("PyTorch Basics Tutorial")
    print("=" * 80)
    
    # Section 1: Introduction
    intro_to_pytorch()
    
    # Section 2: Tensors
    demonstrate_tensors()
    
    # Section 3: Tensor Operations
    demonstrate_tensor_operations()
    
    # Section 4: NumPy Integration
    demonstrate_numpy_integration()
    
    # Section 5: GPU Acceleration
    demonstrate_gpu_acceleration()
    
    # Section 6: Computational Graphs
    demonstrate_computational_graphs()
    
    print("\nTutorial complete!")

if __name__ == '__main__':
    main()