"""
Tutorial 13: Custom Extensions (C++ and CUDA)
============================================

This tutorial demonstrates how to create custom C++ and CUDA extensions
for PyTorch to achieve better performance for specialized operations.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import time
import os
import sys
from torch.utils.cpp_extension import load_inline

# First, let's understand why we might need custom extensions
print("Why Custom Extensions?")
print("=" * 50)
print("1. Performance: C++/CUDA can be much faster than Python")
print("2. Memory efficiency: Better control over memory allocation")
print("3. Novel operations: Implement operations not available in PyTorch")
print("4. Hardware optimization: Leverage specific hardware features")
print()

# Example 1: Simple C++ Extension (Inline JIT Compilation)
print("Example 1: Simple C++ Extension")
print("-" * 30)

# C++ source code for a custom ReLU implementation
cpp_source = '''
#include <torch/extension.h>
#include <vector>

// Forward pass
torch::Tensor custom_relu_forward(torch::Tensor input) {
    auto output = torch::zeros_like(input);
    output = torch::where(input > 0, input, output);
    return output;
}

// Backward pass
torch::Tensor custom_relu_backward(torch::Tensor grad_output, torch::Tensor input) {
    auto grad_input = torch::zeros_like(grad_output);
    grad_input = torch::where(input > 0, grad_output, grad_input);
    return grad_input;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &custom_relu_forward, "Custom ReLU forward");
    m.def("backward", &custom_relu_backward, "Custom ReLU backward");
}
'''

# Load the extension
custom_relu_cpp = load_inline(
    name='custom_relu_cpp',
    cpp_sources=[cpp_source],
    functions=['forward', 'backward'],
    verbose=True,
    build_directory='./cpp_build'
)

# Create a custom autograd Function
class CustomReLUFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        return custom_relu_cpp.forward(input)
    
    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        return custom_relu_cpp.backward(grad_output, input)

# Wrap it in a module
class CustomReLU(nn.Module):
    def forward(self, input):
        return CustomReLUFunction.apply(input)

# Test the custom ReLU
x = torch.randn(10, 10, requires_grad=True)
custom_relu = CustomReLU()
y = custom_relu(x)
loss = y.sum()
loss.backward()

print(f"Input shape: {x.shape}")
print(f"Output shape: {y.shape}")
print(f"Gradient computed: {x.grad is not None}")
print()

# Example 2: CUDA Extension for Matrix Operations
print("Example 2: CUDA Extension")
print("-" * 30)

# Check if CUDA is available
if torch.cuda.is_available():
    # CUDA kernel source code
    cuda_source = '''
    #include <torch/extension.h>
    #include <cuda.h>
    #include <cuda_runtime.h>
    #include <vector>

    template <typename scalar_t>
    __global__ void custom_matmul_kernel(
        const scalar_t* __restrict__ a,
        const scalar_t* __restrict__ b,
        scalar_t* __restrict__ c,
        int m, int n, int k) {
        
        int row = blockIdx.y * blockDim.y + threadIdx.y;
        int col = blockIdx.x * blockDim.x + threadIdx.x;
        
        if (row < m && col < n) {
            scalar_t sum = 0;
            for (int i = 0; i < k; i++) {
                sum += a[row * k + i] * b[i * n + col];
            }
            c[row * n + col] = sum;
        }
    }

    torch::Tensor custom_matmul_cuda(torch::Tensor a, torch::Tensor b) {
        const int m = a.size(0);
        const int k = a.size(1);
        const int n = b.size(1);
        
        auto c = torch::zeros({m, n}, a.options());
        
        const dim3 threads(16, 16);
        const dim3 blocks((n + threads.x - 1) / threads.x,
                         (m + threads.y - 1) / threads.y);
        
        AT_DISPATCH_FLOATING_TYPES(a.type(), "custom_matmul_cuda", ([&] {
            custom_matmul_kernel<scalar_t><<<blocks, threads>>>(
                a.data_ptr<scalar_t>(),
                b.data_ptr<scalar_t>(),
                c.data_ptr<scalar_t>(),
                m, n, k
            );
        }));
        
        return c;
    }
    '''
    
    cpp_source_cuda = '''
    #include <torch/extension.h>
    
    torch::Tensor custom_matmul_cuda(torch::Tensor a, torch::Tensor b);
    
    torch::Tensor custom_matmul(torch::Tensor a, torch::Tensor b) {
        // Check inputs
        TORCH_CHECK(a.dim() == 2, "Matrix A must be 2D");
        TORCH_CHECK(b.dim() == 2, "Matrix B must be 2D");
        TORCH_CHECK(a.size(1) == b.size(0), "Matrix dimensions must match for multiplication");
        
        if (a.is_cuda()) {
            return custom_matmul_cuda(a, b);
        } else {
            // CPU implementation
            return torch::matmul(a, b);
        }
    }
    
    PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
        m.def("matmul", &custom_matmul, "Custom matrix multiplication");
    }
    '''
    
    # Note: CUDA compilation requires nvcc and proper setup
    print("CUDA extension example (pseudo-code for demonstration)")
    print("In practice, you would compile this with setuptools or torch.utils.cpp_extension")
else:
    print("CUDA not available, skipping CUDA example")
print()

# Example 3: Custom Linear Layer with Fused Operations
print("Example 3: Fused Linear Layer")
print("-" * 30)

# C++ code for fused linear layer (bias + activation)
fused_cpp_source = '''
#include <torch/extension.h>
#include <vector>

torch::Tensor fused_linear_relu_forward(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias) {
    
    // Perform linear transformation
    auto output = torch::matmul(input, weight.t());
    
    // Add bias and apply ReLU in one pass
    output = torch::clamp_min(output + bias, 0);
    
    return output;
}

std::vector<torch::Tensor> fused_linear_relu_backward(
    torch::Tensor grad_output,
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor output) {
    
    // ReLU backward
    auto relu_grad = torch::where(output > 0, grad_output, torch::zeros_like(grad_output));
    
    // Linear backward
    auto grad_input = torch::matmul(relu_grad, weight);
    auto grad_weight = torch::matmul(relu_grad.t(), input);
    auto grad_bias = relu_grad.sum(0);
    
    return {grad_input, grad_weight, grad_bias};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &fused_linear_relu_forward, "Fused Linear-ReLU forward");
    m.def("backward", &fused_linear_relu_backward, "Fused Linear-ReLU backward");
}
'''

# Load the fused operation
fused_linear_relu = load_inline(
    name='fused_linear_relu',
    cpp_sources=[fused_cpp_source],
    functions=['forward', 'backward'],
    verbose=True,
    build_directory='./cpp_build'
)

class FusedLinearReLUFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, weight, bias):
        output = fused_linear_relu.forward(input, weight, bias)
        ctx.save_for_backward(input, weight, output)
        return output
    
    @staticmethod
    def backward(ctx, grad_output):
        input, weight, output = ctx.saved_tensors
        grad_input, grad_weight, grad_bias = fused_linear_relu.backward(
            grad_output, input, weight, output
        )
        return grad_input, grad_weight, grad_bias

class FusedLinearReLU(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(out_features, in_features))
        self.bias = nn.Parameter(torch.zeros(out_features))
        
    def forward(self, input):
        return FusedLinearReLUFunction.apply(input, self.weight, self.bias)

# Test the fused layer
fused_layer = FusedLinearReLU(100, 50)
x = torch.randn(32, 100)
y = fused_layer(x)
print(f"Fused layer output shape: {y.shape}")
print()

# Example 4: Custom Optimizer in C++
print("Example 4: Custom Optimizer")
print("-" * 30)

custom_optimizer_source = '''
#include <torch/extension.h>
#include <vector>

void custom_sgd_step(
    torch::Tensor param,
    torch::Tensor grad,
    torch::Tensor momentum_buffer,
    float lr,
    float momentum,
    float weight_decay) {
    
    if (weight_decay != 0) {
        grad = grad + weight_decay * param;
    }
    
    if (momentum != 0) {
        momentum_buffer.mul_(momentum).add_(grad);
        param.add_(momentum_buffer, -lr);
    } else {
        param.add_(grad, -lr);
    }
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("step", &custom_sgd_step, "Custom SGD step");
}
'''

custom_sgd = load_inline(
    name='custom_sgd',
    cpp_sources=[custom_optimizer_source],
    functions=['step'],
    verbose=True,
    build_directory='./cpp_build'
)

class CustomSGD:
    def __init__(self, params, lr=0.01, momentum=0.9, weight_decay=0):
        self.params = list(params)
        self.lr = lr
        self.momentum = momentum
        self.weight_decay = weight_decay
        self.momentum_buffers = {}
        
        for p in self.params:
            self.momentum_buffers[p] = torch.zeros_like(p)
    
    def step(self):
        for p in self.params:
            if p.grad is not None:
                custom_sgd.step(
                    p.data,
                    p.grad.data,
                    self.momentum_buffers[p],
                    self.lr,
                    self.momentum,
                    self.weight_decay
                )
    
    def zero_grad(self):
        for p in self.params:
            if p.grad is not None:
                p.grad.zero_()

# Example 5: Performance Comparison
print("Example 5: Performance Comparison")
print("-" * 30)

def benchmark_operation(name, func, *args, num_runs=1000):
    # Warmup
    for _ in range(10):
        func(*args)
    
    # Benchmark
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    
    start_time = time.time()
    for _ in range(num_runs):
        result = func(*args)
    
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    
    end_time = time.time()
    avg_time = (end_time - start_time) / num_runs * 1000  # Convert to ms
    
    return avg_time, result

# Compare custom ReLU with PyTorch ReLU
x = torch.randn(1000, 1000)
pytorch_relu = nn.ReLU()
custom_relu = CustomReLU()

pytorch_time, _ = benchmark_operation("PyTorch ReLU", pytorch_relu, x)
custom_time, _ = benchmark_operation("Custom ReLU", custom_relu, x)

print(f"PyTorch ReLU: {pytorch_time:.4f} ms")
print(f"Custom ReLU: {custom_time:.4f} ms")
print(f"Speedup: {pytorch_time/custom_time:.2f}x")
print()

# Example 6: Building Extensions with setuptools
print("Example 6: Building with setuptools")
print("-" * 30)

setup_py_content = '''
from setuptools import setup, Extension
from torch.utils import cpp_extension

setup(
    name='custom_ops',
    ext_modules=[
        cpp_extension.CppExtension(
            'custom_ops',
            ['custom_ops.cpp'],
            extra_compile_args=['-O3']
        ),
        cpp_extension.CUDAExtension(
            'custom_cuda_ops',
            ['custom_cuda_ops.cpp', 'custom_cuda_ops_kernel.cu'],
            extra_compile_args={'cxx': ['-O3'],
                              'nvcc': ['-O3', '--use_fast_math']}
        ) if torch.cuda.is_available() else None
    ],
    cmdclass={
        'build_ext': cpp_extension.BuildExtension
    }
)
'''

print("Example setup.py for building extensions:")
print(setup_py_content)
print()

# Example 7: Memory Management in Extensions
print("Example 7: Memory Management")
print("-" * 30)

memory_cpp_source = '''
#include <torch/extension.h>
#include <vector>

// Efficient memory pooling example
class MemoryPool {
private:
    std::vector<torch::Tensor> pool;
    std::vector<bool> in_use;
    
public:
    torch::Tensor allocate(std::vector<int64_t> shape, torch::TensorOptions options) {
        // Try to find a suitable tensor in the pool
        for (size_t i = 0; i < pool.size(); i++) {
            if (!in_use[i] && pool[i].sizes() == shape && pool[i].options() == options) {
                in_use[i] = true;
                return pool[i];
            }
        }
        
        // Allocate new tensor
        auto tensor = torch::empty(shape, options);
        pool.push_back(tensor);
        in_use.push_back(true);
        return tensor;
    }
    
    void release(torch::Tensor tensor) {
        for (size_t i = 0; i < pool.size(); i++) {
            if (pool[i].data_ptr() == tensor.data_ptr()) {
                in_use[i] = false;
                break;
            }
        }
    }
};

// Global memory pool
MemoryPool global_pool;

torch::Tensor pooled_operation(torch::Tensor input) {
    auto shape = input.sizes().vec();
    auto output = global_pool.allocate(shape, input.options());
    
    // Perform operation
    output.copy_(input);
    output.mul_(2.0);
    
    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("pooled_operation", &pooled_operation, "Operation with memory pooling");
}
'''

print("Memory pooling example shown above")
print("This technique can significantly reduce memory allocation overhead")
print()

# Best Practices and Tips
print("Best Practices for Custom Extensions")
print("=" * 50)
print("1. Profile First: Ensure the operation is actually a bottleneck")
print("2. Use Existing Ops: Check if PyTorch already has what you need")
print("3. Memory Layout: Ensure tensors are contiguous when needed")
print("4. Error Handling: Use TORCH_CHECK for input validation")
print("5. Gradient Testing: Always verify gradients with gradcheck")
print("6. Documentation: Document tensor shapes and assumptions")
print("7. Platform Support: Test on different platforms and CUDA versions")
print()

# Debugging Tips
print("Debugging Custom Extensions")
print("-" * 30)
print("1. Use print statements in C++ (std::cout)")
print("2. Enable verbose mode in load_inline")
print("3. Use cuda-gdb for CUDA kernels")
print("4. Check tensor continuity with .is_contiguous()")
print("5. Verify shapes and strides match expectations")
print("6. Use torch.autograd.gradcheck for gradient verification")
print()

# Summary
print("Summary")
print("=" * 50)
print("Custom extensions allow you to:")
print("- Achieve better performance for specialized operations")
print("- Implement novel algorithms not available in PyTorch")
print("- Leverage hardware-specific optimizations")
print("- Create memory-efficient implementations")
print("\nRemember: Only use custom extensions when necessary!")
print("PyTorch's built-in operations are highly optimized and sufficient for most use cases.")