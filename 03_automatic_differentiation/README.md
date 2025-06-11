# Automatic Differentiation with PyTorch Autograd

This tutorial provides a detailed explanation of PyTorch's automatic differentiation system, known as Autograd. Understanding Autograd is crucial for training neural networks as it automates the computation of gradients.

## Table of Contents
1. [Introduction to Automatic Differentiation](#introduction-to-automatic-differentiation)
   - What is Differentiation?
   - Manual vs. Symbolic vs. Automatic Differentiation
   - Why Automatic Differentiation for Deep Learning?
2. [PyTorch Autograd: The Basics](#pytorch-autograd-the-basics)
   - Tensors and `requires_grad`
   - The `grad_fn` (Gradient Function)
   - Computing Gradients: `backward()`
   - Accessing Gradients: `.grad` attribute
3. [The Computational Graph](#the-computational-graph)
   - Dynamic Computational Graphs in PyTorch
   - How Autograd Constructs the Graph
   - Nodes and Edges: Tensors and Operations
   - Leaf Nodes vs. Non-Leaf Nodes
4. [Gradient Accumulation](#gradient-accumulation)
   - How Gradients Accumulate by Default
   - Zeroing Gradients: `optimizer.zero_grad()` or `tensor.grad.zero_()`
   - Use Cases for Gradient Accumulation (e.g., simulating larger batch sizes)
5. [Excluding Tensors from Autograd (`torch.no_grad()`, `detach()`)](#excluding-tensors-from-autograd-torchnograd-detach)
   - `torch.no_grad()`: Context manager to disable gradient computation.
   - `.detach()`: Creates a new tensor that shares the same data but is detached from the computation history.
   - Use cases: Inference, freezing layers, modifying tensors without tracking.
6. [Gradients of Non-Scalar Outputs (Vector-Jacobian Product)](#gradients-of-non-scalar-outputs-vector-jacobian-product)
   - `backward()` on a non-scalar tensor requires a `gradient` argument.
   - Understanding the Vector-Jacobian Product (JVP) concept.
   - Practical examples.
7. [Higher-Order Derivatives](#higher-order-derivatives)
   - Computing gradients of gradients.
   - Using `torch.autograd.grad()` for more control.
   - `create_graph=True` in `backward()` or `torch.autograd.grad()`.
8. [In-place Operations and Autograd](#in-place-operations-and-autograd)
   - Potential issues with in-place operations (ending with `_`).
   - Autograd's need for original values for gradient computation.
   - When they might be problematic and when they are safe.
9. [Custom Autograd Functions (`torch.autograd.Function`)](#custom-autograd-functions-torchautogradfunction)
   - When to use: Implementing novel operations, non-PyTorch computations.
   - Subclassing `torch.autograd.Function`.
   - Defining `forward()` and `backward()` static methods.
   - `ctx` (context) object for saving tensors for backward pass.
   - Example: A custom ReLU or a simple custom operation.
10. [Practical Considerations and Tips](#practical-considerations-and-tips)
    - Checking if a tensor requires gradients: `tensor.requires_grad`.
    - Checking if a tensor is a leaf tensor: `tensor.is_leaf`.
    - Memory usage: Autograd stores intermediate values for backward pass.
    - `retain_graph=True` in `backward()`: When needed and its implications.

## Introduction to Automatic Differentiation

- **What is Differentiation?** Finding the rate of change of a function with respect to its input variables (i.e., its derivatives or gradients).
- **Manual Differentiation:** Deriving gradients by hand. Tedious and error-prone for complex functions like neural networks.
- **Symbolic Differentiation:** Using computer algebra systems to manipulate mathematical expressions and find derivatives (e.g., Wolfram Alpha, SymPy). Can lead to complex and inefficient expressions.
- **Automatic Differentiation (AD):** A set of techniques to numerically evaluate the derivative of a function specified by a computer program. AD decomposes the computation into a sequence of elementary operations (addition, multiplication, sin, exp, etc.) and applies the chain rule repeatedly.
  - **Reverse Mode AD:** What PyTorch uses. Computes gradients by traversing the computational graph backward from output to input. Efficient for functions with many inputs and few outputs (like neural network loss functions).
- **Why AD for Deep Learning?** Neural networks are complex functions with millions of parameters. AD (specifically reverse mode) provides an efficient and accurate way to compute the gradients of the loss function with respect to all these parameters, which is essential for gradient-based optimization (like SGD).

## PyTorch Autograd: The Basics

PyTorch's `autograd` package provides automatic differentiation for all operations on Tensors.

- **Tensors and `requires_grad`:**
  - If a `Tensor` has its `requires_grad` attribute set to `True`, PyTorch tracks all operations on it. This is typically done for learnable parameters (weights, biases) or tensors that are part of a computation leading to a value for which gradients are needed.
  - You can set `requires_grad=True` when creating a tensor or later using `tensor.requires_grad_(True)` (in-place).
- **The `grad_fn` (Gradient Function):**
  - When an operation is performed on tensors that require gradients, the resulting tensor will have a `grad_fn` attribute. This function knows how to compute the gradient of that operation during the backward pass.
  - Leaf tensors (created by the user, not as a result of an operation) with `requires_grad=True` will have `grad_fn=None` initially, but their `.grad` attribute will be populated after `backward()`.
- **Computing Gradients: `backward()`:**
  - To compute gradients, you call `.backward()` on a scalar tensor (e.g., the loss). If the tensor is non-scalar, you need to provide a `gradient` argument (see Section 6).
  - This initiates the backward pass, computing gradients for all tensors in the computational graph that have `requires_grad=True`.
- **Accessing Gradients: `.grad` attribute:**
  - After `loss.backward()` is called, the gradients are accumulated in the `.grad` attribute of the leaf tensors (those for which `requires_grad=True`).

```python
import torch

# Example 1: Basic gradient computation
x = torch.tensor(2.0, requires_grad=True)
y = torch.tensor(3.0, requires_grad=True)
z = x**2 + y**3 # z = 2^2 + 3^3 = 4 + 27 = 31

# Compute gradients
z.backward() # Computes dz/dx and dz/dy

print(f"x: {x}, Gradient dz/dx: {x.grad}") # dz/dx = 2*x = 2*2 = 4
print(f"y: {y}, Gradient dz/dy: {y.grad}") # dz/dy = 3*y^2 = 3*3^2 = 27

# grad_fn example
print(f"z.grad_fn: {z.grad_fn}") # Should show <AddBackward0>
print(f"x.grad_fn: {x.grad_fn}") # Leaf tensor, no grad_fn from previous op
```

## The Computational Graph

- **Dynamic Computational Graphs:** PyTorch builds the computational graph on-the-fly as operations are executed (define-by-run). This allows for more flexibility in model architecture (e.g., using standard Python control flow like loops and conditionals).
- **How Autograd Constructs the Graph:** Each operation on tensors with `requires_grad=True` creates a new node in the graph. Tensors are nodes, and operations (`grad_fn`) are edges that define how to compute gradients.
- **Leaf Nodes:** Tensors created directly by the user (e.g., `torch.tensor(...)`, model parameters). Their gradients are accumulated in `.grad`.
- **Non-Leaf Nodes (Intermediate Tensors):** Tensors resulting from operations. They have a `grad_fn`. By default, their gradients are not saved to save memory, but can be retained using `tensor.retain_grad()`.

## Gradient Accumulation

- **How Gradients Accumulate:** When `backward()` is called multiple times (e.g., in a loop without zeroing gradients), gradients are summed (accumulated) in the `.grad` attribute of leaf tensors.
- **Zeroing Gradients:** It's crucial to zero out gradients before each new backward pass in a typical training loop using `optimizer.zero_grad()` or by manually setting `tensor.grad.zero_()` for each parameter. Otherwise, gradients from previous batches/iterations will interfere.
- **Use Cases for Accumulation:** Deliberate gradient accumulation can be used to simulate a larger effective batch size when GPU memory is limited. You perform several forward/backward passes accumulating gradients and then perform an optimizer step.

```python
x = torch.tensor(1.0, requires_grad=True)
y1 = x * 2
y2 = x * 3

# First backward pass
y1.backward(retain_graph=True) # retain_graph needed if y2.backward() follows on same graph portion
print(f"After y1.backward(), x.grad: {x.grad}") # dy1/dx = 2

# Second backward pass (gradients accumulate)
y2.backward()
print(f"After y2.backward(), x.grad: {x.grad}") # 2 (from y1) + 3 (from y2) = 5

# Zeroing gradients
x.grad.zero_()
print(f"After x.grad.zero_(), x.grad: {x.grad}")
```

## Excluding Tensors from Autograd (`torch.no_grad()`, `detach()`)

- **`torch.no_grad()`:** A context manager that disables gradient computation within its block. Useful for inference (when you don't need gradients) or when modifying model parameters without tracking these changes (e.g., during evaluation).
- **`.detach()`:** Creates a new tensor that shares the same data as the original tensor but is detached from the current computational graph. It won't require gradients, and no operations on it will be tracked. Useful if you need to use a tensor in a computation that shouldn't be part of the gradient calculation, or to copy a tensor without its history.

```python
a = torch.tensor([1.0, 2.0], requires_grad=True)
b = a * 2

with torch.no_grad():
    c = a * 3 # Operation inside no_grad block
    print(f"c.requires_grad inside no_grad: {c.requires_grad}") # False

d = b.detach() # d shares data with b but is detached
print(f"b.requires_grad: {b.requires_grad}") # True
print(f"d.requires_grad: {d.requires_grad}") # False
```

## Gradients of Non-Scalar Outputs (Vector-Jacobian Product)

- If `backward()` is called on a tensor `y` that is not a scalar (e.g., a vector or matrix), PyTorch expects a `gradient` argument. This argument should be a tensor of the same shape as `y` and represents the vector `v` in the vector-Jacobian product `v^T * J`.
- **Vector-Jacobian Product:** Autograd is designed to compute Jacobian-vector products efficiently. If `y = f(x)` and `L` is a scalar loss computed from `y` (i.e., `L = g(y)`), then `dL/dx = (dL/dy) * (dy/dx)`. Here, `dL/dy` is the vector `v` you pass to `y.backward(v)`.
- If you just want the full Jacobian matrix, you'd have to call `backward()` multiple times with one-hot vectors for `gradient`, which is inefficient. `torch.autograd.functional.jacobian` can be used for this if needed.

```python
x = torch.randn(3, requires_grad=True)
y = x * 2       # y is a vector
# y.backward() # This would raise an error

# Provide gradient argument for non-scalar output
# This is equivalent to if we had a scalar loss L = sum(y*v)
# and then called L.backward(). The gradient for x would be 2*v.
v = torch.tensor([0.1, 1.0, 0.001], dtype=torch.float)
y.backward(gradient=v)
print(f"x.grad after y.backward(v): {x.grad}") # Expected: 2*v = [0.2, 2.0, 0.002]
```

## Higher-Order Derivatives

- PyTorch can compute gradients of gradients (and so on).
- **`torch.autograd.grad()`:** A more flexible way to compute gradients. It takes the output tensor(s) and input tensor(s) and returns the gradients of outputs with respect to inputs.
- **`create_graph=True`:** To compute higher-order derivatives, you need to set `create_graph=True` when calling `backward()` or `torch.autograd.grad()`. This tells Autograd to build a computational graph for the backward pass itself, allowing subsequent differentiation.

```python
x = torch.tensor(2.0, requires_grad=True)
y = x**3

# First derivative (dy/dx)
grad_y_x = torch.autograd.grad(outputs=y, inputs=x, create_graph=True)[0]
print(f"dy/dx = 3*x^2 = {grad_y_x}") # 3 * 2^2 = 12

# Second derivative (d^2y/dx^2)
grad2_y_x2 = torch.autograd.grad(outputs=grad_y_x, inputs=x)[0]
print(f"d^2y/dx^2 = 6*x = {grad2_y_x2}") # 6 * 2 = 12
```

## In-place Operations and Autograd

- In-place operations (e.g., `x.add_(1)`, `y.relu_()`) modify tensors directly without creating new ones. This can save memory.
- **Potential Issues:** Autograd needs the original values of tensors involved in the forward pass to compute gradients correctly during the backward pass. If an in-place operation overwrites a value that's needed, it can lead to errors or incorrect gradients.
- PyTorch will often raise an error if an in-place operation that would cause issues is detected (e.g., modifying a leaf variable or a variable needed by `grad_fn`).

## Custom Autograd Functions (`torch.autograd.Function`)

- For operations not natively supported by PyTorch, or if you want to define a custom gradient computation (e.g., for a layer written in C++ or CUDA, or to implement a non-differentiable function with a surrogate gradient).
- **Subclass `torch.autograd.Function`:** Implement `forward()` and `backward()` as static methods.
  - `forward(ctx, input1, input2, ...)`: Performs the operation. `ctx` (context) is used to save tensors or any other objects needed for the backward pass using `ctx.save_for_backward(tensor1, tensor2)`. It must return the output tensor(s).
  - `backward(ctx, grad_output1, grad_output2, ...)`: Computes the gradients of the loss with respect to the inputs of the forward function. It receives the gradients of the loss with respect to the outputs of forward (`grad_output`). It must return as many tensors as there were inputs to `forward`, or `None` for inputs that don't need gradients.

```python
class MyCustomReLU(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input_tensor):
        # ctx is a context object that can be used to stash information
        # for backward computation
        ctx.save_for_backward(input_tensor)
        return input_tensor.clamp(min=0)

    @staticmethod
    def backward(ctx, grad_output):
        # We return as many input gradients as there were arguments.
        # Gradients of non-Tensor arguments to forward must be None.
        input_tensor, = ctx.saved_tensors
        grad_input = grad_output.clone()
        grad_input[input_tensor < 0] = 0
        return grad_input

# Usage:
my_relu_fn = MyCustomReLU.apply # Get the function to use
x = torch.tensor([-1.0, 2.0, -0.5], requires_grad=True)
y = my_relu_fn(x)
print(f"Custom ReLU Output: {y}")
y.backward(torch.tensor([1.0, 1.0, 1.0])) # Example upstream gradients
print(f"Gradients for x after custom ReLU: {x.grad}") # Expected: [0., 1., 0.]
```

## Practical Considerations and Tips
- **`tensor.requires_grad`**: Check if a tensor is tracking history.
- **`tensor.is_leaf`**: Check if a tensor is a leaf node in the graph.
- **Memory Usage**: Autograd stores intermediate activations for the backward pass. For very large models or long sequences, this can lead to high memory usage. Techniques like gradient checkpointing can help.
- **`retain_graph=True`**: Use in `backward()` if you need to perform another backward pass from the same part of the graph. Be mindful of memory implications.

## Running the Tutorial

To run the Python script associated with this tutorial:
```bash
python automatic_differentiation.py
```
We recommend you manually create an `automatic_differentiation.ipynb` notebook and copy the code from the Python script into it for an interactive experience.

## Prerequisites
- Python 3.7+
- PyTorch 1.10+
- NumPy

## Related Tutorials
1. [PyTorch Basics](../01_pytorch_basics/README.md)
2. [Neural Networks Fundamentals](../02_neural_networks_fundamentals/README.md)
3. [Training Neural Networks](../04_training_neural_networks/README.md) 