#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Automatic Differentiation with PyTorch Autograd

This script provides a detailed introduction to PyTorch's Autograd system,
covering `requires_grad`, computational graphs, gradient accumulation,
excluding parts from Autograd, vector-Jacobian products, higher-order derivatives,
in-place operations, custom autograd functions, and practical tips.
"""

import torch
import torch.nn as nn # Though not heavily used, good to have for context with nn.Module
import numpy as np
import matplotlib.pyplot as plt # For graph visualization concept
import os

# Set random seed for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Create output directory for plots if it doesn't exist
output_dir = "03_automatic_differentiation_outputs"
os.makedirs(output_dir, exist_ok=True)

# -----------------------------------------------------------------------------
# Section 1: Introduction to Automatic Differentiation
# (Conceptual, covered in detail in README.md)
# -----------------------------------------------------------------------------

def intro_to_automatic_differentiation_concepts():
    """Prints a summary of Automatic Differentiation concepts."""
    print("\nSection 1: Introduction to Automatic Differentiation")
    print("-" * 70)
    print("Key Concepts (Detailed in README.md):")
    print("  - Differentiation: Manual, Symbolic, Automatic (AD). PyTorch uses Reverse Mode AD.")
    print("  - Why AD for Deep Learning: Efficient gradient computation for complex models.")

# -----------------------------------------------------------------------------
# Section 2: PyTorch Autograd: The Basics
# -----------------------------------------------------------------------------

def demonstrate_autograd_basics():
    """Demonstrates requires_grad, grad_fn, backward(), and .grad attribute."""
    print("\nSection 2: PyTorch Autograd: The Basics")
    print("-" * 70)

    # --- Tensors and `requires_grad` ---
    print("\n--- Tensors and `requires_grad` ---")
    # Create a tensor that requires gradient tracking
    x = torch.tensor(2.0, requires_grad=True, device=device)
    print(f"x: {x}, requires_grad: {x.requires_grad}")

    # Create a tensor that does not require gradients (default)
    w = torch.tensor(5.0, device=device)
    print(f"w: {w}, requires_grad: {w.requires_grad}")

    # Operations involving a tensor with requires_grad=True will result in a tensor that also requires_grad
    y = x * w # y will require gradients because x does
    print(f"y = x * w: {y}, requires_grad: {y.requires_grad}")

    # --- The `grad_fn` (Gradient Function) ---
    print("\n--- The `grad_fn` (Gradient Function) ---")
    print(f"y.grad_fn: {y.grad_fn}") # Shows the function that created y (e.g., <MulBackward0>)
    print(f"x.grad_fn: {x.grad_fn}") # x is a leaf tensor created by user, so grad_fn is None
    print(f"w.grad_fn: {w.grad_fn}") # w does not require grad, so grad_fn is None

    # --- Computing Gradients: `backward()` ---
    print("\n--- Computing Gradients: `backward()` ---")
    # Let's define a scalar output for simplicity
    z = y**2 + x # z = (2*5)^2 + 2 = 10^2 + 2 = 100 + 2 = 102
    print(f"z = y**2 + x: {z}, grad_fn: {z.grad_fn}")
    
    # Compute gradients of z with respect to all tensors with requires_grad=True
    # that z depends on (in this case, only x, as w.requires_grad=False)
    z.backward() # dz/dx is computed

    # --- Accessing Gradients: `.grad` attribute ---
    print("\n--- Accessing Gradients: `.grad` attribute ---")
    # dz/dx = d/dx ((x*w)^2 + x) = 2*(x*w)*w + 1 = 2*x*w^2 + 1
    # At x=2, w=5: dz/dx = 2*2*5^2 + 1 = 4*25 + 1 = 100 + 1 = 101
    print(f"Gradient dz/dx: {x.grad}")
    print(f"Gradient dz/dw (w.grad): {w.grad}") # w.grad is None because w.requires_grad was False

    # Another example
    a = torch.tensor(3.0, requires_grad=True, device=device)
    b = a * a # b = 9
    out = b.mean() # For a single element tensor, mean is itself. Scalar output.
    out.backward()
    print(f"a: {a}, Gradient dout/da: {a.grad}") # d(a^2)/da = 2a = 6

# -----------------------------------------------------------------------------
# Section 3: The Computational Graph
# -----------------------------------------------------------------------------

def demonstrate_computational_graph():
    """Illustrates the concept of the dynamic computational graph."""
    print("\nSection 3: The Computational Graph")
    print("-" * 70)
    print("PyTorch uses dynamic computational graphs (define-by-run).")

    a = torch.tensor(2.0, requires_grad=True, device=device)
    b = torch.tensor(4.0, requires_grad=True, device=device)
    print(f"Leaf tensor a: is_leaf={a.is_leaf}, requires_grad={a.requires_grad}, grad_fn={a.grad_fn}")
    print(f"Leaf tensor b: is_leaf={b.is_leaf}, requires_grad={b.requires_grad}, grad_fn={b.grad_fn}")

    c = a + b # c is an intermediate tensor (non-leaf)
    print(f"Intermediate tensor c = a + b: is_leaf={c.is_leaf}, requires_grad={c.requires_grad}, grad_fn={c.grad_fn}")
    
    d = c * 3 # d is another intermediate tensor
    print(f"Intermediate tensor d = c * 3: is_leaf={d.is_leaf}, requires_grad={d.requires_grad}, grad_fn={d.grad_fn}")

    # The graph is: a --(+)--> c --(*)--> d
    #               b --(+)-->
    #               3 --(*)-->

    # Perform backward pass
    d.backward() # Computes dd/da and dd/db
                 # dd/dc = 3
                 # dc/da = 1, dc/db = 1
                 # dd/da = (dd/dc) * (dc/da) = 3 * 1 = 3
                 # dd/db = (dd/dc) * (dc/db) = 3 * 1 = 3

    print(f"Gradient dd/da: {a.grad}")
    print(f"Gradient dd/db: {b.grad}")
    print(f"Gradient dd/dc (c.grad): {c.grad}") # None by default for non-leaf unless retain_grad() is used
    
    # Simple graph visualization concept (manual plotting)
    fig, ax = plt.subplots(figsize=(6,4))
    ax.text(0.1, 0.8, "a (leaf)", bbox=dict(boxstyle='round,pad=0.3', fc='lightblue', ec='blue', lw=1))
    ax.text(0.1, 0.2, "b (leaf)", bbox=dict(boxstyle='round,pad=0.3', fc='lightblue', ec='blue', lw=1))
    ax.text(0.5, 0.5, "c = a + b\n(grad_fn=AddBackward)", ha='center', va='center', bbox=dict(boxstyle='round,pad=0.3', fc='lightgreen', ec='green', lw=1))
    ax.text(0.9, 0.5, "d = c * 3\n(grad_fn=MulBackward)", ha='center', va='center', bbox=dict(boxstyle='round,pad=0.3', fc='lightcoral', ec='red', lw=1))
    ax.annotate("", xy=(0.4, 0.5), xytext=(0.2, 0.8), arrowprops=dict(arrowstyle="->"))
    ax.annotate("", xy=(0.4, 0.5), xytext=(0.2, 0.2), arrowprops=dict(arrowstyle="->"))
    ax.annotate("", xy=(0.75, 0.5), xytext=(0.65, 0.5), arrowprops=dict(arrowstyle="->"))
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis('off'); plt.title("Conceptual Computational Graph")
    plot_path = os.path.join(output_dir, 'conceptual_graph.png')
    plt.savefig(plot_path); plt.close()
    print(f"Conceptual graph plot saved to '{plot_path}'")

# -----------------------------------------------------------------------------
# Section 4: Gradient Accumulation
# -----------------------------------------------------------------------------

def demonstrate_gradient_accumulation():
    """Shows how gradients accumulate and how to zero them."""
    print("\nSection 4: Gradient Accumulation")
    print("-" * 70)

    x = torch.tensor(3.0, requires_grad=True, device=device)
    
    # First computation
    y1 = x * 2 # dy1/dx = 2
    y1.backward() # Gradients are computed and stored in x.grad
    print(f"After first backward (y1=2x), x.grad: {x.grad}")

    # Second computation on the same x
    # If we don't zero x.grad, new gradients will be added to the existing ones.
    y2 = x * 3 # dy2/dx = 3
    y2.backward()
    print(f"After second backward (y2=3x) WITHOUT zeroing, x.grad: {x.grad}") # Expected: 2 + 3 = 5

    # --- Zeroing Gradients ---
    print("\n--- Zeroing Gradients ---")
    # Method 1: Manually zeroing the .grad attribute
    if x.grad is not None:
        x.grad.zero_()
    print(f"After x.grad.zero_(), x.grad: {x.grad}")

    # Recompute for y1 with fresh gradients
    y1_fresh = x * 2
    y1_fresh.backward()
    print(f"After y1_fresh.backward() with zeroed grad, x.grad: {x.grad}")

    # In a training loop, `optimizer.zero_grad()` is typically used before `loss.backward()`
    # model_params = [x] # Simulate model parameters
    # optimizer = torch.optim.SGD([x], lr=0.1) # Dummy optimizer
    # optimizer.zero_grad() # This would achieve the same for all params in optimizer
    # print(f"After optimizer.zero_grad() (conceptual), x.grad: {x.grad}")
    print("Use case: Simulating larger batch sizes by accumulating gradients over mini-batches.")

# -----------------------------------------------------------------------------
# Section 5: Excluding Tensors from Autograd (`torch.no_grad()`, `detach()`)
# -----------------------------------------------------------------------------

def demonstrate_excluding_from_autograd():
    """Illustrates `torch.no_grad()` and `tensor.detach()`."""
    print("\nSection 5: Excluding Tensors from Autograd")
    print("-" * 70)

    # --- `torch.no_grad()` --- 
    print("\n--- `torch.no_grad()` ---")
    a = torch.tensor(5.0, requires_grad=True, device=device)
    print(f"a: {a}, requires_grad: {a.requires_grad}")

    with torch.no_grad():
        b = a * 2 # Operation within torch.no_grad()
        print(f"Inside torch.no_grad(): b = a * 2: {b}, requires_grad: {b.requires_grad}") # b.requires_grad will be False
        print(f"                         b.grad_fn: {b.grad_fn}") # b.grad_fn will be None
    
    # Operations outside the no_grad block will resume tracking if inputs require_grad
    c = a * 3
    print(f"Outside torch.no_grad(): c = a * 3: {c}, requires_grad: {c.requires_grad}, grad_fn: {c.grad_fn}")
    print("`torch.no_grad()` is useful for inference or when modifying parameters without tracking.")

    # --- `.detach()` --- 
    print("\n--- `.detach()` ---")
    p = torch.tensor(4.0, requires_grad=True, device=device)
    q = p * p # q requires grad and has a grad_fn
    print(f"q = p*p: {q}, requires_grad: {q.requires_grad}, grad_fn: {q.grad_fn}")

    r = q.detach() # r shares data with q but is detached from the graph
    print(f"r = q.detach(): {r}, requires_grad: {r.requires_grad}, grad_fn: {r.grad_fn}")

    # Modifying r does not affect q's gradient computation if q is used later
    # However, if q itself is modified in-place, it can cause issues.
    # r[0] = 100.0 # If r was a multi-element tensor, this would modify q's data too.
    # print(f"q after modifying detached r: {q}")
    
    # If we try to backpropagate from r, it won't affect p
    try:
        # r.sum().backward() # This would error if r had no requires_grad and was leaf, or do nothing for p if not leaf.
        # If we make r require grad AFTER detaching, it starts a new history
        r_requiring_grad = q.detach().requires_grad_(True)
        s = r_requiring_grad * 2
        s.sum().backward() # This computes ds/dr_requiring_grad
        print(f"r_requiring_grad.grad: {r_requiring_grad.grad}") # Will have gradient
        print(f"p.grad after s.backward(): {p.grad}") # p.grad will be None or its old value, not affected by s
    except Exception as e:
        print(f"Error demonstrating backward on detached tensor: {e}")
    print("`.detach()` creates a new tensor sharing data but not history, useful for partial graph operations.")

# -----------------------------------------------------------------------------
# Section 6: Gradients of Non-Scalar Outputs (Vector-Jacobian Product)
# -----------------------------------------------------------------------------

def demonstrate_vector_jacobian_product():
    """Explains and shows `backward()` with non-scalar outputs."""
    print("\nSection 6: Gradients of Non-Scalar Outputs (Vector-Jacobian Product)")
    print("-" * 70)
    print("If `backward()` is called on a non-scalar tensor, a `gradient` argument is needed.")

    x_vjp = torch.randn(3, requires_grad=True, device=device) # Input vector
    y_vjp = x_vjp * 2 + 1 # Output vector y = [2x1+1, 2x2+1, 2x3+1]
    # Jacobian J = dy/dx would be a diagonal matrix with 2s on the diagonal.
    print(f"x_vjp: {x_vjp.data}")
    print(f"y_vjp: {y_vjp.data}")

    # If we call y_vjp.backward() directly, it errors: 
    # "grad can be implicitly created only for scalar outputs"
    try:
        y_vjp.backward()
    except RuntimeError as e:
        print(f"Error calling y_vjp.backward() without gradient arg: {e}")

    # We need to provide a `gradient` tensor (vector v in v^T * J)
    # This is often the gradient of the final scalar loss w.r.t y_vjp
    v = torch.tensor([0.1, 1.0, 0.01], device=device) # Example vector
    y_vjp.backward(gradient=v)
    # x_vjp.grad will be v^T * J. Since J is diag(2,2,2), x_vjp.grad = v * 2
    print(f"Provided vector v: {v}")
    print(f"x_vjp.grad (should be v * 2): {x_vjp.grad}") 

    # Use case: If you have a model producing multiple outputs (y_vjp) and your loss
    # is L = sum(y_vjp * some_weights), then dl/dy_vjp = some_weights, which you pass as `gradient`.

# -----------------------------------------------------------------------------
# Section 7: Higher-Order Derivatives
# -----------------------------------------------------------------------------

def demonstrate_higher_order_derivatives():
    """Shows how to compute gradients of gradients."""
    print("\nSection 7: Higher-Order Derivatives")
    print("-" * 70)
    print("Autograd can compute higher-order derivatives (gradients of gradients).")

    x_ho = torch.tensor(2.0, requires_grad=True, device=device)
    y_ho = x_ho**4 # y = x^4
                   # dy/dx = 4x^3
                   # d^2y/dx^2 = 12x^2
                   # d^3y/dx^3 = 24x

    # --- First derivative --- 
    # Method 1: Using y_ho.backward(create_graph=True) and then x_ho.grad
    # (less direct for just getting the grad value, better for graph construction)
    # y_ho.backward(create_graph=True)
    # dy_dx = x_ho.grad.clone() # Must clone if we zero grad later
    # print(f"dy/dx at x={x_ho.item()}: {dy_dx} (Using backward)")
    # x_ho.grad.zero_() # Clean up for next method

    # Method 2: Using torch.autograd.grad()
    dy_dx = torch.autograd.grad(outputs=y_ho, inputs=x_ho, create_graph=True)[0]
    # `create_graph=True` is crucial for enabling further differentiation on dy_dx
    print(f"y = x^4 at x={x_ho.item()} is {y_ho.item()}")
    print(f"dy/dx (4x^3) at x={x_ho.item()}: {dy_dx.item()}")

    # --- Second derivative --- 
    d2y_dx2 = torch.autograd.grad(outputs=dy_dx, inputs=x_ho, create_graph=True)[0]
    print(f"d^2y/dx^2 (12x^2) at x={x_ho.item()}: {d2y_dx2.item()}")

    # --- Third derivative --- 
    d3y_dx3 = torch.autograd.grad(outputs=d2y_dx2, inputs=x_ho, create_graph=False)[0] # No more differentiation needed
    print(f"d^3y/dx^3 (24x) at x={x_ho.item()}: {d3y_dx3.item()}")
    print("Requires `create_graph=True` in `backward()` or `torch.autograd.grad()`.")

# -----------------------------------------------------------------------------
# Section 8: In-place Operations and Autograd
# -----------------------------------------------------------------------------

def demonstrate_inplace_operations():
    """Discusses potential issues with in-place operations."""
    print("\nSection 8: In-place Operations and Autograd")
    print("-" * 70)
    print("In-place operations (e.g., `x.add_(1)`) modify content directly and can cause issues with autograd if not handled carefully.")

    # Example that works (modifying a tensor that is not needed for gradient of another path)
    a = torch.tensor(2.0, requires_grad=True, device=device)
    b = a * 2
    a.add_(1) # In-place operation on 'a'. 'a' is a leaf.
              # This specific in-place op on a leaf that doesn't affect b's grad_fn
              # might be okay, but it's generally risky if 'a' was an intermediate result.
    # If 'a' was an intermediate result required for b's gradient, this could error.
    # b = a * 2 would mean b's grad_fn depends on the original value of a.
    try:
        b.backward() # b was computed using the original 'a'.
        print(f"Gradient db/da after a.add_(1): {a.grad} (Gradient is for original 'a' used in b's computation)") # db/da = 2
    except RuntimeError as e:
        print(f"Error with in-place op (Scenario 1): {e}")
    
    # Example that can cause issues
    x = torch.tensor(1.0, requires_grad=True, device=device)
    y = x.clone() # Create y from x, so y depends on x
    z = y * 2
    # If we modify y in-place, autograd might not have the original y for z.backward()
    # y.add_(1) # This would modify y, which is needed by z's grad_fn (MulBackward)
    # try:
    #     z.backward()
    #     print(f"x.grad after problematic in-place: {x.grad}")
    # except RuntimeError as e:
    #     print(f"Error due to in-place modification of a tensor needed for backward: {e}")
    print("PyTorch often errors if an in-place op overwrites data needed for backward pass.")
    print("It is generally safer to use out-of-place operations (e.g., `a = a + 1`) when unsure.")

# -----------------------------------------------------------------------------
# Section 9: Custom Autograd Functions (`torch.autograd.Function`)
# -----------------------------------------------------------------------------

class MyCubeFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input_tensor):
        """Computes input_tensor^3."""
        # ctx is a context object to save information for backward pass
        ctx.save_for_backward(input_tensor) # Save input_tensor for gradient calculation
        return input_tensor**3

    @staticmethod
    def backward(ctx, grad_output):
        """Computes gradient of input_tensor^3, which is 3 * input_tensor^2."""
        # grad_output is the gradient of the loss w.r.t. the output of this function
        input_tensor, = ctx.saved_tensors # Retrieve saved tensor
        # Gradient of L w.r.t. input = (gradient of L w.r.t. output) * (gradient of output w.r.t. input)
        # Here, gradient of output w.r.t. input is 3 * input_tensor^2
        grad_input = grad_output * 3 * input_tensor**2
        return grad_input # Must return as many gradients as there were inputs to forward

def demonstrate_custom_autograd_function():
    """Shows how to implement and use a custom autograd Function."""
    print("\nSection 9: Custom Autograd Functions (`torch.autograd.Function`)")
    print("-" * 70)
    print("Allows defining custom operations with user-defined forward and backward passes.")

    # Get the function to apply
    custom_cube = MyCubeFunction.apply

    x_custom = torch.tensor(2.0, requires_grad=True, device=device)
    y_custom = custom_cube(x_custom) # y = x^3 = 2^3 = 8
    print(f"x_custom: {x_custom.item()}, y_custom (x^3): {y_custom.item()}")

    # Perform backward pass
    y_custom.backward() # Computes dy/dx = 3x^2
    print(f"Gradient dy_custom/dx_custom (should be 3*2^2 = 12): {x_custom.grad}")

    # Compare with PyTorch's automatic differentiation
    x_auto = torch.tensor(2.0, requires_grad=True, device=device)
    y_auto = x_auto**3
    y_auto.backward()
    print(f"Gradient dy_auto/dx_auto (using PyTorch AD): {x_auto.grad}")
    print("Custom function gradient matches PyTorch's AD.")

# -----------------------------------------------------------------------------
# Section 10: Practical Considerations and Tips
# -----------------------------------------------------------------------------

def practical_considerations_autograd():
    print("\nSection 10: Practical Considerations and Tips")
    print("-" * 70)
    
    x = torch.randn(2, 2, requires_grad=True, device=device)
    y = x + 2 # y requires_grad, is_leaf=False
    z = y * y * 3
    out = z.mean() # Scalar output

    # Checking `requires_grad`
    print(f"x.requires_grad: {x.requires_grad}") # True
    print(f"y.requires_grad: {y.requires_grad}") # True
    intermediate_tensor = torch.tensor([1., 2.], device=device) # Default requires_grad=False
    print(f"intermediate_tensor.requires_grad: {intermediate_tensor.requires_grad}") # False

    # Checking `is_leaf`
    print(f"x.is_leaf: {x.is_leaf}") # True (created by user with requires_grad=True)
    print(f"y.is_leaf: {y.is_leaf}") # False (result of an operation)
    # Model parameters are leaf tensors: model.weight.is_leaf would be True

    # `retain_graph=True`
    # Normally, the graph is freed after backward(). Use retain_graph=True if you need to backprop again.
    out.backward(retain_graph=True) # First backward pass
    print(f"x.grad after first backward: {x.grad.clone().flatten().cpu().numpy()}") # Clone to print before next backward might change it
    
    # If you call backward again on `out` or another part of the same graph without retain_graph=True
    # in the first call, it would error. With retain_graph=True, it works.
    # Gradients will accumulate if not zeroed.
    x.grad.zero_() # Zero gradients before next backward pass
    out.backward() # Second backward pass (possible due to retain_graph=True)
    print(f"x.grad after second backward (with zeroing): {x.grad.flatten().cpu().numpy()}")
    print("`retain_graph=True` is needed for multiple backward passes from the same graph part.")
    print("Memory usage: Autograd stores intermediate values. `del` tensors or use `torch.no_grad()` for inference to save memory.")

# -----------------------------------------------------------------------------
# Main function to run all sections
# -----------------------------------------------------------------------------

def main():
    """Main function to run all Autograd tutorial sections."""
    print("=" * 80)
    print("Automatic Differentiation with PyTorch Autograd Tutorial")
    print("=" * 80)
    
    intro_to_automatic_differentiation_concepts()
    demonstrate_autograd_basics()
    demonstrate_computational_graph()
    demonstrate_gradient_accumulation()
    demonstrate_excluding_from_autograd()
    demonstrate_vector_jacobian_product()
    demonstrate_higher_order_derivatives()
    demonstrate_inplace_operations()
    demonstrate_custom_autograd_function()
    practical_considerations_autograd()
    
    print("\nTutorial complete! Plots are in '03_automatic_differentiation_outputs' directory.")

if __name__ == '__main__':
    main() 