"""
Tutorial 17: Model Optimization Techniques
==========================================

This tutorial covers advanced model optimization techniques including
quantization, pruning, knowledge distillation, and model compression.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.quantization import quantize_dynamic, quantize_fx, prepare_fx, convert_fx
import torch.nn.utils.prune as prune
import numpy as np
import time
import copy
from typing import Tuple, List, Dict
import matplotlib.pyplot as plt

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")
print()

# Example 1: Model Quantization
print("Example 1: Model Quantization")
print("=" * 50)

class SimpleModel(nn.Module):
    """Simple model for demonstrating optimization techniques"""
    def __init__(self, input_size=784, hidden_size=256, num_classes=10):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, hidden_size // 2)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(hidden_size // 2, num_classes)
        
    def forward(self, x):
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.fc2(x)
        x = self.relu2(x)
        x = self.fc3(x)
        return x

# Create and evaluate original model
def evaluate_model(model, data_loader=None):
    """Evaluate model size, speed, and accuracy"""
    model.eval()
    
    # Model size
    param_size = 0
    buffer_size = 0
    
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
    
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()
    
    model_size = (param_size + buffer_size) / 1024 / 1024  # MB
    
    # Speed test
    dummy_input = torch.randn(100, 784)
    
    # Warmup
    for _ in range(10):
        _ = model(dummy_input)
    
    # Time inference
    start_time = time.time()
    num_iterations = 100
    with torch.no_grad():
        for _ in range(num_iterations):
            _ = model(dummy_input)
    
    inference_time = (time.time() - start_time) / num_iterations * 1000  # ms
    
    return model_size, inference_time

# Original model
original_model = SimpleModel()
original_size, original_time = evaluate_model(original_model)
print(f"Original Model - Size: {original_size:.2f} MB, Inference: {original_time:.2f} ms")

# Dynamic Quantization
print("\nDynamic Quantization:")
dynamic_quantized_model = quantize_dynamic(
    original_model,
    {nn.Linear},
    dtype=torch.qint8
)
dq_size, dq_time = evaluate_model(dynamic_quantized_model)
print(f"Size: {dq_size:.2f} MB ({original_size/dq_size:.2f}x smaller)")
print(f"Inference: {dq_time:.2f} ms ({original_time/dq_time:.2f}x speedup)")

# Static Quantization (requires calibration)
print("\nStatic Quantization:")

class QuantizedModel(nn.Module):
    """Model prepared for static quantization"""
    def __init__(self, input_size=784, hidden_size=256, num_classes=10):
        super().__init__()
        self.quant = torch.quantization.QuantStub()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, hidden_size // 2)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(hidden_size // 2, num_classes)
        self.dequant = torch.quantization.DeQuantStub()
        
    def forward(self, x):
        x = self.quant(x)
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.fc2(x)
        x = self.relu2(x)
        x = self.fc3(x)
        x = self.dequant(x)
        return x

# Prepare for static quantization
static_model = QuantizedModel()
static_model.eval()

# Set quantization config
static_model.qconfig = torch.quantization.get_default_qconfig('fbgemm')
torch.quantization.prepare(static_model, inplace=True)

# Calibrate with representative data
calibration_data = torch.randn(100, 784)
with torch.no_grad():
    static_model(calibration_data)

# Convert to quantized model
torch.quantization.convert(static_model, inplace=True)

sq_size, sq_time = evaluate_model(static_model)
print(f"Size: {sq_size:.2f} MB ({original_size/sq_size:.2f}x smaller)")
print(f"Inference: {sq_time:.2f} ms ({original_time/sq_time:.2f}x speedup)")
print()

# Example 2: Network Pruning
print("Example 2: Network Pruning")
print("=" * 50)

def count_parameters(model):
    """Count total and non-zero parameters"""
    total_params = 0
    nonzero_params = 0
    
    for name, param in model.named_parameters():
        total_params += param.numel()
        if hasattr(param, 'data'):
            nonzero_params += torch.count_nonzero(param.data).item()
        else:
            nonzero_params += param.numel()
    
    return total_params, nonzero_params

# Create model for pruning
pruning_model = SimpleModel()
total_params, nonzero_params = count_parameters(pruning_model)
print(f"Before pruning - Total: {total_params}, Non-zero: {nonzero_params}")

# Unstructured pruning (fine-grained)
print("\nUnstructured Pruning:")

# Prune 50% of weights in each layer
for name, module in pruning_model.named_modules():
    if isinstance(module, nn.Linear):
        prune.l1_unstructured(module, name='weight', amount=0.5)

total_params, nonzero_params = count_parameters(pruning_model)
sparsity = 1.0 - (nonzero_params / total_params)
print(f"After pruning - Total: {total_params}, Non-zero: {nonzero_params}")
print(f"Sparsity: {sparsity:.2%}")

# Make pruning permanent
for name, module in pruning_model.named_modules():
    if isinstance(module, nn.Linear):
        prune.remove(module, 'weight')

# Structured pruning (coarse-grained)
print("\nStructured Pruning:")

structured_model = SimpleModel()

# Prune entire channels/neurons
for name, module in structured_model.named_modules():
    if isinstance(module, nn.Linear):
        prune.ln_structured(module, name='weight', amount=0.3, n=2, dim=0)

print("Structured pruning applied (30% of neurons pruned)")

# Global magnitude pruning
print("\nGlobal Magnitude Pruning:")

global_model = SimpleModel()

# Get all parameters to prune
parameters_to_prune = []
for name, module in global_model.named_modules():
    if isinstance(module, nn.Linear):
        parameters_to_prune.append((module, 'weight'))

# Apply global pruning
prune.global_unstructured(
    parameters_to_prune,
    pruning_method=prune.L1Unstructured,
    amount=0.7
)

total_params, nonzero_params = count_parameters(global_model)
sparsity = 1.0 - (nonzero_params / total_params)
print(f"Global pruning - Sparsity: {sparsity:.2%}")
print()

# Example 3: Knowledge Distillation
print("Example 3: Knowledge Distillation")
print("=" * 50)

class TeacherModel(nn.Module):
    """Large teacher model"""
    def __init__(self, input_size=784, hidden_size=512, num_classes=10):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, hidden_size // 2)
        self.fc4 = nn.Linear(hidden_size // 2, num_classes)
        
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        return x

class StudentModel(nn.Module):
    """Small student model"""
    def __init__(self, input_size=784, hidden_size=128, num_classes=10):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, num_classes)
        
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

def distillation_loss(student_outputs, teacher_outputs, labels, temperature=4.0, alpha=0.7):
    """Knowledge distillation loss"""
    # Soft targets loss
    soft_targets = F.softmax(teacher_outputs / temperature, dim=1)
    soft_loss = F.kl_div(
        F.log_softmax(student_outputs / temperature, dim=1),
        soft_targets,
        reduction='batchmean'
    ) * (temperature ** 2)
    
    # Hard targets loss
    hard_loss = F.cross_entropy(student_outputs, labels)
    
    # Combined loss
    return alpha * soft_loss + (1 - alpha) * hard_loss

# Create teacher and student models
teacher = TeacherModel()
student = StudentModel()

teacher_params = sum(p.numel() for p in teacher.parameters())
student_params = sum(p.numel() for p in student.parameters())

print(f"Teacher parameters: {teacher_params:,}")
print(f"Student parameters: {student_params:,} ({teacher_params/student_params:.1f}x smaller)")

# Simulate distillation training
print("\nDistillation training simulation:")

# Generate dummy data
batch_size = 32
x = torch.randn(batch_size, 784)
labels = torch.randint(0, 10, (batch_size,))

# Teacher predictions
teacher.eval()
with torch.no_grad():
    teacher_outputs = teacher(x)

# Student training step
student.train()
student_optimizer = optim.Adam(student.parameters(), lr=0.001)

student_outputs = student(x)
loss = distillation_loss(student_outputs, teacher_outputs, labels)

student_optimizer.zero_grad()
loss.backward()
student_optimizer.step()

print(f"Distillation loss: {loss.item():.4f}")
print()

# Example 4: Model Compression Techniques
print("Example 4: Advanced Compression Techniques")
print("=" * 50)

# Low-rank factorization
class LowRankLinear(nn.Module):
    """Low-rank factorization of linear layer"""
    def __init__(self, in_features, out_features, rank):
        super().__init__()
        self.U = nn.Linear(in_features, rank, bias=False)
        self.V = nn.Linear(rank, out_features, bias=True)
        
    def forward(self, x):
        return self.V(self.U(x))

# Compare original vs low-rank
original_layer = nn.Linear(512, 512)
lowrank_layer = LowRankLinear(512, 512, rank=64)

orig_params = sum(p.numel() for p in original_layer.parameters())
lr_params = sum(p.numel() for p in lowrank_layer.parameters())

print(f"Original Linear layer: {orig_params:,} parameters")
print(f"Low-rank Linear layer: {lr_params:,} parameters ({orig_params/lr_params:.1f}x compression)")

# Weight sharing/clustering
def apply_weight_clustering(weight, num_clusters=16):
    """Apply k-means clustering to weights"""
    from sklearn.cluster import KMeans
    
    # Flatten weights
    w_flat = weight.flatten().cpu().numpy()
    
    # Cluster weights
    kmeans = KMeans(n_clusters=num_clusters, random_state=42)
    kmeans.fit(w_flat.reshape(-1, 1))
    
    # Replace weights with cluster centers
    clustered = kmeans.cluster_centers_[kmeans.labels_]
    clustered_weight = torch.tensor(clustered.reshape(weight.shape))
    
    return clustered_weight, kmeans

# Example weight clustering
example_weight = torch.randn(256, 128)
clustered_weight, kmeans = apply_weight_clustering(example_weight, num_clusters=16)

print(f"\nWeight clustering:")
print(f"Original unique values: {len(torch.unique(example_weight))}")
print(f"Clustered unique values: {len(torch.unique(clustered_weight))}")
print(f"Compression ratio: {example_weight.numel() / 16:.1f}x")
print()

# Example 5: Hardware-Aware Optimization
print("Example 5: Hardware-Aware Optimization")
print("=" * 50)

class MobileOptimizedBlock(nn.Module):
    """Mobile-optimized building block"""
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        # Depthwise separable convolution
        self.depthwise = nn.Conv2d(in_channels, in_channels, 3, 
                                  stride=stride, padding=1, groups=in_channels)
        self.pointwise = nn.Conv2d(in_channels, out_channels, 1)
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self, x):
        x = self.depthwise(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.pointwise(x)
        x = self.bn2(x)
        x = self.relu(x)
        return x

# Compare standard convolution vs mobile-optimized
standard_conv = nn.Conv2d(128, 256, 3, padding=1)
mobile_block = MobileOptimizedBlock(128, 256)

std_params = sum(p.numel() for p in standard_conv.parameters())
mob_params = sum(p.numel() for p in mobile_block.parameters())

print(f"Standard Conv: {std_params:,} parameters")
print(f"Mobile Block: {mob_params:,} parameters ({std_params/mob_params:.1f}x fewer)")

# Operator fusion example
class FusedConvBNReLU(nn.Module):
    """Fused convolution, batch norm, and ReLU"""
    def __init__(self, in_channels, out_channels, kernel_size=3):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, padding=kernel_size//2)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self, x):
        # In practice, these would be fused at the kernel level
        return self.relu(self.bn(self.conv(x)))
    
    def fuse(self):
        """Fuse conv and bn weights"""
        # Get conv and bn parameters
        w_conv = self.conv.weight.data.clone()
        b_conv = self.conv.bias.data.clone() if self.conv.bias is not None else 0
        
        mean = self.bn.running_mean
        var = self.bn.running_var
        gamma = self.bn.weight.data
        beta = self.bn.bias.data
        eps = self.bn.eps
        
        # Fuse parameters
        std = torch.sqrt(var + eps)
        w_fused = w_conv * (gamma / std).view(-1, 1, 1, 1)
        b_fused = (b_conv - mean) * gamma / std + beta
        
        # Create fused conv
        fused_conv = nn.Conv2d(self.conv.in_channels, self.conv.out_channels, 
                              self.conv.kernel_size, padding=self.conv.padding[0])
        fused_conv.weight.data = w_fused
        fused_conv.bias.data = b_fused
        
        return nn.Sequential(fused_conv, self.relu)

print("\nOperator Fusion example created")
print()

# Example 6: Optimization Pipeline
print("Example 6: Complete Optimization Pipeline")
print("=" * 50)

def optimize_model_pipeline(model, optimization_config):
    """Complete model optimization pipeline"""
    results = {}
    
    # Original model metrics
    original_size, original_time = evaluate_model(model)
    results['original'] = {'size': original_size, 'time': original_time}
    
    # Step 1: Pruning
    if optimization_config.get('pruning', False):
        pruned_model = copy.deepcopy(model)
        
        # Apply pruning
        for name, module in pruned_model.named_modules():
            if isinstance(module, nn.Linear):
                prune.l1_unstructured(module, name='weight', 
                                    amount=optimization_config['pruning_amount'])
        
        pruned_size, pruned_time = evaluate_model(pruned_model)
        results['pruned'] = {'size': pruned_size, 'time': pruned_time}
    
    # Step 2: Quantization
    if optimization_config.get('quantization', False):
        quantized_model = quantize_dynamic(
            pruned_model if 'pruned_model' in locals() else model,
            {nn.Linear},
            dtype=torch.qint8
        )
        
        quant_size, quant_time = evaluate_model(quantized_model)
        results['quantized'] = {'size': quant_size, 'time': quant_time}
    
    return results

# Run optimization pipeline
optimization_config = {
    'pruning': True,
    'pruning_amount': 0.5,
    'quantization': True
}

pipeline_model = SimpleModel()
results = optimize_model_pipeline(pipeline_model, optimization_config)

print("Optimization Pipeline Results:")
print("-" * 40)
for stage, metrics in results.items():
    print(f"{stage.capitalize()}:")
    print(f"  Size: {metrics['size']:.2f} MB")
    print(f"  Inference: {metrics['time']:.2f} ms")
    
    if stage != 'original':
        size_reduction = results['original']['size'] / metrics['size']
        speed_up = results['original']['time'] / metrics['time']
        print(f"  Size reduction: {size_reduction:.2f}x")
        print(f"  Speed up: {speed_up:.2f}x")
    print()

# Visualization of optimization techniques
print("Optimization Techniques Summary")
print("=" * 50)

techniques = {
    "Quantization": {
        "Size Reduction": "2-4x",
        "Speed Up": "2-3x",
        "Accuracy Loss": "< 1%",
        "Difficulty": "Easy"
    },
    "Pruning": {
        "Size Reduction": "10-100x",
        "Speed Up": "2-10x",
        "Accuracy Loss": "1-5%",
        "Difficulty": "Medium"
    },
    "Distillation": {
        "Size Reduction": "5-100x",
        "Speed Up": "5-100x",
        "Accuracy Loss": "1-3%",
        "Difficulty": "Hard"
    },
    "Low-Rank": {
        "Size Reduction": "2-10x",
        "Speed Up": "2-5x",
        "Accuracy Loss": "1-3%",
        "Difficulty": "Medium"
    }
}

print("Technique Comparison:")
print("-" * 70)
print(f"{'Technique':<15} {'Size↓':<15} {'Speed↑':<15} {'Accuracy↓':<15} {'Difficulty':<10}")
print("-" * 70)

for technique, metrics in techniques.items():
    print(f"{technique:<15} {metrics['Size Reduction']:<15} "
          f"{metrics['Speed Up']:<15} {metrics['Accuracy Loss']:<15} "
          f"{metrics['Difficulty']:<10}")

print("\nBest Practices:")
print("1. Start with quantization - easy wins")
print("2. Combine techniques for maximum compression")
print("3. Always validate accuracy after optimization")
print("4. Consider hardware-specific optimizations")
print("5. Use profiling to identify bottlenecks")
print("6. Implement gradual optimization pipeline")