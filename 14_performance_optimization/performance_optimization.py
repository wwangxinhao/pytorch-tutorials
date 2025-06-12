"""
Tutorial 14: Performance Optimization
=====================================

This tutorial covers comprehensive performance optimization techniques
for PyTorch models, from profiling to advanced optimization strategies.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import torchvision
import torchvision.transforms as transforms
import time
import numpy as np
from torch.profiler import profile, record_function, ProfilerActivity
import torch.cuda.amp as amp
from torch.nn.parallel import DataParallel, DistributedDataParallel
import torch.distributed as dist
import torch.multiprocessing as mp
import os
import psutil
import gc

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")
print()

# Example 1: Basic Profiling
print("Example 1: PyTorch Profiler")
print("=" * 50)

class SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 64, 3, padding=1)
        self.conv2 = nn.Conv2d(64, 128, 3, padding=1)
        self.fc1 = nn.Linear(128 * 8 * 8, 256)
        self.fc2 = nn.Linear(256, 10)
        
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Profile the model
model = SimpleModel().to(device)
inputs = torch.randn(32, 3, 32, 32).to(device)

# Use PyTorch profiler
with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
             record_shapes=True,
             profile_memory=True,
             with_stack=True) as prof:
    with record_function("model_inference"):
        for _ in range(10):
            model(inputs)

# Print profiler results
print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))
print()

# Example 2: Memory Optimization
print("Example 2: Memory Optimization")
print("=" * 50)

def get_memory_usage():
    if torch.cuda.is_available():
        return torch.cuda.memory_allocated() / 1024**2  # MB
    else:
        return psutil.Process().memory_info().rss / 1024**2  # MB

# Memory-efficient gradient checkpointing
class CheckpointedModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(1024, 1024),
                nn.ReLU(),
                nn.Dropout(0.1)
            ) for _ in range(10)
        ])
        self.final = nn.Linear(1024, 10)
    
    def forward(self, x):
        for layer in self.layers:
            # Use checkpoint to trade compute for memory
            x = torch.utils.checkpoint.checkpoint(layer, x)
        return self.final(x)

# Compare memory usage
print("Memory usage comparison:")
x = torch.randn(128, 1024).to(device)

# Without checkpointing
regular_model = nn.Sequential(*[
    nn.Sequential(nn.Linear(1024, 1024), nn.ReLU(), nn.Dropout(0.1))
    for _ in range(10)
] + [nn.Linear(1024, 10)]).to(device)

mem_before = get_memory_usage()
y1 = regular_model(x)
loss1 = y1.sum()
loss1.backward()
mem_regular = get_memory_usage() - mem_before
print(f"Regular model: {mem_regular:.2f} MB")

# With checkpointing
checkpointed_model = CheckpointedModel().to(device)
optimizer = torch.optim.Adam(checkpointed_model.parameters())
optimizer.zero_grad()

mem_before = get_memory_usage()
y2 = checkpointed_model(x)
loss2 = y2.sum()
loss2.backward()
mem_checkpoint = get_memory_usage() - mem_before
print(f"Checkpointed model: {mem_checkpoint:.2f} MB")
print(f"Memory saved: {(1 - mem_checkpoint/mem_regular)*100:.1f}%")
print()

# Example 3: Mixed Precision Training
print("Example 3: Mixed Precision Training")
print("=" * 50)

# Create a more complex model for mixed precision demo
class MixedPrecisionModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1)
        )
        self.classifier = nn.Linear(128, 10)
    
    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

# Training with mixed precision
def train_with_amp(model, dataloader, epochs=2):
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters())
    scaler = amp.GradScaler()
    
    model.train()
    total_time = 0
    
    for epoch in range(epochs):
        epoch_start = time.time()
        for i, (inputs, targets) in enumerate(dataloader):
            if i >= 10:  # Limit iterations for demo
                break
                
            inputs, targets = inputs.to(device), targets.to(device)
            
            optimizer.zero_grad()
            
            # Mixed precision forward pass
            with amp.autocast():
                outputs = model(inputs)
                loss = F.cross_entropy(outputs, targets)
            
            # Scaled backward pass
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        
        epoch_time = time.time() - epoch_start
        total_time += epoch_time
    
    return total_time / epochs

# Create dummy dataset
class DummyDataset(Dataset):
    def __init__(self, size=1000):
        self.size = size
    
    def __len__(self):
        return self.size
    
    def __getitem__(self, idx):
        return torch.randn(3, 32, 32), torch.randint(0, 10, (1,)).item()

dataset = DummyDataset()
dataloader = DataLoader(dataset, batch_size=64, num_workers=0)

# Compare training times
model_fp32 = MixedPrecisionModel()
model_amp = MixedPrecisionModel()

print("Training with FP32...")
time_fp32 = train_with_amp(model_fp32, dataloader)
print(f"Average epoch time: {time_fp32:.3f}s")

print("\nTraining with AMP...")
time_amp = train_with_amp(model_amp, dataloader)
print(f"Average epoch time: {time_amp:.3f}s")
print(f"Speedup: {time_fp32/time_amp:.2f}x")
print()

# Example 4: Data Loading Optimization
print("Example 4: Data Loading Optimization")
print("=" * 50)

# Optimized dataset with caching
class OptimizedDataset(Dataset):
    def __init__(self, size=1000, cache_size=100):
        self.size = size
        self.cache_size = cache_size
        self.cache = {}
        self.transform = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(32, padding=4),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
    
    def __len__(self):
        return self.size
    
    def __getitem__(self, idx):
        # Simple caching mechanism
        if idx in self.cache:
            return self.cache[idx]
        
        # Simulate data loading
        image = torch.randn(3, 32, 32)
        label = torch.randint(0, 10, (1,)).item()
        
        # Cache recent items
        if len(self.cache) < self.cache_size:
            self.cache[idx] = (image, label)
        
        return image, label

# Compare data loading performance
def benchmark_dataloader(dataset, num_workers, pin_memory=False):
    dataloader = DataLoader(
        dataset,
        batch_size=128,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=(num_workers > 0)
    )
    
    start_time = time.time()
    for i, (data, target) in enumerate(dataloader):
        if i >= 50:  # Limit iterations
            break
        # Simulate processing
        data = data.to(device, non_blocking=True)
    
    total_time = time.time() - start_time
    return total_time

dataset = OptimizedDataset(5000)

print("Data loading benchmark:")
for num_workers in [0, 2, 4]:
    for pin_memory in [False, True]:
        time_taken = benchmark_dataloader(dataset, num_workers, pin_memory)
        print(f"Workers: {num_workers}, Pin memory: {pin_memory} - Time: {time_taken:.3f}s")
print()

# Example 5: Model Optimization with TorchScript
print("Example 5: TorchScript Optimization")
print("=" * 50)

# Create a model for scripting
class ScriptableModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, 3)
        self.conv2 = nn.Conv2d(32, 64, 3)
        self.fc = nn.Linear(64 * 6 * 6, 10)
    
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x

# Compare scripted vs regular model
model = ScriptableModel().to(device)
model.eval()

# Script the model
scripted_model = torch.jit.script(model)

# Benchmark
x = torch.randn(100, 3, 32, 32).to(device)

# Regular model
torch.cuda.synchronize() if torch.cuda.is_available() else None
start = time.time()
for _ in range(100):
    _ = model(x)
torch.cuda.synchronize() if torch.cuda.is_available() else None
regular_time = time.time() - start

# Scripted model
torch.cuda.synchronize() if torch.cuda.is_available() else None
start = time.time()
for _ in range(100):
    _ = scripted_model(x)
torch.cuda.synchronize() if torch.cuda.is_available() else None
scripted_time = time.time() - start

print(f"Regular model: {regular_time:.3f}s")
print(f"Scripted model: {scripted_time:.3f}s")
print(f"Speedup: {regular_time/scripted_time:.2f}x")
print()

# Example 6: Tensor Operations Optimization
print("Example 6: Tensor Operations Optimization")
print("=" * 50)

# Inefficient operations
def inefficient_operation(x):
    result = torch.zeros_like(x)
    for i in range(x.shape[0]):
        for j in range(x.shape[1]):
            result[i, j] = x[i, j] * 2 + 1
    return result

# Efficient vectorized operation
def efficient_operation(x):
    return x * 2 + 1

# Benchmark
x = torch.randn(1000, 1000).to(device)

start = time.time()
_ = inefficient_operation(x)
inefficient_time = time.time() - start

start = time.time()
_ = efficient_operation(x)
efficient_time = time.time() - start

print(f"Inefficient operation: {inefficient_time:.4f}s")
print(f"Efficient operation: {efficient_time:.4f}s")
print(f"Speedup: {inefficient_time/efficient_time:.0f}x")
print()

# Example 7: Memory-Efficient Attention
print("Example 7: Memory-Efficient Attention")
print("=" * 50)

class EfficientAttention(nn.Module):
    def __init__(self, dim, num_heads=8, chunk_size=256):
        super().__init__()
        self.num_heads = num_heads
        self.chunk_size = chunk_size
        self.scale = (dim // num_heads) ** -0.5
        
        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)
    
    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        # Chunked attention computation
        attn_chunks = []
        for i in range(0, N, self.chunk_size):
            end_idx = min(i + self.chunk_size, N)
            q_chunk = q[:, :, i:end_idx]
            
            # Compute attention for this chunk
            attn = (q_chunk @ k.transpose(-2, -1)) * self.scale
            attn = attn.softmax(dim=-1)
            attn_chunk = attn @ v
            attn_chunks.append(attn_chunk)
        
        # Concatenate chunks
        x = torch.cat(attn_chunks, dim=2)
        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        return x

# Test memory-efficient attention
seq_len = 1024
dim = 512
batch_size = 8

attention = EfficientAttention(dim).to(device)
x = torch.randn(batch_size, seq_len, dim).to(device)

mem_before = get_memory_usage()
output = attention(x)
mem_used = get_memory_usage() - mem_before
print(f"Memory used by efficient attention: {mem_used:.2f} MB")
print(f"Output shape: {output.shape}")
print()

# Example 8: Custom Memory Allocator
print("Example 8: Custom Memory Management")
print("=" * 50)

class TensorPool:
    """Simple tensor pool for reusing allocations"""
    def __init__(self):
        self.pool = {}
    
    def get(self, shape, dtype=torch.float32, device='cpu'):
        key = (tuple(shape), dtype, device)
        if key in self.pool and len(self.pool[key]) > 0:
            return self.pool[key].pop()
        return torch.empty(shape, dtype=dtype, device=device)
    
    def release(self, tensor):
        key = (tuple(tensor.shape), tensor.dtype, tensor.device)
        if key not in self.pool:
            self.pool[key] = []
        self.pool[key].append(tensor)
    
    def clear(self):
        self.pool.clear()

# Example usage
pool = TensorPool()

# Simulate multiple allocations
print("Using tensor pool:")
tensors = []
for i in range(5):
    t = pool.get((100, 100), device=device)
    tensors.append(t)

# Release some tensors back to pool
for t in tensors[:3]:
    pool.release(t)

# Reuse from pool
print(f"Pool size before reuse: {sum(len(v) for v in pool.pool.values())}")
new_tensors = []
for i in range(3):
    t = pool.get((100, 100), device=device)
    new_tensors.append(t)
print(f"Pool size after reuse: {sum(len(v) for v in pool.pool.values())}")
print()

# Best Practices Summary
print("Performance Optimization Best Practices")
print("=" * 50)
print("1. Profile First: Always profile before optimizing")
print("2. Memory Management: Use gradient checkpointing for large models")
print("3. Mixed Precision: Use AMP for faster training")
print("4. Data Loading: Use multiple workers and pin_memory")
print("5. Batch Size: Find optimal batch size for your GPU")
print("6. TorchScript: Script models for production deployment")
print("7. Operator Fusion: Use fused operations when available")
print("8. Distributed Training: Scale across multiple GPUs")
print()

# Performance Checklist
print("Performance Optimization Checklist")
print("-" * 30)
checklist = [
    "Profile with torch.profiler",
    "Enable mixed precision training",
    "Optimize data loading pipeline",
    "Use gradient checkpointing for memory",
    "Apply model quantization",
    "Enable CUDNN benchmarking",
    "Use TorchScript for inference",
    "Implement custom CUDA kernels for bottlenecks",
    "Use distributed training for large models",
    "Monitor GPU utilization"
]

for item in checklist:
    print(f"- [ ] {item}")

print("\nRemember: Premature optimization is the root of all evil!")
print("Always measure and profile before optimizing.")