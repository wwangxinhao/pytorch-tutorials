#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Distributed Training

This script demonstrates various distributed training techniques in PyTorch,
including Data Parallel, Distributed Data Parallel, Model Parallel, and FSDP.
"""

import os
import time
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DataParallel, DistributedDataParallel as DDP
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
import matplotlib.pyplot as plt
import numpy as np

# Set random seed for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# -----------------------------------------------------------------------------
# Section 1: Introduction to Distributed Training
# -----------------------------------------------------------------------------

def intro_to_distributed_training():
    """Introduce distributed training concepts."""
    print("\nSection 1: Introduction to Distributed Training")
    print("-" * 50)
    print("Distributed training enables:")
    print("  - Faster training with multiple GPUs/nodes")
    print("  - Training larger models that don't fit on single GPU")
    print("  - Processing larger batch sizes")
    print("\nTypes of parallelism:")
    print("  - Data Parallel: Split data, replicate model")
    print("  - Model Parallel: Split model across devices")
    print("  - Pipeline Parallel: Split model into stages")
    print(f"\nCUDA available: {torch.cuda.is_available()}")
    print(f"Number of GPUs: {torch.cuda.device_count()}")

# -----------------------------------------------------------------------------
# Section 2: Sample Dataset and Model
# -----------------------------------------------------------------------------

class SyntheticDataset(Dataset):
    """A synthetic dataset for demonstration."""
    def __init__(self, size=10000, input_dim=784, num_classes=10):
        self.size = size
        self.input_dim = input_dim
        self.num_classes = num_classes
        
    def __len__(self):
        return self.size
    
    def __getitem__(self, idx):
        # Generate random data
        data = torch.randn(self.input_dim)
        label = torch.randint(0, self.num_classes, (1,)).item()
        return data, label

class SimpleNet(nn.Module):
    """A simple neural network for demonstration."""
    def __init__(self, input_dim=784, hidden_dim=256, num_classes=10):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, num_classes)
        self.dropout = nn.Dropout(0.2)
        
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        return x

# -----------------------------------------------------------------------------
# Section 3: Data Parallel (DP)
# -----------------------------------------------------------------------------

def demonstrate_data_parallel():
    """Demonstrate Data Parallel training."""
    print("\nSection 2: Data Parallel (DP)")
    print("-" * 50)
    
    if torch.cuda.device_count() < 2:
        print("Data Parallel requires at least 2 GPUs. Simulating with CPU...")
        return
    
    # Create model and wrap with DataParallel
    model = SimpleNet()
    model = DataParallel(model)
    model = model.cuda()
    
    # Create dataset and dataloader
    dataset = SyntheticDataset(size=1000)
    dataloader = DataLoader(dataset, batch_size=64, shuffle=True)
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    # Training loop
    print("Training with DataParallel...")
    start_time = time.time()
    
    for epoch in range(2):
        total_loss = 0
        for batch_idx, (data, target) in enumerate(dataloader):
            data, target = data.cuda(), target.cuda()
            
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
            if batch_idx % 5 == 0:
                print(f"  Epoch {epoch}, Batch {batch_idx}, Loss: {loss.item():.4f}")
        
        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch} - Average Loss: {avg_loss:.4f}")
    
    elapsed_time = time.time() - start_time
    print(f"Training time: {elapsed_time:.2f} seconds")

# -----------------------------------------------------------------------------
# Section 4: Distributed Data Parallel (DDP)
# -----------------------------------------------------------------------------

def setup_ddp(rank, world_size):
    """Initialize the distributed environment."""
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    
    # Initialize process group
    dist.init_process_group("nccl" if torch.cuda.is_available() else "gloo", 
                           rank=rank, world_size=world_size)

def cleanup_ddp():
    """Clean up the distributed environment."""
    dist.destroy_process_group()

def train_ddp(rank, world_size, num_epochs=2):
    """Training function for DDP."""
    print(f"\nProcess {rank}: Initializing DDP training...")
    setup_ddp(rank, world_size)
    
    # Create model and move to device
    device = torch.device(f'cuda:{rank}' if torch.cuda.is_available() else 'cpu')
    model = SimpleNet().to(device)
    
    # Wrap model with DDP
    if torch.cuda.is_available():
        ddp_model = DDP(model, device_ids=[rank])
    else:
        ddp_model = DDP(model)
    
    # Create dataset with DistributedSampler
    dataset = SyntheticDataset(size=1000)
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank)
    dataloader = DataLoader(dataset, batch_size=64, sampler=sampler)
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(ddp_model.parameters(), lr=0.001)
    
    # Training loop
    start_time = time.time()
    
    for epoch in range(num_epochs):
        sampler.set_epoch(epoch)  # Important for proper shuffling
        total_loss = 0
        
        for batch_idx, (data, target) in enumerate(dataloader):
            data, target = data.to(device), target.to(device)
            
            optimizer.zero_grad()
            output = ddp_model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
            if rank == 0 and batch_idx % 5 == 0:
                print(f"  Epoch {epoch}, Batch {batch_idx}, Loss: {loss.item():.4f}")
        
        # Synchronize and compute average loss
        avg_loss = total_loss / len(dataloader)
        if rank == 0:
            print(f"Epoch {epoch} - Average Loss: {avg_loss:.4f}")
    
    elapsed_time = time.time() - start_time
    if rank == 0:
        print(f"DDP Training time: {elapsed_time:.2f} seconds")
    
    cleanup_ddp()

def demonstrate_ddp():
    """Demonstrate Distributed Data Parallel training."""
    print("\nSection 3: Distributed Data Parallel (DDP)")
    print("-" * 50)
    
    world_size = min(torch.cuda.device_count(), 2) if torch.cuda.is_available() else 2
    
    if world_size < 2:
        print("DDP demonstration requires at least 2 processes.")
        print("Simulating with 2 CPU processes...")
    
    mp.spawn(train_ddp, args=(world_size,), nprocs=world_size, join=True)

# -----------------------------------------------------------------------------
# Section 5: Model Parallel
# -----------------------------------------------------------------------------

class ModelParallelNet(nn.Module):
    """A model split across multiple devices."""
    def __init__(self, input_dim=784, hidden_dim=256, num_classes=10):
        super().__init__()
        
        # Determine devices
        self.device1 = torch.device('cuda:0' if torch.cuda.device_count() > 0 else 'cpu')
        self.device2 = torch.device('cuda:1' if torch.cuda.device_count() > 1 else 'cpu')
        
        # Split model across devices
        self.fc1 = nn.Linear(input_dim, hidden_dim).to(self.device1)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim).to(self.device2)
        self.fc3 = nn.Linear(hidden_dim, num_classes).to(self.device2)
        
    def forward(self, x):
        x = x.to(self.device1)
        x = F.relu(self.fc1(x))
        
        x = x.to(self.device2)
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        
        return x

def demonstrate_model_parallel():
    """Demonstrate Model Parallel training."""
    print("\nSection 4: Model Parallel")
    print("-" * 50)
    
    if torch.cuda.device_count() < 2:
        print("Model Parallel requires at least 2 GPUs.")
        print("Demonstrating concept with CPU...")
    
    # Create model parallel network
    model = ModelParallelNet()
    
    # Create small dataset for demonstration
    dataset = SyntheticDataset(size=200)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
    
    # Loss and optimizer
    device2 = torch.device('cuda:1' if torch.cuda.device_count() > 1 else 'cpu')
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    # Training loop
    print("Training with Model Parallel...")
    start_time = time.time()
    
    for epoch in range(2):
        total_loss = 0
        for batch_idx, (data, target) in enumerate(dataloader):
            target = target.to(device2)
            
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
            if batch_idx % 5 == 0:
                print(f"  Epoch {epoch}, Batch {batch_idx}, Loss: {loss.item():.4f}")
        
        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch} - Average Loss: {avg_loss:.4f}")
    
    elapsed_time = time.time() - start_time
    print(f"Training time: {elapsed_time:.2f} seconds")

# -----------------------------------------------------------------------------
# Section 6: Pipeline Parallel (Conceptual Demo)
# -----------------------------------------------------------------------------

def demonstrate_pipeline_parallel():
    """Demonstrate Pipeline Parallel concepts."""
    print("\nSection 5: Pipeline Parallel")
    print("-" * 50)
    print("Pipeline Parallelism splits the model into stages and processes")
    print("micro-batches in a pipeline fashion to improve GPU utilization.")
    print("\nKey concepts:")
    print("  - Model is split into sequential stages")
    print("  - Each stage processes micro-batches")
    print("  - Reduces bubble (idle) time")
    print("  - Can be combined with data parallelism")
    
    # Simple visualization of pipeline scheduling
    print("\nPipeline Schedule Visualization:")
    print("Time →")
    print("GPU0: [F1][F2][F3][F4][B4][B3][B2][B1]")
    print("GPU1:    [F1][F2][F3][F4][B4][B3][B2][B1]")
    print("GPU2:       [F1][F2][F3][F4][B4][B3][B2][B1]")
    print("GPU3:          [F1][F2][F3][F4][B4][B3][B2][B1]")
    print("\nF=Forward, B=Backward, Numbers=Micro-batch IDs")

# -----------------------------------------------------------------------------
# Section 7: Fully Sharded Data Parallel (FSDP) Demo
# -----------------------------------------------------------------------------

def demonstrate_fsdp_concepts():
    """Demonstrate FSDP concepts."""
    print("\nSection 6: Fully Sharded Data Parallel (FSDP)")
    print("-" * 50)
    print("FSDP enables training of extremely large models by:")
    print("  - Sharding model parameters across GPUs")
    print("  - Sharding optimizer states")
    print("  - Sharding gradients")
    print("  - Optional CPU offloading")
    print("\nMemory savings example:")
    print("  Standard DDP: Each GPU stores full model")
    print("  FSDP: Each GPU stores 1/N of model (N = number of GPUs)")
    
    # Calculate memory savings
    model_size_gb = 7  # Example: 7B parameter model
    num_gpus = 8
    
    print(f"\nExample with {model_size_gb}B parameter model on {num_gpus} GPUs:")
    print(f"  DDP memory per GPU: {model_size_gb} GB")
    print(f"  FSDP memory per GPU: {model_size_gb/num_gpus:.2f} GB")
    print(f"  Memory reduction: {(1 - 1/num_gpus)*100:.1f}%")

# -----------------------------------------------------------------------------
# Section 8: Performance Comparison
# -----------------------------------------------------------------------------

def plot_performance_comparison():
    """Create a performance comparison visualization."""
    print("\nSection 7: Performance Comparison")
    print("-" * 50)
    
    # Simulated performance data
    methods = ['Single GPU', 'DP (4 GPUs)', 'DDP (4 GPUs)', 'FSDP (4 GPUs)']
    throughput = [100, 320, 380, 350]  # Images/second
    memory_usage = [16, 64, 64, 20]  # GB
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Throughput comparison
    ax1.bar(methods, throughput, color=['blue', 'green', 'orange', 'red'])
    ax1.set_ylabel('Throughput (samples/sec)')
    ax1.set_title('Training Throughput Comparison')
    ax1.set_ylim(0, 400)
    
    # Memory usage comparison
    ax2.bar(methods, memory_usage, color=['blue', 'green', 'orange', 'red'])
    ax2.set_ylabel('Memory Usage (GB)')
    ax2.set_title('Memory Usage Comparison')
    ax2.set_ylim(0, 70)
    
    plt.tight_layout()
    plt.savefig('distributed_training_comparison.png')
    print("Performance comparison saved to 'distributed_training_comparison.png'")

# -----------------------------------------------------------------------------
# Section 9: Best Practices
# -----------------------------------------------------------------------------

def print_best_practices():
    """Print distributed training best practices."""
    print("\nSection 8: Best Practices")
    print("-" * 50)
    print("1. Data Loading:")
    print("   - Use DistributedSampler for DDP")
    print("   - Pin memory for GPU training")
    print("   - Use multiple workers for data loading")
    print("\n2. Gradient Synchronization:")
    print("   - Use gradient accumulation for large batches")
    print("   - Consider gradient compression for bandwidth")
    print("\n3. Checkpointing:")
    print("   - Save checkpoints only from rank 0")
    print("   - Use torch.save with map_location for loading")
    print("\n4. Debugging:")
    print("   - Set TORCH_DISTRIBUTED_DEBUG=DETAIL")
    print("   - Use torch.distributed.barrier() for synchronization")
    print("   - Monitor GPU utilization and memory")
    print("\n5. Performance:")
    print("   - Profile with torch.profiler")
    print("   - Use mixed precision training")
    print("   - Overlap computation and communication")

# -----------------------------------------------------------------------------
# Main Function
# -----------------------------------------------------------------------------

def main():
    """Main function to run all demonstrations."""
    parser = argparse.ArgumentParser(description='Distributed Training Tutorial')
    parser.add_argument('--distributed', action='store_true', 
                       help='Run distributed training examples')
    args = parser.parse_args()
    
    print("=" * 70)
    print("Distributed Training Tutorial")
    print("=" * 70)
    
    # Run demonstrations
    intro_to_distributed_training()
    
    if torch.cuda.device_count() >= 2:
        demonstrate_data_parallel()
    else:
        print("\nSkipping Data Parallel demo (requires 2+ GPUs)")
    
    if args.distributed:
        demonstrate_ddp()
    else:
        print("\nSkipping DDP demo (use --distributed flag to run)")
    
    if torch.cuda.device_count() >= 2:
        demonstrate_model_parallel()
    else:
        print("\nSkipping Model Parallel demo (requires 2+ GPUs)")
    
    demonstrate_pipeline_parallel()
    demonstrate_fsdp_concepts()
    plot_performance_comparison()
    print_best_practices()
    
    print("\n" + "=" * 70)
    print("Tutorial completed!")
    print("=" * 70)

if __name__ == "__main__":
    main()