# Distributed Training

This tutorial covers distributed training techniques in PyTorch, enabling you to scale your models across multiple GPUs and machines for faster training and larger model capacity.

## Table of Contents
1. [Introduction to Distributed Training](#introduction-to-distributed-training)
2. [Data Parallel (DP)](#data-parallel-dp)
3. [Distributed Data Parallel (DDP)](#distributed-data-parallel-ddp)
4. [Model Parallel](#model-parallel)
5. [Pipeline Parallelism](#pipeline-parallelism)
6. [Fully Sharded Data Parallel (FSDP)](#fully-sharded-data-parallel-fsdp)

## Introduction to Distributed Training

- Why distributed training?
- Types of parallelism
- Communication backends
- Hardware requirements

## Data Parallel (DP)

- Single-machine multi-GPU training
- Automatic gradient averaging
- Limitations and performance considerations
- When to use DP vs DDP

## Distributed Data Parallel (DDP)

- Multi-GPU and multi-node training
- Process groups and initialization
- Gradient synchronization
- Best practices for DDP

## Model Parallel

- Splitting models across devices
- Forward and backward pass coordination
- Memory management
- Use cases for very large models

## Pipeline Parallelism

- Micro-batch processing
- Pipeline stages
- Bubble overhead optimization
- Combining with data parallelism

## Fully Sharded Data Parallel (FSDP)

- Sharding model parameters, gradients, and optimizer states
- Memory efficiency for large models
- Configuration options
- Performance tuning

## Running the Tutorial

To run this tutorial:

```bash
# Single GPU example
python distributed_training.py

# Multi-GPU DDP example
torchrun --nproc_per_node=2 distributed_training.py --distributed

# Multi-node example
torchrun --nproc_per_node=2 --nnodes=2 --node_rank=0 --master_addr="192.168.1.1" --master_port=29500 distributed_training.py --distributed
```

Alternatively, you can follow along with the Jupyter notebook `distributed_training.ipynb` for an interactive experience.

## Prerequisites

- Python 3.7+
- PyTorch 1.10+
- Multiple GPUs (for multi-GPU examples)
- NCCL backend (for optimal performance)

## Related Tutorials

1. [Training Neural Networks](../04_training_neural_networks/README.md)
2. [PyTorch Lightning](../11_pytorch_lightning/README.md)

## Introduction to Distributed Training

Distributed training is essential for modern deep learning, allowing you to:
- **Reduce training time** by leveraging multiple GPUs/machines
- **Train larger models** that don't fit on a single GPU
- **Process larger batches** for better gradient estimates

### Types of Parallelism

1. **Data Parallelism**: Split data across devices, replicate model
2. **Model Parallelism**: Split model across devices
3. **Pipeline Parallelism**: Split model into stages processed sequentially
4. **Hybrid Approaches**: Combine multiple parallelism strategies

### Communication Backends

PyTorch supports multiple backends for inter-process communication:
- **NCCL** (recommended for GPUs): Optimized for NVIDIA GPUs
- **Gloo**: CPU and GPU support, good for development
- **MPI**: Message Passing Interface, requires separate installation

## Data Parallel (DP)

DataParallel is the simplest way to use multiple GPUs on a single machine:

```python
import torch
import torch.nn as nn

# Create model
model = nn.Sequential(
    nn.Linear(10, 100),
    nn.ReLU(),
    nn.Linear(100, 10)
)

# Wrap with DataParallel
if torch.cuda.device_count() > 1:
    model = nn.DataParallel(model)
model = model.to('cuda')

# Forward pass automatically uses all GPUs
input = torch.randn(32, 10).to('cuda')
output = model(input)
```

### Limitations of DP

- Python GIL bottleneck
- Imbalanced GPU memory usage
- Lower performance compared to DDP
- Single-machine only

## Distributed Data Parallel (DDP)

DDP is the recommended approach for distributed training:

```python
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP

def setup(rank, world_size):
    """Initialize the distributed environment."""
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

def cleanup():
    """Clean up the distributed environment."""
    dist.destroy_process_group()

def train(rank, world_size):
    setup(rank, world_size)
    
    # Create model and move to GPU
    model = nn.Sequential(
        nn.Linear(10, 100),
        nn.ReLU(),
        nn.Linear(100, 10)
    ).to(rank)
    
    # Wrap with DDP
    ddp_model = DDP(model, device_ids=[rank])
    
    # Create data loader with DistributedSampler
    dataset = YourDataset()
    sampler = torch.utils.data.distributed.DistributedSampler(
        dataset, num_replicas=world_size, rank=rank
    )
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=32, sampler=sampler
    )
    
    # Training loop
    optimizer = torch.optim.Adam(ddp_model.parameters())
    for epoch in range(num_epochs):
        sampler.set_epoch(epoch)  # Ensure different shuffling per epoch
        for data, target in dataloader:
            optimizer.zero_grad()
            output = ddp_model(data.to(rank))
            loss = loss_fn(output, target.to(rank))
            loss.backward()
            optimizer.step()
    
    cleanup()

# Launch distributed training
if __name__ == "__main__":
    world_size = torch.cuda.device_count()
    mp.spawn(train, args=(world_size,), nprocs=world_size, join=True)
```

### DDP Best Practices

1. **Use DistributedSampler** to ensure each process gets different data
2. **Set random seeds** per process for reproducibility
3. **Synchronize metrics** across processes when needed
4. **Save checkpoints** from only one process (usually rank 0)
5. **Use gradient accumulation** for large effective batch sizes

## Model Parallel

For models too large to fit on a single GPU:

```python
class ModelParallelNet(nn.Module):
    def __init__(self):
        super().__init__()
        # Place different parts on different GPUs
        self.layer1 = nn.Linear(10, 100).to('cuda:0')
        self.layer2 = nn.Linear(100, 100).to('cuda:1')
        self.layer3 = nn.Linear(100, 10).to('cuda:1')
    
    def forward(self, x):
        x = self.layer1(x.to('cuda:0'))
        x = self.layer2(x.to('cuda:1'))
        x = self.layer3(x)
        return x
```

### Challenges with Model Parallel

- Device idle time during forward/backward
- Complex implementation for arbitrary models
- Communication overhead between devices

## Pipeline Parallelism

Pipeline parallelism addresses idle time by processing micro-batches:

```python
from torch.distributed.pipeline.sync import Pipe

# Define sequential model
model = nn.Sequential(
    nn.Linear(10, 100),
    nn.ReLU(),
    nn.Linear(100, 100),
    nn.ReLU(),
    nn.Linear(100, 10)
)

# Create pipeline (splits model into balanced stages)
model = Pipe(model, balance=[2, 3], devices=['cuda:0', 'cuda:1'])

# Forward pass with micro-batches
output = model(input)
```

### Pipeline Parallelism Benefits

- Better GPU utilization
- Automatic micro-batch scheduling
- Can combine with data parallelism
- Suitable for deep networks

## Fully Sharded Data Parallel (FSDP)

FSDP enables training of extremely large models by sharding parameters:

```python
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp.wrap import wrap

class FSDPModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = wrap(nn.Linear(10, 100))
        self.layer2 = wrap(nn.Linear(100, 100))
        self.layer3 = wrap(nn.Linear(100, 10))
    
    def forward(self, x):
        x = torch.relu(self.layer1(x))
        x = torch.relu(self.layer2(x))
        return self.layer3(x)

# Wrap entire model with FSDP
model = FSDP(FSDPModel())

# Training works as normal
optimizer = torch.optim.Adam(model.parameters())
for data, target in dataloader:
    optimizer.zero_grad()
    output = model(data)
    loss = loss_fn(output, target)
    loss.backward()
    optimizer.step()
```

### FSDP Configuration

```python
from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
    MixedPrecision,
    BackwardPrefetch,
    ShardingStrategy,
)

# Configure FSDP
fsdp_config = {
    "sharding_strategy": ShardingStrategy.FULL_SHARD,
    "cpu_offload": CPUOffload(offload_params=True),
    "mixed_precision": MixedPrecision(
        param_dtype=torch.float16,
        reduce_dtype=torch.float16,
        buffer_dtype=torch.float16,
    ),
    "backward_prefetch": BackwardPrefetch.BACKWARD_PRE,
}

model = FSDP(model, **fsdp_config)
```

## Performance Optimization Tips

1. **Profile your code** to identify bottlenecks
2. **Overlap computation and communication** when possible
3. **Use mixed precision training** for faster computation
4. **Tune batch sizes** for optimal GPU utilization
5. **Monitor GPU memory** and adjust accordingly

## Common Pitfalls and Solutions

### Hanging Processes
- Ensure all processes execute the same number of collective operations
- Use proper error handling and cleanup

### Gradient Synchronization Issues
- Verify all processes have the same model architecture
- Check for conditional logic that might cause divergence

### Memory Imbalance
- Balance model partitioning for model parallel
- Use gradient checkpointing for memory-intensive models

## Monitoring and Debugging

```python
# Log only from main process
if rank == 0:
    print(f"Epoch {epoch}, Loss: {loss.item()}")

# Synchronize before timing
dist.barrier()
start_time = time.time()

# Use distributed.all_reduce for metrics
dist.all_reduce(loss, op=dist.ReduceOp.AVG)
```

## Conclusion

Distributed training is essential for modern deep learning. Key takeaways:
- Use DDP for most multi-GPU scenarios
- Consider FSDP for very large models
- Combine strategies for optimal performance
- Always profile and monitor your training

The next tutorials will explore more advanced optimization techniques and deployment strategies.