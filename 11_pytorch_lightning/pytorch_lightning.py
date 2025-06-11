#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
PyTorch Lightning Tutorial

This script demonstrates how to use PyTorch Lightning to simplify deep learning
workflows, including automatic optimization, distributed training, and logging.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
import torchvision
import torchvision.transforms as transforms
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor
from pytorch_lightning.loggers import TensorBoardLogger
import numpy as np
import os
from typing import Optional
import warnings
warnings.filterwarnings('ignore')

# Set random seed for reproducibility
pl.seed_everything(42)

# Create output directory
output_dir = "11_pytorch_lightning_outputs"
os.makedirs(output_dir, exist_ok=True)

# -----------------------------------------------------------------------------
# Section 1: Introduction to PyTorch Lightning
# -----------------------------------------------------------------------------

def intro_to_lightning():
    """Introduction to PyTorch Lightning concepts."""
    print("\nSection 1: Introduction to PyTorch Lightning")
    print("-" * 70)
    print("PyTorch Lightning is a lightweight wrapper for PyTorch that:")
    print("  - Eliminates boilerplate code")
    print("  - Provides automatic optimization and device placement")
    print("  - Enables easy distributed training")
    print("  - Integrates logging and checkpointing")
    print("  - Ensures reproducibility and best practices")

# -----------------------------------------------------------------------------
# Section 2: Lightning Module
# -----------------------------------------------------------------------------

class LitMNISTClassifier(pl.LightningModule):
    """A simple CNN for MNIST classification using PyTorch Lightning."""
    
    def __init__(self, learning_rate=1e-3):
        super().__init__()
        self.save_hyperparameters()
        
        # Define model architecture
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2)
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)
        self.dropout = nn.Dropout(0.25)
        
        # Metrics
        self.train_accuracy = pl.metrics.Accuracy()
        self.val_accuracy = pl.metrics.Accuracy()
        
    def forward(self, x):
        """Forward pass."""
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x
    
    def training_step(self, batch, batch_idx):
        """Training step - called for each batch."""
        x, y = batch
        logits = self(x)
        loss = F.cross_entropy(logits, y)
        
        # Calculate accuracy
        preds = torch.argmax(logits, dim=1)
        acc = self.train_accuracy(preds, y)
        
        # Log metrics
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log('train_acc', acc, on_step=True, on_epoch=True, prog_bar=True)
        
        return loss
    
    def validation_step(self, batch, batch_idx):
        """Validation step - called for each validation batch."""
        x, y = batch
        logits = self(x)
        loss = F.cross_entropy(logits, y)
        
        # Calculate accuracy
        preds = torch.argmax(logits, dim=1)
        acc = self.val_accuracy(preds, y)
        
        # Log metrics
        self.log('val_loss', loss, on_epoch=True, prog_bar=True)
        self.log('val_acc', acc, on_epoch=True, prog_bar=True)
        
        return loss
    
    def test_step(self, batch, batch_idx):
        """Test step - called for each test batch."""
        x, y = batch
        logits = self(x)
        loss = F.cross_entropy(logits, y)
        
        # Calculate accuracy
        preds = torch.argmax(logits, dim=1)
        acc = (preds == y).float().mean()
        
        # Log metrics
        self.log('test_loss', loss)
        self.log('test_acc', acc)
        
        return loss
    
    def configure_optimizers(self):
        """Configure optimizers and schedulers."""
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate)
        
        # Optional: add learning rate scheduler
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=5
        )
        
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'monitor': 'val_loss',
                'interval': 'epoch',
                'frequency': 1
            }
        }

def demonstrate_lightning_module():
    """Demonstrate Lightning Module basics."""
    print("\nSection 2: Lightning Module")
    print("-" * 70)
    
    # Create model
    model = LitMNISTClassifier(learning_rate=1e-3)
    print("Created Lightning Module:")
    print(f"  - Number of parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"  - Hyperparameters: {model.hparams}")
    
    # Show example forward pass
    dummy_input = torch.randn(1, 1, 28, 28)
    output = model(dummy_input)
    print(f"  - Input shape: {dummy_input.shape}")
    print(f"  - Output shape: {output.shape}")

# -----------------------------------------------------------------------------
# Section 3: Data Module
# -----------------------------------------------------------------------------

class MNISTDataModule(pl.LightningDataModule):
    """PyTorch Lightning DataModule for MNIST."""
    
    def __init__(self, data_dir='./data', batch_size=64, num_workers=2):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        
        # Define transforms
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
    
    def prepare_data(self):
        """Download data if needed. Called only on 1 GPU/process."""
        torchvision.datasets.MNIST(self.data_dir, train=True, download=True)
        torchvision.datasets.MNIST(self.data_dir, train=False, download=True)
    
    def setup(self, stage: Optional[str] = None):
        """Setup train/val/test datasets. Called on every GPU."""
        if stage == 'fit' or stage is None:
            mnist_full = torchvision.datasets.MNIST(
                self.data_dir, train=True, transform=self.transform
            )
            # Split into train and validation
            self.mnist_train, self.mnist_val = random_split(
                mnist_full, [55000, 5000]
            )
        
        if stage == 'test' or stage is None:
            self.mnist_test = torchvision.datasets.MNIST(
                self.data_dir, train=False, transform=self.transform
            )
    
    def train_dataloader(self):
        return DataLoader(
            self.mnist_train, 
            batch_size=self.batch_size, 
            shuffle=True,
            num_workers=self.num_workers
        )
    
    def val_dataloader(self):
        return DataLoader(
            self.mnist_val, 
            batch_size=self.batch_size, 
            shuffle=False,
            num_workers=self.num_workers
        )
    
    def test_dataloader(self):
        return DataLoader(
            self.mnist_test, 
            batch_size=self.batch_size, 
            shuffle=False,
            num_workers=self.num_workers
        )

def demonstrate_data_module():
    """Demonstrate Lightning DataModule."""
    print("\nSection 3: Data Module")
    print("-" * 70)
    
    # Create data module
    data_module = MNISTDataModule(batch_size=64, num_workers=0)
    data_module.prepare_data()
    data_module.setup('fit')
    
    print("Created DataModule:")
    print(f"  - Train samples: {len(data_module.mnist_train)}")
    print(f"  - Val samples: {len(data_module.mnist_val)}")
    print(f"  - Batch size: {data_module.batch_size}")
    
    # Show sample batch
    train_loader = data_module.train_dataloader()
    batch = next(iter(train_loader))
    print(f"  - Sample batch shape: {batch[0].shape}")

# -----------------------------------------------------------------------------
# Section 4: Training with Trainer
# -----------------------------------------------------------------------------

def demonstrate_basic_training():
    """Demonstrate basic training with PyTorch Lightning."""
    print("\nSection 4: Basic Training with Trainer")
    print("-" * 70)
    
    # Create model and data
    model = LitMNISTClassifier(learning_rate=1e-3)
    data_module = MNISTDataModule(batch_size=64, num_workers=0)
    
    # Create trainer
    trainer = pl.Trainer(
        max_epochs=3,
        gpus=1 if torch.cuda.is_available() else 0,
        progress_bar_refresh_rate=20,
        log_every_n_steps=50,
        default_root_dir=output_dir
    )
    
    print("Training model...")
    # Train model
    trainer.fit(model, data_module)
    
    # Test model
    print("\nTesting model...")
    test_results = trainer.test(model, data_module)
    print(f"Test results: {test_results}")

# -----------------------------------------------------------------------------
# Section 5: Advanced Features - Callbacks
# -----------------------------------------------------------------------------

class CustomCallback(pl.Callback):
    """Custom callback example."""
    
    def on_epoch_start(self, trainer, pl_module):
        print(f"\nEpoch {trainer.current_epoch} starting...")
    
    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        if batch_idx % 100 == 0:
            print(f"  Batch {batch_idx}: loss = {outputs['loss']:.4f}")

def demonstrate_callbacks():
    """Demonstrate callbacks in PyTorch Lightning."""
    print("\nSection 5: Callbacks")
    print("-" * 70)
    
    # Create model and data
    model = LitMNISTClassifier(learning_rate=1e-3)
    data_module = MNISTDataModule(batch_size=64, num_workers=0)
    
    # Define callbacks
    checkpoint_callback = ModelCheckpoint(
        dirpath=os.path.join(output_dir, 'checkpoints'),
        filename='mnist-{epoch:02d}-{val_loss:.2f}',
        monitor='val_loss',
        mode='min',
        save_top_k=3
    )
    
    early_stop_callback = EarlyStopping(
        monitor='val_loss',
        patience=3,
        mode='min'
    )
    
    lr_monitor = LearningRateMonitor(logging_interval='epoch')
    
    # Create trainer with callbacks
    trainer = pl.Trainer(
        max_epochs=5,
        gpus=1 if torch.cuda.is_available() else 0,
        callbacks=[checkpoint_callback, early_stop_callback, lr_monitor],
        progress_bar_refresh_rate=0,  # Disable for cleaner output
        default_root_dir=output_dir
    )
    
    print("Training with callbacks:")
    print("  - ModelCheckpoint: saves best models")
    print("  - EarlyStopping: stops when validation loss stops improving")
    print("  - LearningRateMonitor: tracks learning rate")
    
    # Train model
    trainer.fit(model, data_module)
    
    print(f"\nBest model path: {checkpoint_callback.best_model_path}")

# -----------------------------------------------------------------------------
# Section 6: Logging and Visualization
# -----------------------------------------------------------------------------

def demonstrate_logging():
    """Demonstrate logging with TensorBoard."""
    print("\nSection 6: Logging and Visualization")
    print("-" * 70)
    
    # Create model and data
    model = LitMNISTClassifier(learning_rate=1e-3)
    data_module = MNISTDataModule(batch_size=64, num_workers=0)
    
    # Create logger
    tb_logger = TensorBoardLogger(
        save_dir=output_dir,
        name='mnist_logs',
        version='v1'
    )
    
    # Create trainer with logger
    trainer = pl.Trainer(
        max_epochs=3,
        gpus=1 if torch.cuda.is_available() else 0,
        logger=tb_logger,
        progress_bar_refresh_rate=0,
        log_every_n_steps=20
    )
    
    print("Training with TensorBoard logging...")
    print(f"Log directory: {tb_logger.log_dir}")
    print("To view logs, run: tensorboard --logdir={}".format(output_dir))
    
    # Train model
    trainer.fit(model, data_module)

# -----------------------------------------------------------------------------
# Section 7: Multi-GPU Training
# -----------------------------------------------------------------------------

def demonstrate_distributed_training():
    """Demonstrate distributed training capabilities."""
    print("\nSection 7: Multi-GPU Training")
    print("-" * 70)
    
    # Check available GPUs
    num_gpus = torch.cuda.device_count()
    print(f"Available GPUs: {num_gpus}")
    
    if num_gpus > 1:
        # Create model and data
        model = LitMNISTClassifier(learning_rate=1e-3)
        data_module = MNISTDataModule(batch_size=64 * num_gpus, num_workers=4)
        
        # Create trainer for multi-GPU
        trainer = pl.Trainer(
            max_epochs=2,
            gpus=num_gpus,
            accelerator='ddp',  # Distributed Data Parallel
            progress_bar_refresh_rate=20
        )
        
        print(f"Training on {num_gpus} GPUs with DDP...")
        trainer.fit(model, data_module)
    else:
        print("Multi-GPU training requires multiple GPUs.")
        print("PyTorch Lightning supports:")
        print("  - DataParallel (dp)")
        print("  - DistributedDataParallel (ddp)")
        print("  - Model sharding with FairScale")
        print("  - TPU training")

# -----------------------------------------------------------------------------
# Section 8: Advanced Lightning Module Example
# -----------------------------------------------------------------------------

class AdvancedLitModel(pl.LightningModule):
    """Advanced Lightning module with more features."""
    
    def __init__(self, input_dim=784, hidden_dim=256, output_dim=10, 
                 learning_rate=1e-3, weight_decay=1e-4):
        super().__init__()
        self.save_hyperparameters()
        
        # Model architecture
        self.model = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, output_dim)
        )
        
        # Loss function
        self.criterion = nn.CrossEntropyLoss()
        
        # Metrics
        self.train_acc = pl.metrics.Accuracy()
        self.val_acc = pl.metrics.Accuracy()
        self.test_acc = pl.metrics.Accuracy()
        
        # For visualization
        self.example_input_array = torch.randn(1, input_dim)
    
    def forward(self, x):
        return self.model(x)
    
    def training_step(self, batch, batch_idx):
        x, y = batch
        x = x.view(x.size(0), -1)
        
        # Forward pass
        logits = self(x)
        loss = self.criterion(logits, y)
        
        # Metrics
        preds = torch.argmax(logits, dim=1)
        acc = self.train_acc(preds, y)
        
        # Logging
        self.log_dict({
            'train_loss': loss,
            'train_acc': acc,
            'learning_rate': self.optimizers().param_groups[0]['lr']
        }, on_step=False, on_epoch=True, prog_bar=True)
        
        return loss
    
    def validation_step(self, batch, batch_idx):
        x, y = batch
        x = x.view(x.size(0), -1)
        
        # Forward pass
        logits = self(x)
        loss = self.criterion(logits, y)
        
        # Metrics
        preds = torch.argmax(logits, dim=1)
        acc = self.val_acc(preds, y)
        
        # Logging
        self.log_dict({
            'val_loss': loss,
            'val_acc': acc
        }, on_epoch=True, prog_bar=True)
        
        return loss
    
    def test_step(self, batch, batch_idx):
        x, y = batch
        x = x.view(x.size(0), -1)
        
        # Forward pass
        logits = self(x)
        loss = self.criterion(logits, y)
        
        # Metrics
        preds = torch.argmax(logits, dim=1)
        acc = self.test_acc(preds, y)
        
        # Logging
        self.log_dict({
            'test_loss': loss,
            'test_acc': acc
        })
        
        return loss
    
    def configure_optimizers(self):
        # Optimizer
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.hparams.learning_rate,
            weight_decay=self.hparams.weight_decay
        )
        
        # Scheduler
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=self.hparams.learning_rate,
            total_steps=self.trainer.estimated_stepping_batches,
            pct_start=0.1
        )
        
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'interval': 'step'
            }
        }
    
    def on_train_epoch_end(self):
        # Custom epoch-end logic
        print(f"Epoch {self.current_epoch} completed. Train acc: {self.train_acc.compute():.4f}")
        self.train_acc.reset()

def demonstrate_advanced_features():
    """Demonstrate advanced Lightning features."""
    print("\nSection 8: Advanced Features")
    print("-" * 70)
    
    # Create advanced model
    model = AdvancedLitModel(
        input_dim=784,
        hidden_dim=256,
        output_dim=10,
        learning_rate=1e-3,
        weight_decay=1e-4
    )
    
    # Create data module
    data_module = MNISTDataModule(batch_size=128, num_workers=0)
    
    # Advanced trainer configuration
    trainer = pl.Trainer(
        max_epochs=5,
        gpus=1 if torch.cuda.is_available() else 0,
        precision=16 if torch.cuda.is_available() else 32,  # Mixed precision
        gradient_clip_val=1.0,  # Gradient clipping
        accumulate_grad_batches=2,  # Gradient accumulation
        auto_lr_find=True,  # Automatic learning rate finding
        auto_scale_batch_size='power',  # Automatic batch size scaling
        profiler='simple',  # Performance profiling
        progress_bar_refresh_rate=20,
        default_root_dir=output_dir
    )
    
    print("Advanced trainer features:")
    print("  - Mixed precision training (16-bit)")
    print("  - Gradient clipping")
    print("  - Gradient accumulation")
    print("  - Automatic learning rate finding")
    print("  - Automatic batch size scaling")
    print("  - Performance profiling")
    
    # Optional: Find optimal learning rate
    # lr_finder = trainer.tuner.lr_find(model, data_module)
    # print(f"Suggested learning rate: {lr_finder.suggestion()}")
    
    # Train model
    trainer.fit(model, data_module)

# -----------------------------------------------------------------------------
# Section 9: Production Best Practices
# -----------------------------------------------------------------------------

def lightning_best_practices():
    """Print PyTorch Lightning best practices."""
    print("\nSection 9: Production Best Practices")
    print("-" * 70)
    
    practices = """
1. Code Organization:
   - Keep model logic in LightningModule
   - Use DataModules for data handling
   - Separate concerns (model, data, training)
   
2. Reproducibility:
   - Use pl.seed_everything()
   - Save hyperparameters with save_hyperparameters()
   - Version control your code and configs
   
3. Monitoring:
   - Use appropriate loggers (TensorBoard, W&B, etc.)
   - Log relevant metrics at appropriate intervals
   - Monitor hardware utilization
   
4. Checkpointing:
   - Save best models based on validation metrics
   - Use ModelCheckpoint callback
   - Consider saving at regular intervals
   
5. Optimization:
   - Use mixed precision training when possible
   - Enable gradient accumulation for large batches
   - Profile your code to find bottlenecks
   
6. Distributed Training:
   - Start with DDP for multi-GPU
   - Consider model sharding for very large models
   - Test on single GPU first
   
7. Hyperparameter Tuning:
   - Use Lightning's built-in tuner
   - Consider integration with Optuna/Ray Tune
   - Log all hyperparameters
"""
    
    print(practices)

# -----------------------------------------------------------------------------
# Main function
# -----------------------------------------------------------------------------

def main():
    """Main function to run all PyTorch Lightning examples."""
    print("=" * 80)
    print("PyTorch Lightning Tutorial")
    print("=" * 80)
    
    # Check PyTorch Lightning installation
    try:
        print(f"PyTorch Lightning version: {pl.__version__}")
    except:
        print("PyTorch Lightning not installed!")
        print("Install with: pip install pytorch-lightning")
        return
    
    # Run demonstrations
    intro_to_lightning()
    demonstrate_lightning_module()
    demonstrate_data_module()
    demonstrate_basic_training()
    demonstrate_callbacks()
    demonstrate_logging()
    demonstrate_distributed_training()
    demonstrate_advanced_features()
    lightning_best_practices()
    
    print(f"\nAll outputs saved to '{output_dir}' directory")
    print("Tutorial complete!")

if __name__ == '__main__':
    main()