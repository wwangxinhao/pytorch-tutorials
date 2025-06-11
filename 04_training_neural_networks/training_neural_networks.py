#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Training Neural Networks in PyTorch: A Comprehensive Guide

This script demonstrates essential techniques for training neural networks in PyTorch,
covering data preparation, training loops, validation, saving/loading models,
hyperparameter tuning, learning rate scheduling, regularization, gradient clipping,
weight initialization, batch normalization, and TensorBoard monitoring.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset, random_split, TensorDataset
from torch.optim.lr_scheduler import StepLR, MultiStepLR, ExponentialLR, CosineAnnealingLR, ReduceLROnPlateau
from torch.nn.utils import clip_grad_norm_, clip_grad_value_
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt
import numpy as np
import os
import time
import math # For math.sqrt in Kaiming init example

# Set random seed for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Create output directory for plots, models, and logs
output_dir = "04_training_neural_networks_outputs"
os.makedirs(output_dir, exist_ok=True)
os.makedirs(os.path.join(output_dir, "runs"), exist_ok=True) # For TensorBoard logs
os.makedirs(os.path.join(output_dir, "saved_models"), exist_ok=True) # For saved models

# -----------------------------------------------------------------------------
# Section 1: Introduction to Neural Network Training (Conceptual - in README)
# -----------------------------------------------------------------------------

def intro_neural_network_training():
    print("\nSection 1: Introduction to Neural Network Training")
    print("-" * 70)
    print("This section is conceptual and detailed in the README.md.")
    print("Covers: Goal, Core Components (Model, Data, Loss, Optimizer), Iterative Process.")

# -----------------------------------------------------------------------------
# Section 2: Preparing Your Data with `Dataset` and `DataLoader`
# -----------------------------------------------------------------------------

class MyCustomDataset(Dataset):
    """Example of a custom Dataset."""
    def __init__(self, num_samples=1000, input_features=10, num_classes=2, transform=None):
        # Generate some random data for demonstration
        self.data = torch.randn(num_samples, input_features)
        self.targets = torch.randint(0, num_classes, (num_samples,))
        self.transform = transform
        print(f"CustomDataset: Created {num_samples} samples.")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        target = self.targets[idx]
        if self.transform:
            sample = self.transform(sample)
        # For this demo, ensure target is also a tensor
        return sample, torch.tensor(target, dtype=torch.long)

def demonstrate_dataset_dataloader():
    print("\nSection 2: Preparing Your Data with `Dataset` and `DataLoader`")
    print("-" * 70)

    # Using the custom dataset
    print("\n--- Custom Dataset Example ---")
    custom_train_dataset = MyCustomDataset(num_samples=100, input_features=5, num_classes=3)
    sample_data, sample_target = custom_train_dataset[0]
    print(f"First sample data shape: {sample_data.shape}, target: {sample_target}")

    # Using DataLoader with the custom dataset
    custom_train_loader = DataLoader(custom_train_dataset, batch_size=32, shuffle=True)
    print(f"Number of batches in custom_train_loader: {len(custom_train_loader)}")
    for i, (batch_data, batch_targets) in enumerate(custom_train_loader):
        print(f"Batch {i+1} data shape: {batch_data.shape}, targets shape: {batch_targets.shape}")
        if i == 0: # Print only first batch details
            break
    
    # Using torchvision for a standard dataset (MNIST)
    print("\n--- torchvision MNIST Dataset and DataLoader Example ---")
    mnist_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)) # MNIST mean and std
    ])
    mnist_train_dataset = torchvision.datasets.MNIST(
        root='./data', train=True, download=True, transform=mnist_transform
    )
    mnist_train_loader = DataLoader(mnist_train_dataset, batch_size=64, shuffle=True)
    print(f"Number of samples in MNIST training set: {len(mnist_train_dataset)}")
    print(f"Number of batches in MNIST train_loader: {len(mnist_train_loader)}")
    mnist_batch_data, mnist_batch_targets = next(iter(mnist_train_loader))
    print(f"MNIST first batch data shape: {mnist_batch_data.shape}, targets shape: {mnist_batch_targets.shape}")
    print("`Dataset` manages data samples, `DataLoader` provides batches for training.")

# -----------------------------------------------------------------------------
# Helper: Define a Simple Neural Network for subsequent sections
# -----------------------------------------------------------------------------
class SimpleNN(nn.Module):
    def __init__(self, input_size=28*28, hidden_size=128, num_classes=10, use_dropout=False, use_bn=False):
        super(SimpleNN, self).__init__()
        self.use_dropout = use_dropout
        self.use_bn = use_bn
        
        self.fc1 = nn.Linear(input_size, hidden_size)
        if self.use_bn:
            self.bn1 = nn.BatchNorm1d(hidden_size)
        self.relu = nn.ReLU()
        if self.use_dropout:
            self.dropout = nn.Dropout(0.5) # Example dropout rate
        self.fc2 = nn.Linear(hidden_size, num_classes)
    
    def forward(self, x):
        x = x.view(x.size(0), -1)  # Flatten the input
        x = self.fc1(x)
        if self.use_bn:
            x = self.bn1(x)
        x = self.relu(x)
        if self.use_dropout:
            x = self.dropout(x)
        x = self.fc2(x)
        return x

# -----------------------------------------------------------------------------
# Section 3: The Essential Training Loop
# -----------------------------------------------------------------------------
def essential_training_loop_demo():
    print("\nSection 3: The Essential Training Loop")
    print("-" * 70)

    # Dummy data for demonstration
    dummy_inputs = torch.randn(64, 28*28).to(device) # Batch of 64 flattened images
    dummy_targets = torch.randint(0, 10, (64,)).to(device) # 64 labels for 10 classes

    model = SimpleNN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    print("Running one iteration of the training loop...")
    # 1. Set model to training mode
    model.train()

    # 2. Zeroing gradients (typically at start of batch loop)
    optimizer.zero_grad()

    # 3. Forward pass: Getting predictions
    outputs = model(dummy_inputs)
    print(f"  Output shape: {outputs.shape}")

    # 4. Calculating the loss
    loss = criterion(outputs, dummy_targets)
    print(f"  Calculated loss: {loss.item():.4f}")

    # 5. Backward pass: Computing gradients
    loss.backward()
    print(f"  Gradients computed (e.g., model.fc1.weight.grad is not None: {model.fc1.weight.grad is not None})")

    # 6. Optimizer step: Updating weights
    optimizer.step()
    print(f"  Optimizer step taken (weights updated).")
    print("This forms one iteration. A full epoch repeats this for all batches.")

# -----------------------------------------------------------------------------
# Section 4: Validation: Evaluating Model Performance
# -----------------------------------------------------------------------------

def demonstrate_validation():
    print("\nSection 4: Validation: Evaluating Model Performance")
    print("-" * 70)
    
    # Using MNIST data for a more realistic validation scenario
    _, val_loader, _ = get_mnist_loaders() # Get a validation loader

    model = SimpleNN().to(device)
    criterion = nn.CrossEntropyLoss()
    # Assume model has been trained for some iterations or loaded
    # For demo, we use an untrained model here.

    print("Running one validation epoch...")
    # 1. Set model to evaluation mode
    model.eval()
    val_loss = 0.0
    correct_predictions = 0
    total_samples = 0

    # 2. Disable gradient computation
    with torch.no_grad():
        for inputs, targets in val_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            
            val_loss += loss.item() * inputs.size(0)
            _, predicted_classes = outputs.max(1)
            total_samples += targets.size(0)
            correct_predictions += predicted_classes.eq(targets).sum().item()
            if len(val_losses_hist) < 1: # Just for first batch print
                 print(f"  Validation batch: Loss={loss.item():.4f}")

    epoch_loss = val_loss / total_samples
    epoch_accuracy = correct_predictions / total_samples
    print(f"Validation Results: Average Loss: {epoch_loss:.4f}, Accuracy: {epoch_accuracy*100:.2f}%")
    print("`model.eval()` and `torch.no_grad()` are crucial for correct validation.")
    # K-Fold Cross-Validation is a more robust technique for smaller datasets (conceptual here)
    print("K-Fold Cross-Validation (conceptual): Split data into K folds, train K models, average results.")

# -----------------------------------------------------------------------------
# Section 5: Saving and Loading Models
# -----------------------------------------------------------------------------

def demonstrate_saving_loading_models():
    print("\nSection 5: Saving and Loading Models")
    print("-" * 70)

    model_to_save = SimpleNN().to(device)
    # Simulate some training
    optimizer = optim.Adam(model_to_save.parameters(), lr=0.001)
    dummy_input = torch.randn(10, 28*28).to(device)
    dummy_target = torch.randint(0,10,(10,)).to(device)
    criterion = nn.CrossEntropyLoss()
    for _ in range(2): # Few dummy steps
        optimizer.zero_grad()
        loss = criterion(model_to_save(dummy_input), dummy_target)
        loss.backward()
        optimizer.step()
        
    model_path = os.path.join(output_dir, "saved_models", "simple_nn_statedict.pth")
    checkpoint_path = os.path.join(output_dir, "saved_models", "checkpoint.pth")

    # --- Saving and Loading State Dictionary (Recommended) ---
    print("\n--- Saving and Loading Model State Dictionary ---")
    torch.save(model_to_save.state_dict(), model_path)
    print(f"Model state_dict saved to: {model_path}")

    # Load the state_dict
    model_loaded_state_dict = SimpleNN().to(device) # Create a new instance of the model
    model_loaded_state_dict.load_state_dict(torch.load(model_path))
    model_loaded_state_dict.eval() # Set to evaluation mode
    print("Model loaded from state_dict successfully.")
    # You can now use model_loaded_state_dict for inference

    # --- Saving and Loading Checkpoints (for resuming training) ---
    print("\n--- Saving and Loading Checkpoints ---")
    epoch = 5
    current_loss = 0.123
    checkpoint = {
        'epoch': epoch + 1, # Save next epoch to start from
        'model_state_dict': model_to_save.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': current_loss,
    }
    torch.save(checkpoint, checkpoint_path)
    print(f"Checkpoint saved to: {checkpoint_path}")

    # Load from checkpoint
    model_for_resume = SimpleNN().to(device)
    optimizer_for_resume = optim.Adam(model_for_resume.parameters(), lr=0.0001) # LR might be different or saved too

    loaded_checkpoint = torch.load(checkpoint_path)
    model_for_resume.load_state_dict(loaded_checkpoint['model_state_dict'])
    optimizer_for_resume.load_state_dict(loaded_checkpoint['optimizer_state_dict'])
    start_epoch = loaded_checkpoint['epoch']
    previous_loss = loaded_checkpoint['loss']
    model_for_resume.train() # Set to train mode to resume training
    print(f"Checkpoint loaded. Resuming from epoch {start_epoch}, previous loss: {previous_loss:.4f}")
    print("Always save `state_dict` for portability and flexibility.")

# -----------------------------------------------------------------------------
# Section 6: Hyperparameter Tuning Strategies (Conceptual demonstration)
# -----------------------------------------------------------------------------

def demonstrate_hyperparameter_tuning_concepts():
    print("\nSection 6: Hyperparameter Tuning Strategies")
    print("-" * 70)
    print("Hyperparameters are settings not learned during training (e.g., learning rate, batch size).")
    # Dummy training function for hyperparameter tuning demo
    def dummy_train_for_hparam(lr, hidden_size, batch_size):
        print(f"  Tuning: lr={lr}, hidden_size={hidden_size}, batch_size={batch_size}")
        # model = SimpleNN(hidden_size=hidden_size) # ... train ... validate ...
        time.sleep(0.1) # Simulate training time
        # Return a dummy validation accuracy
        return np.random.rand() 

    print("\n--- Manual/Grid Search Example (Conceptual) ---")
    learning_rates = [0.01, 0.005]
    hidden_sizes = [64, 128]
    batch_sizes = [32, 64]
    best_accuracy = -1
    best_hparams = {}

    for lr_test in learning_rates:
        for hs_test in hidden_sizes:
            for bs_test in batch_sizes:
                val_acc = dummy_train_for_hparam(lr_test, hs_test, bs_test)
                print(f"    Resulting val_acc: {val_acc:.4f}")
                if val_acc > best_accuracy:
                    best_accuracy = val_acc
                    best_hparams = {'lr': lr_test, 'hidden_size': hs_test, 'batch_size': bs_test}
    print(f"Best validation accuracy from manual search: {best_accuracy:.4f} with params: {best_hparams}")
    print("\n--- Advanced Tools (Conceptual) ---")
    print("Optuna, Ray Tune, Weights & Biases Sweeps offer more sophisticated automated tuning.")
    print("Example Optuna (conceptual - requires `pip install optuna`):")
    print("  def objective(trial):")
    print("      lr = trial.suggest_float('lr', 1e-4, 1e-1, log=True)")
    print("      hidden = trial.suggest_int('hidden', 32, 256)")
    print("      # ... train model with these params ...")
    print("      return validation_accuracy")
    print("  study = optuna.create_study(direction='maximize')")
    print("  study.optimize(objective, n_trials=50)")

# -----------------------------------------------------------------------------
# Section 7: Learning Rate Scheduling
# -----------------------------------------------------------------------------

def demonstrate_lr_scheduling():
    print("\nSection 7: Learning Rate Scheduling")
    print("-" * 70)
    print("Adjusting learning rate during training can improve convergence and performance.")

    model = SimpleNN(hidden_size=32).to(device) # Smaller model for quick demo
    optimizer = optim.SGD(model.parameters(), lr=0.1)
    num_epochs_lr_demo = 15
    dummy_loader = DataLoader(TensorDataset(torch.randn(100, 28*28), torch.randint(0,10,(100,))), batch_size=10)
    criterion = nn.CrossEntropyLoss()

    schedulers_to_test = {
        "StepLR (step=5, gamma=0.5)": StepLR(optimizer, step_size=5, gamma=0.5),
        "MultiStepLR (milestones=[5,10], gamma=0.1)": MultiStepLR(optimizer, milestones=[5,10], gamma=0.1),
        "ExponentialLR (gamma=0.85)": ExponentialLR(optimizer, gamma=0.85),
        "CosineAnnealingLR (T_max=15)": CosineAnnealingLR(optimizer, T_max=num_epochs_lr_demo),
        "ReduceLROnPlateau (patience=2)": ReduceLROnPlateau(optimizer, 'min', patience=2, factor=0.5, verbose=True)
    }

    for name, scheduler in schedulers_to_test.items():
        print(f"\n--- Testing Scheduler: {name} ---")
        # Reset optimizer for each scheduler test (important for fresh LR)
        optimizer = optim.SGD(model.parameters(), lr=0.1) # Re-initialize optimizer with base LR
        # Attach the current scheduler to the re-initialized optimizer
        if isinstance(scheduler, ReduceLROnPlateau): 
            # ReduceLROnPlateau needs its own instance with the new optimizer
            current_scheduler_instance = ReduceLROnPlateau(optimizer, 'min', patience=2, factor=0.5, verbose=False)
        else:
            # Other schedulers can be re-created or re-assigned if they don't store too much state related to optimizer id
            # For simplicity, re-creating scheduler attached to the new optimizer
            if isinstance(scheduler, StepLR): current_scheduler_instance = StepLR(optimizer, step_size=5, gamma=0.5)
            elif isinstance(scheduler, MultiStepLR): current_scheduler_instance = MultiStepLR(optimizer, milestones=[5,10], gamma=0.1)
            elif isinstance(scheduler, ExponentialLR): current_scheduler_instance = ExponentialLR(optimizer, gamma=0.85)
            elif isinstance(scheduler, CosineAnnealingLR): current_scheduler_instance = CosineAnnealingLR(optimizer, T_max=num_epochs_lr_demo)
        
        lr_history = []
        dummy_val_loss = 10.0 # For ReduceLROnPlateau
        for epoch in range(num_epochs_lr_demo):
            lr_history.append(optimizer.param_groups[0]['lr'])
            # Dummy training step
            for _ in dummy_loader: optimizer.zero_grad(); model(torch.randn(10, 28*28).to(device)).sum().backward(); optimizer.step(); break
            
            if isinstance(current_scheduler_instance, ReduceLROnPlateau):
                current_scheduler_instance.step(dummy_val_loss) # Simulate validation loss
                dummy_val_loss *= 0.95 # Simulate improving loss, then plateau
                if epoch > 5 : dummy_val_loss = 2.0 # Simulate plateau
            else:
                current_scheduler_instance.step()
        
        plt.plot(lr_history, label=name)

    plt.title("Learning Rate Schedules")
    plt.xlabel("Epoch")
    plt.ylabel("Learning Rate")
    plt.legend(loc='upper right', bbox_to_anchor=(1.0, 1.0), fontsize='small')
    plt.grid(True)
    plt.tight_layout()
    lr_plot_path = os.path.join(output_dir, "learning_rate_schedules.png")
    plt.savefig(lr_plot_path)
    plt.close()
    print(f"Learning rate schedules plot saved to '{lr_plot_path}'")

# -----------------------------------------------------------------------------
# Section 8: Regularization Techniques to Prevent Overfitting
# -----------------------------------------------------------------------------

def demonstrate_regularization():
    print("\nSection 8: Regularization Techniques to Prevent Overfitting")
    print("-" * 70)

    # --- L2 Regularization (Weight Decay) ---
    print("\n--- L2 Regularization (Weight Decay) ---")
    model_wd = SimpleNN().to(device)
    # Add weight_decay to the optimizer
    optimizer_wd = optim.Adam(model_wd.parameters(), lr=0.001, weight_decay=1e-4) # Common value for weight_decay
    print(f"Optimizer with weight_decay (L2 penalty): {optimizer_wd}")
    # During optimizer.step(), this penalty is effectively added to the loss for weights.

    # --- Dropout --- 
    print("\n--- Dropout ---")
    model_dropout = SimpleNN(use_dropout=True).to(device)
    print("Model with Dropout layer:")
    print(model_dropout)
    # During model.train(), dropout randomly zeros elements.
    # During model.eval(), dropout is disabled (acts as identity).
    model_dropout.train()
    dummy_input_reg = torch.randn(5, 28*28).to(device)
    output_train = model_dropout(dummy_input_reg)
    print(f"Output with dropout (train mode, some elements might be zeroed in hidden layer):\n{output_train[0,:5]}")
    model_dropout.eval()
    output_eval = model_dropout(dummy_input_reg)
    print(f"Output with dropout (eval mode, no zeroing):\n{output_eval[0,:5]}")

    # --- Early Stopping (Class defined and used in complete_training_pipeline_example) ---
    print("\n--- Early Stopping ---")
    print("Stops training if validation performance doesn't improve for 'patience' epochs.")
    print("EarlyStopping class will be shown in the complete pipeline example.")
    print("Data Augmentation is another key regularization technique (often in Dataset/DataLoader). Example: torchvision.transforms.")

# -----------------------------------------------------------------------------
# Section 9: Gradient Clipping
# -----------------------------------------------------------------------------

def demonstrate_gradient_clipping():
    print("\nSection 9: Gradient Clipping")
    print("-" * 70)
    print("Prevents exploding gradients by capping their norm or value.")

    model_gc = SimpleNN().to(device)
    optimizer_gc = optim.SGD(model_gc.parameters(), lr=0.01)
    criterion_gc = nn.MSELoss()
    dummy_input_gc = torch.randn(5, 28*28).to(device)
    dummy_target_gc = torch.randn(5, 10).to(device)

    optimizer_gc.zero_grad()
    outputs_gc = model_gc(dummy_input_gc)
    loss_gc = criterion_gc(outputs_gc, dummy_target_gc)
    loss_gc.backward()
    
    # Example: Print norm of gradients before clipping for one layer
    original_grad_norm = model_gc.fc1.weight.grad.norm().item()
    print(f"Original grad norm for fc1.weight: {original_grad_norm:.4f}")

    # Clip gradient norm (applied to all parameters in model_gc.parameters())
    max_norm = 1.0
    total_norm_clipped = clip_grad_norm_(model_gc.parameters(), max_norm=max_norm)
    print(f"Total norm of gradients after clipping by norm to {max_norm}: {total_norm_clipped:.4f}")
    clipped_grad_norm = model_gc.fc1.weight.grad.norm().item()
    print(f"Clipped grad norm for fc1.weight: {clipped_grad_norm:.4f}")

    # Clip gradient value (applied element-wise)
    # Re-compute gradients for value clipping demo
    optimizer_gc.zero_grad()
    outputs_gc = model_gc(dummy_input_gc)
    loss_gc = criterion_gc(outputs_gc, dummy_target_gc)
    loss_gc.backward()
    clip_val = 0.1
    clip_grad_value_(model_gc.parameters(), clip_value=clip_val)
    print(f"Gradients after clipping values to +/- {clip_val}. Check fc1.weight.grad min/max:")
    print(f"  Min grad value: {model_gc.fc1.weight.grad.min().item():.4f}, Max: {model_gc.fc1.weight.grad.max().item():.4f}")
    # optimizer_gc.step() # Then update weights
    print("Gradient clipping is applied after .backward() and before .step().")

# -----------------------------------------------------------------------------
# Section 10: Weight Initialization Strategies
# -----------------------------------------------------------------------------

def demonstrate_weight_initialization():
    print("\nSection 10: Weight Initialization Strategies")
    print("-" * 70)
    print("Proper weight initialization helps with convergence and prevents vanishing/exploding gradients.")

    class ModelWithInit(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc1 = nn.Linear(10, 20) # For Xavier/Glorot demo
            self.relu = nn.ReLU()
            self.fc2 = nn.Linear(20, 5)  # For Kaiming/He demo
            self._initialize_weights()
        
        def _initialize_weights(self):
            print("  Initializing weights...")
            # Xavier Uniform for fc1 (assuming it might be followed by Tanh/Sigmoid, though we use ReLU here)
            nn.init.xavier_uniform_(self.fc1.weight)
            if self.fc1.bias is not None:
                nn.init.zeros_(self.fc1.bias)
            print(f"    fc1.weight after Xavier init (sample): {self.fc1.weight.data[0,0].item():.4f}")

            # Kaiming Normal for fc2 (because it follows ReLU)
            nn.init.kaiming_normal_(self.fc2.weight, mode='fan_in', nonlinearity='relu')
            if self.fc2.bias is not None:
                nn.init.constant_(self.fc2.bias, 0.1) # Example: small constant for bias
            print(f"    fc2.weight after Kaiming init (sample): {self.fc2.weight.data[0,0].item():.4f}")
            print(f"    fc2.bias after const init (sample): {self.fc2.bias.data[0].item():.4f}")

        def forward(self, x):
            x = torch.tanh(self.fc1(x)) # Using Tanh here to match Xavier rationale conceptually
            x = self.relu(self.fc2(x))
            return x

    model_init_demo = ModelWithInit().to(device)
    # Alternatively, use model.apply(init_function)
    def weights_init_apply_func(m):
        if isinstance(m, nn.Linear):
            print(f"  Applying custom init to Linear layer: {m}")
            nn.init.normal_(m.weight, mean=0, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
    
    model_apply_init = SimpleNN(hidden_size=16).to(device)
    print("\nApplying init function with model.apply():")
    model_apply_init.apply(weights_init_apply_func)
    print(f"  model_apply_init.fc1.weight (sample after apply): {model_apply_init.fc1.weight.data[0,0].item():.4f}")

# -----------------------------------------------------------------------------
# Section 11: Batch Normalization
# -----------------------------------------------------------------------------

def demonstrate_batch_normalization():
    print("\nSection 11: Batch Normalization")
    print("-" * 70)
    print("Batch Normalization normalizes activations, stabilizes and accelerates training.")

    model_with_bn = SimpleNN(use_bn=True).to(device)
    print("Model with Batch Normalization layer:")
    print(model_with_bn)

    dummy_input_bn = torch.randn(5, 28*28).to(device)

    # Behavior in train() mode
    model_with_bn.train() # Set to training mode
    output_bn_train = model_with_bn(dummy_input_bn)
    print(f"Output with BatchNorm (train mode) shape: {output_bn_train.shape}")
    # In train mode, BN uses batch statistics and updates its running mean/var.
    print(f"  Running mean of bn1 after one forward pass (train): {model_with_bn.bn1.running_mean[0].item():.4f}")

    # Behavior in eval() mode
    model_with_bn.eval() # Set to evaluation mode
    output_bn_eval = model_with_bn(dummy_input_bn)
    print(f"Output with BatchNorm (eval mode) shape: {output_bn_eval.shape}")
    # In eval mode, BN uses its computed running mean/var and does not update them.
    print("`model.train()` and `model.eval()` are crucial for BatchNorm to work correctly.")

# -----------------------------------------------------------------------------
# Section 12: Monitoring Training with TensorBoard
# -----------------------------------------------------------------------------

def demonstrate_tensorboard_logging():
    print("\nSection 12: Monitoring Training with TensorBoard")
    print("-" * 70)
    print("TensorBoard allows visualization of metrics, model graph, histograms, etc.")

    # Create a SummaryWriter instance
    # Logs will be saved to '04_training_neural_networks_outputs/runs/tensorboard_demo_run'
    log_path = os.path.join(output_dir, "runs", "tensorboard_demo_run")
    writer = SummaryWriter(log_dir=log_path)
    print(f"TensorBoard logs will be written to: {log_path}")
    print(f"To view TensorBoard, run: tensorboard --logdir={os.path.abspath(output_dir)}/runs")

    # Dummy model and data for logging
    tb_model = SimpleNN(hidden_size=10).to(device)
    tb_optimizer = optim.Adam(tb_model.parameters())
    tb_criterion = nn.CrossEntropyLoss()
    dummy_data_tb = torch.randn(16, 28*28).to(device)
    dummy_targets_tb = torch.randint(0,10,(16,)).to(device)

    for epoch in range(5): # Simulate 5 epochs
        # Dummy training step
        tb_optimizer.zero_grad()
        outputs = tb_model(dummy_data_tb)
        loss = tb_criterion(outputs, dummy_targets_tb)
        loss.backward()
        tb_optimizer.step()
        
        accuracy = (outputs.max(1)[1] == dummy_targets_tb).float().mean().item()

        # Logging scalars
        writer.add_scalar('Loss/train_dummy', loss.item(), global_step=epoch)
        writer.add_scalar('Accuracy/train_dummy', accuracy, global_step=epoch)
        writer.add_scalar('LearningRate_dummy', tb_optimizer.param_groups[0]['lr'], global_step=epoch)

        # Logging histograms of weights and gradients (for one layer)
        writer.add_histogram('fc1.weights', tb_model.fc1.weight, global_step=epoch)
        if tb_model.fc1.weight.grad is not None:
            writer.add_histogram('fc1.gradients', tb_model.fc1.weight.grad, global_step=epoch)
        
        # Add model graph (only once typically)
        if epoch == 0:
            writer.add_graph(tb_model, dummy_data_tb) # Log model graph
    
    # Add hparams example (conceptual)
    # hparams = {'lr': 0.01, 'batch_size': 16}
    # metrics = {'hparam/accuracy': 0.75, 'hparam/loss': 0.5}
    # writer.add_hparams(hparams, metrics)

    writer.close() # Important to close the writer
    print("TensorBoard logging demonstrated. Check the specified log directory.")

# -----------------------------------------------------------------------------
# Section 13: A Complete Training Pipeline Example
# (This will integrate many of the above concepts)
# -----------------------------------------------------------------------------

# Helper: Load MNIST data (can be defined globally or passed around)
# Re-defining for clarity within this section, though it's similar to earlier one.
def get_mnist_loaders(batch_size=64, validation_split=0.1):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    full_train_dataset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    test_dataset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)

    num_train = len(full_train_dataset)
    val_size = int(validation_split * num_train)
    train_size = num_train - val_size
    
    train_subset, val_subset = random_split(full_train_dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    return train_loader, val_loader, test_loader

# Helper: EarlyStopping class
class EarlyStopping:
    def __init__(self, patience=7, delta=0, path='checkpoint.pt', verbose=True):
        self.patience = patience
        self.delta = delta
        self.path = path
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf

    def __call__(self, val_loss, model):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.verbose:
                print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), self.path)
        self.val_loss_min = val_loss

def complete_training_pipeline_example():
    print("\nSection 13: A Complete Training Pipeline Example")
    print("-" * 70)

    # --- Configuration ---
    LEARNING_RATE = 0.001
    BATCH_SIZE = 128
    NUM_EPOCHS = 2 # Low for demo, typically 10-100+
    HIDDEN_SIZE = 256
    INPUT_SIZE = 28*28
    NUM_CLASSES = 10
    PATIENCE_EARLY_STOP = 5
    WEIGHT_DECAY = 1e-5
    CLIP_GRAD_NORM = 1.0

    run_name = f"mnist_run_{time.strftime('%Y%m%d-%H%M%S')}"
    tb_writer = SummaryWriter(log_dir=os.path.join(output_dir, "runs", run_name))
    best_model_path = os.path.join(output_dir, "saved_models", f"{run_name}_best_model.pth")

    # --- 1. Data Loading ---
    print("\n--- 1. Loading Data ---")
    train_loader, val_loader, test_loader = get_mnist_loaders(batch_size=BATCH_SIZE)
    print(f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}, Test batches: {len(test_loader)}")

    # --- 2. Model Definition ---
    print("\n--- 2. Defining Model ---")
    model = SimpleNN(INPUT_SIZE, HIDDEN_SIZE, NUM_CLASSES, use_dropout=True, use_bn=True).to(device)
    # Weight Initialization (Example - can be more sophisticated or a function)
    for m in model.modules():
        if isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
            if m.bias is not None: nn.init.zeros_(m.bias)
    print(model)

    # --- 3. Loss Function and Optimizer ---
    print("\n--- 3. Defining Loss, Optimizer, Scheduler ---")
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    # Learning Rate Scheduler
    scheduler = ReduceLROnPlateau(optimizer, 'min', patience=3, factor=0.5, verbose=True)

    # Early Stopping Initialization
    early_stopper = EarlyStopping(patience=PATIENCE_EARLY_STOP, path=best_model_path, verbose=True)

    # --- 4. Training Loop ---
    print("\n--- 4. Starting Training Loop ---")
    training_start_time = time.time()
    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': [], 'lr': []}

    for epoch in range(NUM_EPOCHS):
        epoch_start_time = time.time()
        model.train() # Set model to training mode
        current_epoch_train_loss = 0.0
        current_epoch_train_correct = 0
        current_epoch_train_total = 0

        for batch_idx, (inputs, targets) in enumerate(train_loader):
            inputs, targets = inputs.to(device), targets.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            
            # Gradient Clipping
            if CLIP_GRAD_NORM > 0:
                clip_grad_norm_(model.parameters(), max_norm=CLIP_GRAD_NORM)
            
            optimizer.step()

            current_epoch_train_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            current_epoch_train_total += targets.size(0)
            current_epoch_train_correct += predicted.eq(targets).sum().item()

            if (batch_idx + 1) % 100 == 0: # Log every 100 batches
                batch_loss = (loss.item() * inputs.size(0)) / inputs.size(0) # Avg loss for this batch
                batch_acc = predicted.eq(targets).sum().item() / targets.size(0)
                print(f'  Epoch {epoch+1}/{NUM_EPOCHS} | Batch {batch_idx+1}/{len(train_loader)} | Train Loss: {batch_loss:.4f} | Train Acc: {batch_acc*100:.2f}%')

        avg_epoch_train_loss = current_epoch_train_loss / current_epoch_train_total
        avg_epoch_train_acc = current_epoch_train_correct / current_epoch_train_total
        history['train_loss'].append(avg_epoch_train_loss)
        history['train_acc'].append(avg_epoch_train_acc)
        tb_writer.add_scalar('Loss/Train', avg_epoch_train_loss, epoch)
        tb_writer.add_scalar('Accuracy/Train', avg_epoch_train_acc, epoch)

        # Validation
        model.eval() # Set model to evaluation mode
        current_epoch_val_loss = 0.0
        current_epoch_val_correct = 0
        current_epoch_val_total = 0
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                current_epoch_val_loss += loss.item() * inputs.size(0)
                _, predicted = outputs.max(1)
                current_epoch_val_total += targets.size(0)
                current_epoch_val_correct += predicted.eq(targets).sum().item()
        
        avg_epoch_val_loss = current_epoch_val_loss / current_epoch_val_total
        avg_epoch_val_acc = current_epoch_val_correct / current_epoch_val_total
        history['val_loss'].append(avg_epoch_val_loss)
        history['val_acc'].append(avg_epoch_val_acc)
        tb_writer.add_scalar('Loss/Validation', avg_epoch_val_loss, epoch)
        tb_writer.add_scalar('Accuracy/Validation', avg_epoch_val_acc, epoch)

        # LR Scheduler Step
        current_lr = optimizer.param_groups[0]['lr']
        history['lr'].append(current_lr)
        tb_writer.add_scalar('LearningRate', current_lr, epoch)
        scheduler.step(avg_epoch_val_loss) # For ReduceLROnPlateau

        epoch_duration = time.time() - epoch_start_time
        print(f"Epoch {epoch+1}/{NUM_EPOCHS} Summary: Duration: {epoch_duration:.2f}s")
        print(f"  Train Loss: {avg_epoch_train_loss:.4f}, Train Acc: {avg_epoch_train_acc*100:.2f}% | Val Loss: {avg_epoch_val_loss:.4f}, Val Acc: {avg_epoch_val_acc*100:.2f}% | LR: {current_lr:.6f}")

        # Early Stopping Check
        early_stopper(avg_epoch_val_loss, model)
        if early_stopper.early_stop:
            print("Early stopping triggered.")
            break
            
    training_duration = time.time() - training_start_time
    print(f"\n--- Training Finished in {training_duration:.2f} seconds ---")

    # Load best model from early stopping
    print(f"Loading best model from: {best_model_path}")
    model.load_state_dict(torch.load(best_model_path))

    # --- 5. Final Evaluation on Test Set ---
    print("\n--- 5. Evaluating on Test Set with Best Model ---")
    model.eval()
    test_loss = 0.0
    test_correct = 0
    test_total = 0
    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            test_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            test_total += targets.size(0)
            test_correct += predicted.eq(targets).sum().item()
    
    avg_test_loss = test_loss / test_total
    avg_test_acc = test_correct / test_total
    print(f"Test Set Performance: Loss: {avg_test_loss:.4f}, Accuracy: {avg_test_acc*100:.2f}%")
    tb_writer.add_hparams(
        { # Dictionary of hparams
            'lr': LEARNING_RATE, 'batch_size': BATCH_SIZE, 'hidden_size': HIDDEN_SIZE,
            'weight_decay': WEIGHT_DECAY, 'clip_norm': CLIP_GRAD_NORM
        },
        { # Dictionary of metrics
            'hparam/test_accuracy': avg_test_acc,
            'hparam/test_loss': avg_test_loss,
            'hparam/best_val_accuracy': early_stopper.best_score * -1 if early_stopper.best_score else 0 # score is -val_loss
        }
    )

    tb_writer.close()
    print(f"TensorBoard logs for this run are in: {os.path.join(output_dir, 'runs', run_name)}")
    
    # Plot training history
    plt.figure(figsize=(12, 10))
    plt.subplot(3,1,1)
    plt.plot(history['train_loss'], label='Train Loss')
    plt.plot(history['val_loss'], label='Validation Loss')
    plt.title('Loss over Epochs'); plt.xlabel('Epoch'); plt.ylabel('Loss'); plt.legend(); plt.grid(True)
    plt.subplot(3,1,2)
    plt.plot(history['train_acc'], label='Train Accuracy')
    plt.plot(history['val_acc'], label='Validation Accuracy')
    plt.title('Accuracy over Epochs'); plt.xlabel('Epoch'); plt.ylabel('Accuracy'); plt.legend(); plt.grid(True)
    plt.subplot(3,1,3)
    plt.plot(history['lr'], label='Learning Rate')
    plt.title('Learning Rate over Epochs'); plt.xlabel('Epoch'); plt.ylabel('Learning Rate'); plt.legend(); plt.grid(True)
    plt.tight_layout()
    history_plot_path = os.path.join(output_dir, f"{run_name}_training_history.png")
    plt.savefig(history_plot_path)
    plt.close()
    print(f"Training history plot saved to {history_plot_path}")

# -----------------------------------------------------------------------------
# Main function to run selected demonstrations or full pipeline
# -----------------------------------------------------------------------------

def main():
    """Main function to run tutorial sections."""
    print("=" * 80)
    print("PyTorch Training Neural Networks Tutorial")
    print("=" * 80)

    # Run individual demonstrations
    intro_neural_network_training() # Section 1 (Conceptual)
    demonstrate_dataset_dataloader()    # Section 2
    essential_training_loop_demo()    # Section 3
    demonstrate_validation()          # Section 4
    demonstrate_saving_loading_models() # Section 5
    demonstrate_hyperparameter_tuning_concepts() # Section 6
    demonstrate_lr_scheduling()       # Section 7
    demonstrate_regularization()      # Section 8
    demonstrate_gradient_clipping()   # Section 9
    demonstrate_weight_initialization() # Section 10
    demonstrate_batch_normalization() # Section 11
    demonstrate_tensorboard_logging() # Section 12
    
    # Run the complete pipeline example
    complete_training_pipeline_example() # Section 13

    print("\nTutorial complete! Outputs are in '04_training_neural_networks_outputs' directory.")

if __name__ == '__main__':
    main() 