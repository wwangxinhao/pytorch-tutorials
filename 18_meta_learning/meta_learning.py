"""
Tutorial 18: Meta-Learning and Few-Shot Learning
================================================

This tutorial explores meta-learning and few-shot learning techniques,
including MAML, Prototypical Networks, and Matching Networks.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from typing import Tuple, List, Dict
import matplotlib.pyplot as plt
from collections import OrderedDict
import copy
import random

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")
print()

# Example 1: Meta-Learning Basics
print("Example 1: Meta-Learning Concepts")
print("=" * 50)

class SimpleClassifier(nn.Module):
    """Simple neural network for few-shot classification"""
    def __init__(self, input_size=84*84*3, hidden_size=128, output_size=5):
        super().__init__()
        self.features = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU()
        )
        self.classifier = nn.Linear(hidden_size, output_size)
        
    def forward(self, x):
        x = x.view(x.size(0), -1)
        features = self.features(x)
        out = self.classifier(features)
        return out, features

# Generate synthetic few-shot task
def generate_task(n_way=5, k_shot=5, q_queries=15, feature_dim=100):
    """Generate a synthetic N-way K-shot task"""
    # Generate class prototypes
    prototypes = torch.randn(n_way, feature_dim)
    
    support_set = []
    support_labels = []
    query_set = []
    query_labels = []
    
    for class_idx in range(n_way):
        # Support set
        class_samples = prototypes[class_idx] + 0.3 * torch.randn(k_shot, feature_dim)
        support_set.append(class_samples)
        support_labels.extend([class_idx] * k_shot)
        
        # Query set
        class_queries = prototypes[class_idx] + 0.3 * torch.randn(q_queries, feature_dim)
        query_set.append(class_queries)
        query_labels.extend([class_idx] * q_queries)
    
    support_set = torch.cat(support_set, dim=0)
    query_set = torch.cat(query_set, dim=0)
    support_labels = torch.tensor(support_labels)
    query_labels = torch.tensor(query_labels)
    
    return support_set, support_labels, query_set, query_labels

# Visualize a few-shot task
support_x, support_y, query_x, query_y = generate_task(n_way=3, k_shot=5, q_queries=10, feature_dim=2)

plt.figure(figsize=(10, 6))
colors = ['red', 'blue', 'green']
markers = ['o', 's', '^']

for class_idx in range(3):
    # Plot support set
    support_mask = support_y == class_idx
    plt.scatter(support_x[support_mask, 0], support_x[support_mask, 1], 
               c=colors[class_idx], marker=markers[class_idx], s=100, 
               label=f'Class {class_idx} (support)', edgecolors='black')
    
    # Plot query set
    query_mask = query_y == class_idx
    plt.scatter(query_x[query_mask, 0], query_x[query_mask, 1], 
               c=colors[class_idx], marker=markers[class_idx], s=50, 
               label=f'Class {class_idx} (query)', alpha=0.5)

plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.title('Few-Shot Learning Task (3-way 5-shot)')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()

print("Few-shot task generated:")
print(f"Support set shape: {support_x.shape}")
print(f"Query set shape: {query_x.shape}")
print()

# Example 2: Model-Agnostic Meta-Learning (MAML)
print("Example 2: Model-Agnostic Meta-Learning (MAML)")
print("=" * 50)

class MAML:
    """Model-Agnostic Meta-Learning implementation"""
    def __init__(self, model, inner_lr=0.01, meta_lr=0.001, inner_steps=5):
        self.model = model
        self.inner_lr = inner_lr
        self.meta_lr = meta_lr
        self.inner_steps = inner_steps
        self.meta_optimizer = optim.Adam(self.model.parameters(), lr=meta_lr)
        
    def inner_loop(self, support_x, support_y, fast_weights=None):
        """Inner loop adaptation on support set"""
        if fast_weights is None:
            fast_weights = OrderedDict(self.model.named_parameters())
        
        for step in range(self.inner_steps):
            # Forward pass with fast weights
            logits = self.functional_forward(support_x, fast_weights)
            loss = F.cross_entropy(logits, support_y)
            
            # Compute gradients
            grads = torch.autograd.grad(loss, fast_weights.values(), create_graph=True)
            
            # Update fast weights
            fast_weights = OrderedDict(
                (name, param - self.inner_lr * grad)
                for (name, param), grad in zip(fast_weights.items(), grads)
            )
        
        return fast_weights
    
    def functional_forward(self, x, params):
        """Forward pass using provided parameters"""
        x = x.view(x.size(0), -1)
        
        # Manually apply layers with given parameters
        x = F.linear(x, params['features.0.weight'], params['features.0.bias'])
        x = F.relu(x)
        x = F.linear(x, params['features.2.weight'], params['features.2.bias'])
        x = F.relu(x)
        x = F.linear(x, params['classifier.weight'], params['classifier.bias'])
        
        return x
    
    def meta_train_step(self, tasks):
        """Meta-training step on batch of tasks"""
        meta_loss = 0
        
        for support_x, support_y, query_x, query_y in tasks:
            # Inner loop adaptation
            fast_weights = self.inner_loop(support_x, support_y)
            
            # Evaluate on query set
            query_logits = self.functional_forward(query_x, fast_weights)
            query_loss = F.cross_entropy(query_logits, query_y)
            
            meta_loss += query_loss
        
        # Meta-update
        meta_loss = meta_loss / len(tasks)
        self.meta_optimizer.zero_grad()
        meta_loss.backward()
        self.meta_optimizer.step()
        
        return meta_loss.item()
    
    def adapt_and_evaluate(self, support_x, support_y, query_x, query_y):
        """Adapt to new task and evaluate"""
        # Adapt on support set
        fast_weights = self.inner_loop(support_x, support_y)
        
        # Evaluate on query set
        with torch.no_grad():
            query_logits = self.functional_forward(query_x, fast_weights)
            predictions = query_logits.argmax(dim=1)
            accuracy = (predictions == query_y).float().mean()
        
        return accuracy.item()

# Create MAML model
input_size = 100
maml_model = SimpleClassifier(input_size=input_size, output_size=5)
maml = MAML(maml_model, inner_lr=0.01, meta_lr=0.001, inner_steps=5)

# Meta-training simulation
print("Meta-training MAML...")
meta_losses = []

for episode in range(100):
    # Generate batch of tasks
    tasks = []
    for _ in range(4):  # 4 tasks per meta-batch
        task = generate_task(n_way=5, k_shot=5, q_queries=15, feature_dim=input_size)
        tasks.append(task)
    
    # Meta-train step
    loss = maml.meta_train_step(tasks)
    meta_losses.append(loss)
    
    if episode % 20 == 0:
        print(f"Episode {episode}, Meta Loss: {loss:.4f}")

print()

# Example 3: Prototypical Networks
print("Example 3: Prototypical Networks")
print("=" * 50)

class PrototypicalNetwork(nn.Module):
    """Prototypical Networks for few-shot classification"""
    def __init__(self, input_size, hidden_size=128, embedding_size=64):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, embedding_size)
        )
    
    def forward(self, x):
        x = x.view(x.size(0), -1)
        return self.encoder(x)
    
    def compute_prototypes(self, support_embeddings, support_labels, n_way):
        """Compute class prototypes from support set"""
        prototypes = torch.zeros(n_way, support_embeddings.size(1)).to(support_embeddings.device)
        
        for class_idx in range(n_way):
            mask = support_labels == class_idx
            class_embeddings = support_embeddings[mask]
            prototypes[class_idx] = class_embeddings.mean(dim=0)
        
        return prototypes
    
    def prototypical_loss(self, prototypes, query_embeddings, query_labels):
        """Compute prototypical loss"""
        # Compute distances from queries to prototypes
        distances = torch.cdist(query_embeddings, prototypes)
        
        # Convert to similarities (negative distance)
        log_p_y = F.log_softmax(-distances, dim=1)
        
        # Compute loss
        loss = F.nll_loss(log_p_y, query_labels)
        
        # Compute accuracy
        predictions = (-distances).argmax(dim=1)
        accuracy = (predictions == query_labels).float().mean()
        
        return loss, accuracy

# Create and train Prototypical Network
proto_net = PrototypicalNetwork(input_size=100, embedding_size=64).to(device)
optimizer = optim.Adam(proto_net.parameters(), lr=0.001)

print("Training Prototypical Network...")
proto_losses = []
proto_accuracies = []

for episode in range(200):
    # Generate task
    support_x, support_y, query_x, query_y = generate_task(
        n_way=5, k_shot=5, q_queries=15, feature_dim=100
    )
    
    support_x = support_x.to(device)
    support_y = support_y.to(device)
    query_x = query_x.to(device)
    query_y = query_y.to(device)
    
    # Forward pass
    support_embeddings = proto_net(support_x)
    query_embeddings = proto_net(query_x)
    
    # Compute prototypes
    prototypes = proto_net.compute_prototypes(support_embeddings, support_y, n_way=5)
    
    # Compute loss
    loss, accuracy = proto_net.prototypical_loss(prototypes, query_embeddings, query_y)
    
    # Backward pass
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    proto_losses.append(loss.item())
    proto_accuracies.append(accuracy.item())
    
    if episode % 50 == 0:
        print(f"Episode {episode}, Loss: {loss.item():.4f}, Accuracy: {accuracy.item():.4f}")

print()

# Example 4: Matching Networks
print("Example 4: Matching Networks")
print("=" * 50)

class MatchingNetwork(nn.Module):
    """Matching Networks with attention mechanism"""
    def __init__(self, input_size, hidden_size=128, embedding_size=64):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, embedding_size)
        )
        
        # Bidirectional LSTM for full context embeddings
        self.lstm = nn.LSTM(embedding_size, embedding_size, batch_first=True, bidirectional=True)
        self.attention_fc = nn.Linear(embedding_size * 2, embedding_size)
    
    def forward(self, x):
        x = x.view(x.size(0), -1)
        return self.encoder(x)
    
    def attention(self, query, support, support_labels):
        """Compute attention-weighted predictions"""
        # Compute cosine similarity
        query_norm = F.normalize(query, p=2, dim=1)
        support_norm = F.normalize(support, p=2, dim=1)
        similarities = torch.mm(query_norm, support_norm.t())
        
        # Apply softmax to get attention weights
        attention_weights = F.softmax(similarities, dim=1)
        
        # Convert labels to one-hot
        n_way = support_labels.max().item() + 1
        support_labels_onehot = F.one_hot(support_labels, n_way).float()
        
        # Weighted sum of support labels
        predictions = torch.mm(attention_weights, support_labels_onehot)
        
        return predictions, attention_weights
    
    def full_context_embedding(self, embeddings):
        """Apply bidirectional LSTM for full context"""
        # Add batch dimension if needed
        if embeddings.dim() == 2:
            embeddings = embeddings.unsqueeze(0)
        
        lstm_out, _ = self.lstm(embeddings)
        lstm_out = self.attention_fc(lstm_out)
        
        return lstm_out.squeeze(0)

# Create Matching Network
matching_net = MatchingNetwork(input_size=100).to(device)
matching_optimizer = optim.Adam(matching_net.parameters(), lr=0.001)

print("Training Matching Network...")
matching_losses = []

for episode in range(200):
    # Generate task
    support_x, support_y, query_x, query_y = generate_task(
        n_way=5, k_shot=5, q_queries=15, feature_dim=100
    )
    
    support_x = support_x.to(device)
    support_y = support_y.to(device)
    query_x = query_x.to(device)
    query_y = query_y.to(device)
    
    # Get embeddings
    support_embeddings = matching_net(support_x)
    query_embeddings = matching_net(query_x)
    
    # Apply full context embedding
    support_embeddings = matching_net.full_context_embedding(support_embeddings)
    
    # Get predictions using attention
    predictions, attention_weights = matching_net.attention(
        query_embeddings, support_embeddings, support_y
    )
    
    # Compute loss
    loss = F.cross_entropy(predictions, query_y)
    
    # Backward pass
    matching_optimizer.zero_grad()
    loss.backward()
    matching_optimizer.step()
    
    matching_losses.append(loss.item())
    
    if episode % 50 == 0:
        accuracy = (predictions.argmax(dim=1) == query_y).float().mean()
        print(f"Episode {episode}, Loss: {loss.item():.4f}, Accuracy: {accuracy.item():.4f}")

print()

# Example 5: Reptile Algorithm
print("Example 5: Reptile Algorithm")
print("=" * 50)

class Reptile:
    """Reptile meta-learning algorithm (simplified MAML)"""
    def __init__(self, model, inner_lr=0.01, meta_lr=0.1, inner_steps=10):
        self.model = model
        self.inner_lr = inner_lr
        self.meta_lr = meta_lr
        self.inner_steps = inner_steps
        self.meta_weights = OrderedDict(model.named_parameters())
    
    def inner_train(self, support_x, support_y):
        """Inner loop training on a task"""
        # Copy model parameters
        inner_model = copy.deepcopy(self.model)
        inner_optimizer = optim.SGD(inner_model.parameters(), lr=self.inner_lr)
        
        # Train on support set
        for _ in range(self.inner_steps):
            logits, _ = inner_model(support_x)
            loss = F.cross_entropy(logits, support_y)
            
            inner_optimizer.zero_grad()
            loss.backward()
            inner_optimizer.step()
        
        return OrderedDict(inner_model.named_parameters())
    
    def meta_update(self, tasks):
        """Meta-update using Reptile algorithm"""
        sum_grads = None
        
        for support_x, support_y, _, _ in tasks:
            # Get adapted parameters
            adapted_weights = self.inner_train(support_x, support_y)
            
            # Compute gradients (parameter differences)
            if sum_grads is None:
                sum_grads = OrderedDict()
                for name in self.meta_weights:
                    sum_grads[name] = adapted_weights[name] - self.meta_weights[name]
            else:
                for name in self.meta_weights:
                    sum_grads[name] += adapted_weights[name] - self.meta_weights[name]
        
        # Apply meta-update
        for name in self.meta_weights:
            self.meta_weights[name] = self.meta_weights[name] + \
                                    self.meta_lr * sum_grads[name] / len(tasks)
        
        # Update model with new meta-weights
        self.model.load_state_dict(self.meta_weights)

# Create Reptile model
reptile_model = SimpleClassifier(input_size=100, output_size=5).to(device)
reptile = Reptile(reptile_model, inner_lr=0.01, meta_lr=0.1)

print("Training with Reptile...")
for episode in range(100):
    # Generate batch of tasks
    tasks = []
    for _ in range(5):
        task = generate_task(n_way=5, k_shot=5, q_queries=15, feature_dim=100)
        tasks.append([t.to(device) for t in task])
    
    # Meta-update
    reptile.meta_update(tasks)
    
    if episode % 20 == 0:
        # Evaluate
        test_task = generate_task(n_way=5, k_shot=5, q_queries=15, feature_dim=100)
        support_x, support_y, query_x, query_y = [t.to(device) for t in test_task]
        
        # Adapt to test task
        adapted_weights = reptile.inner_train(support_x, support_y)
        temp_model = copy.deepcopy(reptile.model)
        temp_model.load_state_dict(adapted_weights)
        
        # Evaluate
        with torch.no_grad():
            logits, _ = temp_model(query_x)
            accuracy = (logits.argmax(dim=1) == query_y).float().mean()
        
        print(f"Episode {episode}, Test Accuracy: {accuracy.item():.4f}")

print()

# Example 6: Comparison and Visualization
print("Example 6: Algorithm Comparison")
print("=" * 50)

# Plot training curves
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# MAML loss
axes[0, 0].plot(meta_losses)
axes[0, 0].set_title('MAML Meta-Loss')
axes[0, 0].set_xlabel('Episode')
axes[0, 0].set_ylabel('Loss')
axes[0, 0].grid(True, alpha=0.3)

# Prototypical Networks
axes[0, 1].plot(proto_losses, label='Loss', alpha=0.7)
axes[0, 1].plot(proto_accuracies, label='Accuracy', alpha=0.7)
axes[0, 1].set_title('Prototypical Networks')
axes[0, 1].set_xlabel('Episode')
axes[0, 1].set_ylabel('Value')
axes[0, 1].legend()
axes[0, 1].grid(True, alpha=0.3)

# Matching Networks
axes[1, 0].plot(matching_losses)
axes[1, 0].set_title('Matching Networks Loss')
axes[1, 0].set_xlabel('Episode')
axes[1, 0].set_ylabel('Loss')
axes[1, 0].grid(True, alpha=0.3)

# Algorithm comparison
algorithms = ['MAML', 'ProtoNet', 'MatchingNet', 'Reptile']
properties = {
    'Complexity': [4, 2, 3, 1],
    'Performance': [4, 3, 3, 3],
    'Speed': [1, 4, 3, 4],
    'Flexibility': [4, 2, 3, 3]
}

x = np.arange(len(algorithms))
width = 0.2

for i, (prop, values) in enumerate(properties.items()):
    axes[1, 1].bar(x + i*width, values, width, label=prop)

axes[1, 1].set_xlabel('Algorithm')
axes[1, 1].set_ylabel('Score')
axes[1, 1].set_title('Algorithm Comparison')
axes[1, 1].set_xticks(x + width * 1.5)
axes[1, 1].set_xticklabels(algorithms)
axes[1, 1].legend()
axes[1, 1].grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.show()

# Best practices summary
print("Meta-Learning Best Practices")
print("=" * 50)
print("1. Task Design:")
print("   - Ensure tasks are representative of test distribution")
print("   - Balance task difficulty during training")
print("   - Use episodic training with proper train/val/test splits")
print()
print("2. Model Selection:")
print("   - MAML: When you need fast adaptation with few gradient steps")
print("   - ProtoNet: For simple, efficient few-shot classification")
print("   - MatchingNet: When attention mechanisms are beneficial")
print("   - Reptile: For simplicity and scalability")
print()
print("3. Hyperparameters:")
print("   - Inner learning rate: Usually higher than outer")
print("   - Number of inner steps: Balance adaptation vs computation")
print("   - Task batch size: Larger is generally better")
print()
print("4. Evaluation:")
print("   - Test on completely unseen classes/tasks")
print("   - Report confidence intervals over multiple runs")
print("   - Consider both accuracy and adaptation speed")

# Summary table
print("\nAlgorithm Summary:")
print("-" * 70)
print(f"{'Algorithm':<15} {'Type':<20} {'Key Feature':<35}")
print("-" * 70)
print(f"{'MAML':<15} {'Optimization-based':<20} {'Fast adaptation via gradient descent':<35}")
print(f"{'ProtoNet':<15} {'Metric-based':<20} {'Class prototypes in embedding space':<35}")
print(f"{'MatchingNet':<15} {'Metric-based':<20} {'Attention over support set':<35}")
print(f"{'Reptile':<15} {'Optimization-based':<20} {'Simplified MAML without second-order':<35}")