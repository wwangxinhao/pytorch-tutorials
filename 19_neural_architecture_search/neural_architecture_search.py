"""
Tutorial 19: Neural Architecture Search
=======================================

This tutorial explores Neural Architecture Search (NAS) techniques for
automatically designing optimal neural network architectures.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from typing import List, Dict, Tuple, Optional
import random
import copy
import time
import matplotlib.pyplot as plt
from collections import defaultdict

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")
print()

# Example 1: Basic Search Space Design
print("Example 1: Search Space Design")
print("=" * 50)

# Define basic operations for search space
class ConvBlock(nn.Module):
    """Basic convolutional block"""
    def __init__(self, in_channels, out_channels, kernel_size, stride=1):
        super().__init__()
        padding = kernel_size // 2
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()
        
    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))

class DepthwiseSeparableConv(nn.Module):
    """Depthwise separable convolution"""
    def __init__(self, in_channels, out_channels, kernel_size, stride=1):
        super().__init__()
        padding = kernel_size // 2
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size, 
                                  stride, padding, groups=in_channels)
        self.pointwise = nn.Conv2d(in_channels, out_channels, 1)
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()
        
    def forward(self, x):
        x = self.relu(self.bn1(self.depthwise(x)))
        x = self.bn2(self.pointwise(x))
        return self.relu(x)

class IdentityBlock(nn.Module):
    """Identity/skip connection"""
    def __init__(self, channels):
        super().__init__()
        self.channels = channels
        
    def forward(self, x):
        return x

# Search space definition
PRIMITIVES = {
    'conv3x3': lambda c: ConvBlock(c, c, 3),
    'conv5x5': lambda c: ConvBlock(c, c, 5),
    'dw_conv3x3': lambda c: DepthwiseSeparableConv(c, c, 3),
    'dw_conv5x5': lambda c: DepthwiseSeparableConv(c, c, 5),
    'max_pool3x3': lambda c: nn.MaxPool2d(3, stride=1, padding=1),
    'avg_pool3x3': lambda c: nn.AvgPool2d(3, stride=1, padding=1),
    'identity': lambda c: IdentityBlock(c)
}

print("Search Space Operations:")
for i, (name, _) in enumerate(PRIMITIVES.items()):
    print(f"{i}: {name}")
print()

# Architecture representation
class Architecture:
    """Represents a neural architecture as a list of operations"""
    def __init__(self, layers: List[str]):
        self.layers = layers
        
    def __repr__(self):
        return f"Architecture({' -> '.join(self.layers)})"
    
    def mutate(self, mutation_prob=0.1):
        """Random mutation for evolutionary search"""
        new_layers = self.layers.copy()
        for i in range(len(new_layers)):
            if random.random() < mutation_prob:
                new_layers[i] = random.choice(list(PRIMITIVES.keys()))
        return Architecture(new_layers)
    
    def crossover(self, other: 'Architecture'):
        """Crossover for evolutionary search"""
        crossover_point = random.randint(1, len(self.layers) - 1)
        new_layers = self.layers[:crossover_point] + other.layers[crossover_point:]
        return Architecture(new_layers)

# Example architectures
arch1 = Architecture(['conv3x3', 'max_pool3x3', 'conv5x5', 'identity'])
arch2 = Architecture(['dw_conv3x3', 'avg_pool3x3', 'dw_conv5x5', 'conv3x3'])

print("Example Architectures:")
print(f"Architecture 1: {arch1}")
print(f"Architecture 2: {arch2}")
print(f"Mutated Arch 1: {arch1.mutate(0.3)}")
print(f"Crossover: {arch1.crossover(arch2)}")
print()

# Example 2: Random Search
print("Example 2: Random Search")
print("=" * 50)

class NASModel(nn.Module):
    """Model that can be built from architecture description"""
    def __init__(self, architecture: Architecture, input_channels=3, num_classes=10):
        super().__init__()
        self.architecture = architecture
        
        # Build layers from architecture
        layers = []
        channels = 32  # Starting channels
        
        # Initial conv
        layers.append(ConvBlock(input_channels, channels, 3))
        
        # Architecture-defined layers
        for op_name in architecture.layers:
            if 'pool' in op_name:
                layers.append(PRIMITIVES[op_name](channels))
            else:
                layers.append(PRIMITIVES[op_name](channels))
        
        self.features = nn.Sequential(*layers)
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(channels, num_classes)
        
    def forward(self, x):
        x = self.features(x)
        x = self.global_pool(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

def evaluate_architecture(architecture: Architecture, num_epochs=5):
    """Quick evaluation of an architecture"""
    model = NASModel(architecture).to(device)
    
    # Count parameters
    num_params = sum(p.numel() for p in model.parameters())
    
    # Simulate training (in practice, you'd train on real data)
    # Here we just return random performance for demonstration
    accuracy = np.random.uniform(0.6, 0.95)
    latency = num_params / 1e6 * np.random.uniform(0.8, 1.2)  # Simulated latency
    
    return {
        'accuracy': accuracy,
        'params': num_params,
        'latency': latency,
        'architecture': architecture
    }

# Random search
def random_search(num_architectures=20, layers_per_arch=4):
    """Random architecture search"""
    results = []
    
    print("Running Random Search...")
    for i in range(num_architectures):
        # Generate random architecture
        layers = [random.choice(list(PRIMITIVES.keys())) for _ in range(layers_per_arch)]
        arch = Architecture(layers)
        
        # Evaluate
        result = evaluate_architecture(arch)
        results.append(result)
        
        if i % 5 == 0:
            print(f"Evaluated {i+1}/{num_architectures} architectures")
    
    # Sort by accuracy
    results.sort(key=lambda x: x['accuracy'], reverse=True)
    
    return results

random_results = random_search(num_architectures=30)

print("\nTop 5 Architectures (Random Search):")
for i, result in enumerate(random_results[:5]):
    print(f"{i+1}. Accuracy: {result['accuracy']:.3f}, "
          f"Params: {result['params']/1e6:.2f}M, "
          f"Arch: {result['architecture'].layers[:3]}...")
print()

# Example 3: Evolutionary Search
print("Example 3: Evolutionary Algorithm")
print("=" * 50)

class EvolutionarySearch:
    """Evolutionary algorithm for NAS"""
    def __init__(self, population_size=20, mutation_prob=0.1, layers_per_arch=4):
        self.population_size = population_size
        self.mutation_prob = mutation_prob
        self.layers_per_arch = layers_per_arch
        
    def initialize_population(self):
        """Create initial random population"""
        population = []
        for _ in range(self.population_size):
            layers = [random.choice(list(PRIMITIVES.keys())) 
                     for _ in range(self.layers_per_arch)]
            population.append(Architecture(layers))
        return population
    
    def evaluate_population(self, population):
        """Evaluate all architectures in population"""
        results = []
        for arch in population:
            result = evaluate_architecture(arch)
            results.append(result)
        return results
    
    def select_parents(self, population_results, num_parents):
        """Tournament selection"""
        parents = []
        for _ in range(num_parents):
            # Tournament of size 3
            tournament = random.sample(population_results, 3)
            winner = max(tournament, key=lambda x: x['accuracy'])
            parents.append(winner['architecture'])
        return parents
    
    def evolve(self, num_generations=20):
        """Run evolutionary search"""
        # Initialize
        population = self.initialize_population()
        history = {'best_accuracy': [], 'avg_accuracy': []}
        
        print("Running Evolutionary Search...")
        for generation in range(num_generations):
            # Evaluate current population
            results = self.evaluate_population(population)
            
            # Track statistics
            accuracies = [r['accuracy'] for r in results]
            history['best_accuracy'].append(max(accuracies))
            history['avg_accuracy'].append(np.mean(accuracies))
            
            # Select parents
            num_parents = self.population_size // 2
            parents = self.select_parents(results, num_parents)
            
            # Generate offspring
            offspring = []
            while len(offspring) < self.population_size:
                if random.random() < 0.5 and len(parents) >= 2:
                    # Crossover
                    p1, p2 = random.sample(parents, 2)
                    child = p1.crossover(p2)
                else:
                    # Mutation
                    parent = random.choice(parents)
                    child = parent.mutate(self.mutation_prob)
                offspring.append(child)
            
            # Replace population
            population = offspring
            
            if generation % 5 == 0:
                print(f"Generation {generation}: "
                      f"Best Acc = {history['best_accuracy'][-1]:.3f}, "
                      f"Avg Acc = {history['avg_accuracy'][-1]:.3f}")
        
        # Final evaluation
        final_results = self.evaluate_population(population)
        final_results.sort(key=lambda x: x['accuracy'], reverse=True)
        
        return final_results, history

# Run evolutionary search
evo_search = EvolutionarySearch(population_size=30)
evo_results, evo_history = evo_search.evolve(num_generations=20)

print("\nTop 5 Architectures (Evolutionary Search):")
for i, result in enumerate(evo_results[:5]):
    print(f"{i+1}. Accuracy: {result['accuracy']:.3f}, "
          f"Params: {result['params']/1e6:.2f}M")
print()

# Example 4: Differentiable Architecture Search (DARTS)
print("Example 4: Differentiable Architecture Search (DARTS)")
print("=" * 50)

class MixedOperation(nn.Module):
    """Mixed operation for DARTS"""
    def __init__(self, channels):
        super().__init__()
        self.ops = nn.ModuleList([
            op(channels) for op in PRIMITIVES.values()
        ])
        
    def forward(self, x, weights):
        """weights: softmax over operations"""
        return sum(w * op(x) for w, op in zip(weights, self.ops))

class DARTSCell(nn.Module):
    """DARTS cell with learnable architecture"""
    def __init__(self, channels, num_nodes=4):
        super().__init__()
        self.num_nodes = num_nodes
        
        # Mixed operations for each edge
        self.ops = nn.ModuleList()
        for i in range(num_nodes):
            for j in range(i):
                self.ops.append(MixedOperation(channels))
        
        # Architecture parameters (to be learned)
        self.arch_params = nn.Parameter(
            torch.randn(len(self.ops), len(PRIMITIVES)) / 10
        )
        
    def forward(self, x):
        states = [x]
        offset = 0
        
        # Compute intermediate nodes
        for i in range(1, self.num_nodes):
            s = []
            for j in range(i):
                weights = F.softmax(self.arch_params[offset], dim=0)
                s.append(self.ops[offset](states[j], weights))
                offset += 1
            states.append(sum(s))
        
        # Output is concatenation of intermediate nodes
        return torch.cat(states[1:], dim=1)
    
    def get_genotype(self):
        """Extract discrete architecture"""
        gene = []
        offset = 0
        
        for i in range(1, self.num_nodes):
            edges = []
            for j in range(i):
                weights = F.softmax(self.arch_params[offset], dim=0)
                op_idx = weights.argmax().item()
                op_name = list(PRIMITIVES.keys())[op_idx]
                edges.append((op_name, j))
                offset += 1
            
            # Select top 2 edges
            edges.sort(key=lambda x: weights[x[1]], reverse=True)
            gene.extend(edges[:2])
        
        return gene

class DARTS(nn.Module):
    """DARTS model"""
    def __init__(self, input_channels=3, num_classes=10, num_cells=8):
        super().__init__()
        channels = 16
        
        self.stem = nn.Sequential(
            nn.Conv2d(input_channels, channels, 3, padding=1),
            nn.BatchNorm2d(channels)
        )
        
        # Stack of cells
        self.cells = nn.ModuleList()
        for i in range(num_cells):
            # Double channels at 1/3 and 2/3
            if i in [num_cells // 3, 2 * num_cells // 3]:
                channels *= 2
            self.cells.append(DARTSCell(channels))
        
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(channels * 3, num_classes)  # 3 intermediate nodes
        
    def forward(self, x):
        x = self.stem(x)
        
        for cell in self.cells:
            x = cell(x)
        
        x = self.global_pool(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x
    
    def get_arch_parameters(self):
        """Get architecture parameters for optimization"""
        arch_params = []
        for cell in self.cells:
            arch_params.append(cell.arch_params)
        return arch_params

# Simplified DARTS training (demonstration)
print("DARTS Architecture Search")
print("Creating DARTS model...")
darts_model = DARTS(num_cells=4).to(device)

# In practice, you would:
# 1. Alternate between updating weights and architecture parameters
# 2. Use validation loss to update architecture parameters
# 3. Extract final architecture using get_genotype()

print(f"DARTS model created with {sum(p.numel() for p in darts_model.parameters())/1e6:.2f}M parameters")
print()

# Example 5: Early Stopping and Performance Prediction
print("Example 5: Performance Prediction")
print("=" * 50)

class PerformancePredictor:
    """Predict final performance from early training curves"""
    def __init__(self):
        self.history = []
        
    def fit(self, early_curves, final_accuracies):
        """Fit predictor on historical data"""
        # Simple linear regression on curve features
        from sklearn.linear_model import LinearRegression
        
        features = []
        for curve in early_curves:
            # Extract features: mean, std, slope
            mean_acc = np.mean(curve)
            std_acc = np.std(curve)
            slope = (curve[-1] - curve[0]) / len(curve)
            features.append([mean_acc, std_acc, slope])
        
        self.model = LinearRegression()
        self.model.fit(features, final_accuracies)
        
    def predict(self, early_curve):
        """Predict final accuracy from early curve"""
        mean_acc = np.mean(early_curve)
        std_acc = np.std(early_curve)
        slope = (early_curve[-1] - early_curve[0]) / len(early_curve)
        features = [[mean_acc, std_acc, slope]]
        return self.model.predict(features)[0]

# Simulate training curves
def simulate_training_curve(final_acc, noise=0.05, epochs=20):
    """Simulate a training curve"""
    curve = []
    current = 0.1  # Start from low accuracy
    
    for epoch in range(epochs):
        # Exponential approach to final accuracy
        current += (final_acc - current) * 0.2
        current += np.random.normal(0, noise)
        curve.append(np.clip(current, 0, 1))
    
    return curve

# Generate training data
print("Generating training curves for performance prediction...")
num_architectures = 50
early_epochs = 5
total_epochs = 20

training_data = []
for _ in range(num_architectures):
    final_acc = np.random.uniform(0.6, 0.95)
    full_curve = simulate_training_curve(final_acc, epochs=total_epochs)
    early_curve = full_curve[:early_epochs]
    training_data.append((early_curve, final_acc))

# Split data
train_size = int(0.8 * len(training_data))
train_data = training_data[:train_size]
test_data = training_data[train_size:]

# Train predictor
predictor = PerformancePredictor()
early_curves_train = [d[0] for d in train_data]
final_accs_train = [d[1] for d in train_data]
predictor.fit(early_curves_train, final_accs_train)

# Test predictor
early_curves_test = [d[0] for d in test_data]
final_accs_test = [d[1] for d in test_data]
predictions = [predictor.predict(curve) for curve in early_curves_test]

# Calculate error
mae = np.mean(np.abs(np.array(predictions) - np.array(final_accs_test)))
print(f"Performance Predictor MAE: {mae:.3f}")
print(f"Can predict final accuracy from {early_epochs} epochs instead of {total_epochs}")
print()

# Example 6: Multi-Objective NAS
print("Example 6: Multi-Objective NAS")
print("=" * 50)

def pareto_frontier(results, objectives=['accuracy', 'latency']):
    """Find Pareto frontier for multi-objective optimization"""
    pareto_front = []
    
    for candidate in results:
        dominated = False
        
        for other in results:
            if other == candidate:
                continue
                
            # Check if candidate is dominated
            better_in_all = all(
                other[obj] >= candidate[obj] if obj == 'accuracy' 
                else other[obj] <= candidate[obj]
                for obj in objectives
            )
            better_in_one = any(
                other[obj] > candidate[obj] if obj == 'accuracy'
                else other[obj] < candidate[obj]
                for obj in objectives
            )
            
            if better_in_all and better_in_one:
                dominated = True
                break
        
        if not dominated:
            pareto_front.append(candidate)
    
    return pareto_front

# Find Pareto optimal architectures
pareto_architectures = pareto_frontier(random_results)

print(f"Found {len(pareto_architectures)} Pareto optimal architectures")
print("Pareto Front:")
for i, arch in enumerate(pareto_architectures[:5]):
    print(f"{i+1}. Accuracy: {arch['accuracy']:.3f}, "
          f"Latency: {arch['latency']:.3f}ms")

# Visualization
print("\nGenerating visualizations...")

# Plot 1: Search progression
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# Evolution history
axes[0, 0].plot(evo_history['best_accuracy'], label='Best', linewidth=2)
axes[0, 0].plot(evo_history['avg_accuracy'], label='Average', linewidth=2)
axes[0, 0].set_xlabel('Generation')
axes[0, 0].set_ylabel('Accuracy')
axes[0, 0].set_title('Evolutionary Search Progress')
axes[0, 0].legend()
axes[0, 0].grid(True, alpha=0.3)

# Accuracy vs Parameters scatter
axes[0, 1].scatter([r['params']/1e6 for r in random_results], 
                   [r['accuracy'] for r in random_results],
                   alpha=0.6, label='All architectures')
axes[0, 1].scatter([r['params']/1e6 for r in pareto_architectures], 
                   [r['accuracy'] for r in pareto_architectures],
                   color='red', s=100, label='Pareto optimal')
axes[0, 1].set_xlabel('Parameters (M)')
axes[0, 1].set_ylabel('Accuracy')
axes[0, 1].set_title('Accuracy vs Model Size')
axes[0, 1].legend()
axes[0, 1].grid(True, alpha=0.3)

# Performance prediction
axes[1, 0].scatter(final_accs_test, predictions, alpha=0.7)
axes[1, 0].plot([0.6, 0.95], [0.6, 0.95], 'r--', label='Perfect prediction')
axes[1, 0].set_xlabel('True Final Accuracy')
axes[1, 0].set_ylabel('Predicted Final Accuracy')
axes[1, 0].set_title('Performance Prediction')
axes[1, 0].legend()
axes[1, 0].grid(True, alpha=0.3)

# Operation frequency
op_counts = defaultdict(int)
for result in random_results[:10]:  # Top 10 architectures
    for op in result['architecture'].layers:
        op_counts[op] += 1

ops = list(op_counts.keys())
counts = list(op_counts.values())
axes[1, 1].bar(ops, counts)
axes[1, 1].set_xlabel('Operation')
axes[1, 1].set_ylabel('Frequency in Top Architectures')
axes[1, 1].set_title('Popular Operations')
axes[1, 1].tick_params(axis='x', rotation=45)

plt.tight_layout()
plt.savefig('nas_results.png', dpi=150, bbox_inches='tight')
plt.close()

print("Visualizations saved to 'nas_results.png'")
print()

# Summary
print("NAS Methods Summary")
print("=" * 50)

methods = {
    "Random Search": {
        "Pros": "Simple, parallelizable, no bias",
        "Cons": "Inefficient, no learning",
        "Best for": "Small search spaces, baseline"
    },
    "Evolutionary": {
        "Pros": "Population-based, handles discrete spaces",
        "Cons": "Many evaluations needed",
        "Best for": "Medium search spaces, multi-objective"
    },
    "DARTS": {
        "Pros": "Efficient, end-to-end differentiable",
        "Cons": "Memory intensive, approximations",
        "Best for": "Large search spaces, continuous relaxation"
    },
    "Predictor-based": {
        "Pros": "Few full evaluations, learns from history",
        "Cons": "Requires good predictors",
        "Best for": "Expensive evaluations"
    }
}

for method, details in methods.items():
    print(f"\n{method}:")
    for key, value in details.items():
        print(f"  {key}: {value}")

print("\nKey Takeaways:")
print("- NAS automates architecture design")
print("- Trade-off between search efficiency and quality")
print("- Multi-objective optimization is often necessary")
print("- Early stopping and predictors save computation")
print("- Hardware-aware NAS is increasingly important")