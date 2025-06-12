"""
Tutorial 20: Bayesian Deep Learning
===================================

This tutorial explores Bayesian approaches to deep learning, focusing on
uncertainty quantification and probabilistic modeling.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Normal, Categorical
import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, List, Optional
import seaborn as sns
from scipy import stats

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")
print()

# Example 1: Understanding Uncertainty
print("Example 1: Types of Uncertainty")
print("=" * 50)

# Generate synthetic data with different types of uncertainty
def generate_data_with_uncertainty(n_samples=100):
    """Generate data with epistemic and aleatoric uncertainty"""
    x = np.linspace(-3, 3, n_samples)
    
    # True function
    y_true = np.sin(x)
    
    # Aleatoric uncertainty (noise in data)
    noise_std = 0.1 + 0.1 * np.abs(x)  # Heteroscedastic noise
    y_aleatoric = y_true + np.random.normal(0, noise_std)
    
    # Epistemic uncertainty (lack of data in some regions)
    mask = (x > -1) & (x < 1)  # Remove data in middle region
    x_epistemic = x[~mask]
    y_epistemic = y_aleatoric[~mask]
    
    return x, y_true, y_aleatoric, x_epistemic, y_epistemic, noise_std

x, y_true, y_aleatoric, x_epistemic, y_epistemic, noise_std = generate_data_with_uncertainty()

# Visualize uncertainty types
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Aleatoric uncertainty
axes[0].plot(x, y_true, 'k-', label='True function', linewidth=2)
axes[0].scatter(x, y_aleatoric, alpha=0.5, s=20, label='Noisy observations')
axes[0].fill_between(x, y_true - 2*noise_std, y_true + 2*noise_std, 
                     alpha=0.3, label='Aleatoric uncertainty')
axes[0].set_xlabel('x')
axes[0].set_ylabel('y')
axes[0].set_title('Aleatoric Uncertainty (Data Noise)')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# Epistemic uncertainty
axes[1].plot(x, y_true, 'k-', label='True function', linewidth=2)
axes[1].scatter(x_epistemic, y_epistemic, alpha=0.5, s=20, label='Available data')
axes[1].axvspan(-1, 1, alpha=0.2, color='red', label='High epistemic uncertainty')
axes[1].set_xlabel('x')
axes[1].set_ylabel('y')
axes[1].set_title('Epistemic Uncertainty (Lack of Data)')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

print("Aleatoric uncertainty: Irreducible noise in the data")
print("Epistemic uncertainty: Reducible uncertainty due to lack of knowledge")
print()

# Example 2: Monte Carlo Dropout
print("Example 2: Monte Carlo Dropout")
print("=" * 50)

class MCDropoutNet(nn.Module):
    """Neural network with Monte Carlo Dropout for uncertainty estimation"""
    def __init__(self, input_dim=1, hidden_dim=100, output_dim=1, dropout_rate=0.1):
        super().__init__()
        self.dropout_rate = dropout_rate
        
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, output_dim)
        
        # Dropout layers
        self.dropout = nn.Dropout(dropout_rate)
        
    def forward(self, x, training=True):
        # Apply dropout even during evaluation for MC Dropout
        if training:
            self.train()
        else:
            self.eval()
            # Override eval mode for dropout layers
            for m in self.modules():
                if isinstance(m, nn.Dropout):
                    m.train()
        
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        return x
    
    def predict_with_uncertainty(self, x, n_samples=50):
        """Make predictions with uncertainty using MC Dropout"""
        predictions = []
        
        for _ in range(n_samples):
            with torch.no_grad():
                pred = self.forward(x, training=False)
                predictions.append(pred)
        
        predictions = torch.stack(predictions)
        mean = predictions.mean(dim=0)
        std = predictions.std(dim=0)
        
        return mean, std, predictions

# Train MC Dropout model
def train_mc_dropout_model(x_train, y_train, epochs=1000):
    """Train MC Dropout model"""
    model = MCDropoutNet(dropout_rate=0.2).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    
    x_tensor = torch.FloatTensor(x_train).unsqueeze(1).to(device)
    y_tensor = torch.FloatTensor(y_train).unsqueeze(1).to(device)
    
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        
        output = model(x_tensor)
        loss = F.mse_loss(output, y_tensor)
        
        loss.backward()
        optimizer.step()
        
        if epoch % 200 == 0:
            print(f"Epoch {epoch}, Loss: {loss.item():.4f}")
    
    return model

# Train on epistemic uncertainty data
print("Training MC Dropout model...")
mc_model = train_mc_dropout_model(x_epistemic, y_epistemic, epochs=1000)

# Make predictions with uncertainty
x_test = torch.FloatTensor(x).unsqueeze(1).to(device)
mean_pred, std_pred, all_preds = mc_model.predict_with_uncertainty(x_test, n_samples=100)

# Convert to numpy
mean_pred = mean_pred.cpu().numpy().flatten()
std_pred = std_pred.cpu().numpy().flatten()

# Visualize MC Dropout predictions
plt.figure(figsize=(10, 6))
plt.plot(x, y_true, 'k-', label='True function', linewidth=2)
plt.scatter(x_epistemic, y_epistemic, alpha=0.5, s=20, label='Training data')
plt.plot(x, mean_pred, 'b-', label='MC Dropout mean', linewidth=2)
plt.fill_between(x, mean_pred - 2*std_pred, mean_pred + 2*std_pred, 
                 alpha=0.3, label='Uncertainty (±2σ)')

# Highlight high uncertainty region
plt.axvspan(-1, 1, alpha=0.1, color='red')
plt.xlabel('x')
plt.ylabel('y')
plt.title('Monte Carlo Dropout Uncertainty Estimation')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()

print(f"Average uncertainty in data region: {std_pred[~((x > -1) & (x < 1))].mean():.3f}")
print(f"Average uncertainty in no-data region: {std_pred[(x > -1) & (x < 1)].mean():.3f}")
print()

# Example 3: Bayesian Neural Networks
print("Example 3: Bayesian Neural Networks")
print("=" * 50)

class BayesianLinear(nn.Module):
    """Bayesian linear layer with weight uncertainty"""
    def __init__(self, in_features, out_features):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        
        # Weight parameters
        self.weight_mu = nn.Parameter(torch.randn(out_features, in_features) * 0.1)
        self.weight_rho = nn.Parameter(torch.randn(out_features, in_features) * -3)
        
        # Bias parameters
        self.bias_mu = nn.Parameter(torch.zeros(out_features))
        self.bias_rho = nn.Parameter(torch.randn(out_features) * -3)
        
        # Prior distributions
        self.weight_prior = Normal(0, 1)
        self.bias_prior = Normal(0, 1)
        
    def forward(self, x):
        # Sample weights and biases
        weight_sigma = torch.log1p(torch.exp(self.weight_rho))
        weight_dist = Normal(self.weight_mu, weight_sigma)
        weight = weight_dist.rsample()
        
        bias_sigma = torch.log1p(torch.exp(self.bias_rho))
        bias_dist = Normal(self.bias_mu, bias_sigma)
        bias = bias_dist.rsample()
        
        # Compute KL divergence
        self.kl_divergence = (
            torch.distributions.kl_divergence(weight_dist, self.weight_prior).sum() +
            torch.distributions.kl_divergence(bias_dist, self.bias_prior).sum()
        )
        
        return F.linear(x, weight, bias)

class BayesianNN(nn.Module):
    """Bayesian Neural Network"""
    def __init__(self, input_dim=1, hidden_dim=50, output_dim=1):
        super().__init__()
        self.fc1 = BayesianLinear(input_dim, hidden_dim)
        self.fc2 = BayesianLinear(hidden_dim, hidden_dim)
        self.fc3 = BayesianLinear(hidden_dim, output_dim)
        
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
    
    def kl_divergence(self):
        """Total KL divergence of the model"""
        kl = 0
        for module in self.modules():
            if isinstance(module, BayesianLinear):
                kl += module.kl_divergence
        return kl
    
    def predict_with_uncertainty(self, x, n_samples=50):
        """Make predictions with uncertainty"""
        predictions = []
        
        for _ in range(n_samples):
            with torch.no_grad():
                pred = self.forward(x)
                predictions.append(pred)
        
        predictions = torch.stack(predictions)
        mean = predictions.mean(dim=0)
        std = predictions.std(dim=0)
        
        return mean, std, predictions

# Train Bayesian Neural Network
def train_bayesian_nn(x_train, y_train, epochs=1000, kl_weight=0.01):
    """Train Bayesian Neural Network with variational inference"""
    model = BayesianNN().to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    
    x_tensor = torch.FloatTensor(x_train).unsqueeze(1).to(device)
    y_tensor = torch.FloatTensor(y_train).unsqueeze(1).to(device)
    
    n_batches = 1
    
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        
        # Forward pass
        output = model(x_tensor)
        
        # ELBO loss = -log likelihood + KL divergence
        nll = F.mse_loss(output, y_tensor, reduction='sum')
        kl = model.kl_divergence()
        loss = nll + kl_weight * kl / n_batches
        
        loss.backward()
        optimizer.step()
        
        if epoch % 200 == 0:
            print(f"Epoch {epoch}, Loss: {loss.item():.4f}, "
                  f"NLL: {nll.item():.4f}, KL: {kl.item():.4f}")
    
    return model

# Train Bayesian NN
print("Training Bayesian Neural Network...")
bnn_model = train_bayesian_nn(x_epistemic, y_epistemic, epochs=1000)

# Make predictions
mean_bnn, std_bnn, _ = bnn_model.predict_with_uncertainty(x_test, n_samples=100)
mean_bnn = mean_bnn.cpu().numpy().flatten()
std_bnn = std_bnn.cpu().numpy().flatten()

print()

# Example 4: Deep Ensembles
print("Example 4: Deep Ensembles")
print("=" * 50)

class SimpleNN(nn.Module):
    """Simple neural network for ensemble"""
    def __init__(self, input_dim=1, hidden_dim=50, output_dim=1):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, output_dim)
        
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

class DeepEnsemble:
    """Deep ensemble for uncertainty estimation"""
    def __init__(self, n_models=5, **model_kwargs):
        self.n_models = n_models
        self.models = [SimpleNN(**model_kwargs).to(device) for _ in range(n_models)]
        
    def train(self, x_train, y_train, epochs=1000):
        """Train ensemble members independently"""
        x_tensor = torch.FloatTensor(x_train).unsqueeze(1).to(device)
        y_tensor = torch.FloatTensor(y_train).unsqueeze(1).to(device)
        
        for i, model in enumerate(self.models):
            print(f"Training model {i+1}/{self.n_models}")
            optimizer = optim.Adam(model.parameters(), lr=0.01)
            
            for epoch in range(epochs):
                model.train()
                optimizer.zero_grad()
                
                output = model(x_tensor)
                loss = F.mse_loss(output, y_tensor)
                
                loss.backward()
                optimizer.step()
                
                if epoch % 500 == 0:
                    print(f"  Epoch {epoch}, Loss: {loss.item():.4f}")
    
    def predict_with_uncertainty(self, x):
        """Make predictions with uncertainty using ensemble"""
        predictions = []
        
        for model in self.models:
            model.eval()
            with torch.no_grad():
                pred = model(x)
                predictions.append(pred)
        
        predictions = torch.stack(predictions)
        mean = predictions.mean(dim=0)
        
        # Total uncertainty = epistemic + aleatoric
        # For regression, we use variance of predictions as epistemic uncertainty
        epistemic_std = predictions.std(dim=0)
        
        return mean, epistemic_std, predictions

# Train deep ensemble
print("Training Deep Ensemble...")
ensemble = DeepEnsemble(n_models=5)
ensemble.train(x_epistemic, y_epistemic, epochs=1000)

# Make predictions
mean_ensemble, std_ensemble, ensemble_preds = ensemble.predict_with_uncertainty(x_test)
mean_ensemble = mean_ensemble.cpu().numpy().flatten()
std_ensemble = std_ensemble.cpu().numpy().flatten()

print()

# Example 5: Uncertainty Calibration
print("Example 5: Uncertainty Calibration")
print("=" * 50)

def evaluate_calibration(predictions, uncertainties, true_values, n_bins=10):
    """Evaluate calibration of uncertainty estimates"""
    # Sort by uncertainty
    indices = np.argsort(uncertainties)
    sorted_pred = predictions[indices]
    sorted_unc = uncertainties[indices]
    sorted_true = true_values[indices]
    
    # Compute calibration
    bin_boundaries = np.linspace(0, len(predictions), n_bins + 1, dtype=int)
    calibration_data = []
    
    for i in range(n_bins):
        start, end = bin_boundaries[i], bin_boundaries[i + 1]
        bin_pred = sorted_pred[start:end]
        bin_unc = sorted_unc[start:end]
        bin_true = sorted_true[start:end]
        
        if len(bin_pred) > 0:
            # Expected confidence (using Gaussian assumption)
            expected_conf = stats.norm.cdf(1) - stats.norm.cdf(-1)  # ~68% for 1 std
            
            # Observed confidence
            errors = np.abs(bin_pred - bin_true)
            observed_conf = np.mean(errors <= bin_unc.mean())
            
            calibration_data.append({
                'expected': expected_conf,
                'observed': observed_conf,
                'avg_uncertainty': bin_unc.mean(),
                'rmse': np.sqrt(np.mean(errors**2))
            })
    
    return calibration_data

# Evaluate calibration for each method
y_test_true = np.sin(x)

methods = {
    'MC Dropout': (mean_pred, std_pred),
    'Bayesian NN': (mean_bnn, std_bnn),
    'Deep Ensemble': (mean_ensemble, std_ensemble)
}

# Visualization comparison
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# Plot all methods
ax = axes[0, 0]
ax.plot(x, y_true, 'k-', label='True function', linewidth=2)
ax.scatter(x_epistemic, y_epistemic, alpha=0.3, s=20, label='Training data')

colors = ['blue', 'green', 'red']
for (method_name, (mean, std)), color in zip(methods.items(), colors):
    ax.plot(x, mean, color=color, label=f'{method_name} mean', linewidth=2)
    ax.fill_between(x, mean - 2*std, mean + 2*std, 
                    color=color, alpha=0.2, label=f'{method_name} ±2σ')

ax.axvspan(-1, 1, alpha=0.1, color='gray')
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_title('Uncertainty Estimation Comparison')
ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
ax.grid(True, alpha=0.3)

# Uncertainty in different regions
ax = axes[0, 1]
data_mask = ~((x > -1) & (x < 1))
no_data_mask = (x > -1) & (x < 1)

for i, (method_name, (_, std)) in enumerate(methods.items()):
    ax.bar(i*2, std[data_mask].mean(), color=colors[i], alpha=0.7, label=method_name)
    ax.bar(i*2+1, std[no_data_mask].mean(), color=colors[i], alpha=0.4)

ax.set_xticks(range(0, 6, 2))
ax.set_xticklabels(['Data\nRegion', 'No Data\nRegion'] * 3)
ax.set_ylabel('Average Uncertainty')
ax.set_title('Uncertainty by Region')
ax.legend()
ax.grid(True, alpha=0.3, axis='y')

# Calibration plots
for idx, (method_name, (mean, std)) in enumerate(methods.items()):
    ax = axes[1, idx] if idx < 2 else axes[1, 1]
    
    # Calibration scatter plot
    errors = np.abs(mean - y_test_true)
    within_1std = errors <= std
    
    ax.scatter(std, errors, alpha=0.5, s=10, label='Predictions')
    ax.plot([0, std.max()], [0, std.max()], 'r--', label='Perfect calibration')
    ax.set_xlabel('Predicted Uncertainty (σ)')
    ax.set_ylabel('Actual Error')
    ax.set_title(f'{method_name} Calibration')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Print calibration metric
    coverage = np.mean(within_1std)
    print(f"{method_name} - 1σ coverage: {coverage:.2%} (expected: ~68%)")

plt.tight_layout()
plt.show()

print()

# Example 6: Practical Applications
print("Example 6: Out-of-Distribution Detection")
print("=" * 50)

# Generate out-of-distribution data
x_ood = np.linspace(5, 8, 50)
x_ood_tensor = torch.FloatTensor(x_ood).unsqueeze(1).to(device)

# Get predictions for OOD data
ood_results = {}
for method_name, model in [('MC Dropout', mc_model), ('Bayesian NN', bnn_model)]:
    if hasattr(model, 'predict_with_uncertainty'):
        mean, std, _ = model.predict_with_uncertainty(x_ood_tensor)
        ood_results[method_name] = (mean.cpu().numpy().flatten(), 
                                   std.cpu().numpy().flatten())

# Deep ensemble OOD
mean_ens_ood, std_ens_ood, _ = ensemble.predict_with_uncertainty(x_ood_tensor)
ood_results['Deep Ensemble'] = (mean_ens_ood.cpu().numpy().flatten(),
                                std_ens_ood.cpu().numpy().flatten())

# Visualize OOD detection
plt.figure(figsize=(12, 6))

# Training data region
plt.axvspan(x_epistemic.min(), x_epistemic.max(), alpha=0.1, color='green', 
            label='Training data region')

# Plot uncertainties
for (method_name, (_, std_in)), color in zip(methods.items(), colors):
    _, std_ood = ood_results[method_name]
    
    plt.plot(x, std_in, color=color, label=f'{method_name} (in-dist)', linewidth=2)
    plt.plot(x_ood, std_ood, color=color, linestyle='--', 
             label=f'{method_name} (OOD)', linewidth=2)

plt.xlabel('x')
plt.ylabel('Uncertainty (σ)')
plt.title('Out-of-Distribution Detection via Uncertainty')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()

# Calculate OOD detection metrics
for method_name in methods.keys():
    _, std_in = methods[method_name]
    _, std_ood = ood_results[method_name]
    
    threshold = np.percentile(std_in, 95)
    ood_detected = np.mean(std_ood > threshold)
    
    print(f"{method_name} - OOD detection rate: {ood_detected:.2%}")

print()

# Summary and Best Practices
print("Bayesian Deep Learning Summary")
print("=" * 50)

summary = """
Methods for Uncertainty Estimation:

1. Monte Carlo Dropout
   - Pros: Easy to implement, works with existing models
   - Cons: May underestimate uncertainty, requires tuning
   - Use when: Quick uncertainty estimates needed

2. Bayesian Neural Networks
   - Pros: Principled uncertainty, theoretical foundation
   - Cons: Computationally expensive, complex implementation
   - Use when: Accurate uncertainty crucial

3. Deep Ensembles
   - Pros: Simple, often best empirical performance
   - Cons: High computational cost, multiple models
   - Use when: Resources available, best performance needed

Best Practices:
- Always validate uncertainty estimates
- Consider computational constraints
- Use multiple metrics for evaluation
- Calibrate uncertainties when needed
- Test on out-of-distribution data

Applications:
- Medical diagnosis
- Autonomous systems
- Financial modeling
- Active learning
- Anomaly detection
"""

print(summary)