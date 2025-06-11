#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Model Deployment with PyTorch

This script demonstrates various methods for deploying PyTorch models
including TorchScript, ONNX export, quantization, and mobile deployment.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.quantization
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import numpy as np
import time
import os
import warnings
warnings.filterwarnings('ignore')

# Set random seed for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Create output directory
output_dir = "10_model_deployment_outputs"
os.makedirs(output_dir, exist_ok=True)

# -----------------------------------------------------------------------------
# Section 1: Simple Model for Deployment Examples
# -----------------------------------------------------------------------------

class SimpleConvNet(nn.Module):
    """A simple CNN for MNIST classification."""
    def __init__(self):
        super(SimpleConvNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)
        self.pool = nn.MaxPool2d(2)
        self.dropout = nn.Dropout(0.25)
        
    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

def train_simple_model():
    """Train a simple model on MNIST for deployment examples."""
    print("\nSection 1: Training a Simple Model for Deployment")
    print("-" * 70)
    
    # Load MNIST dataset
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    train_dataset = torchvision.datasets.MNIST(
        root='./data', train=True, download=True, transform=transform
    )
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    
    # Create and train model
    model = SimpleConvNet().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    print("Training model...")
    model.train()
    for epoch in range(2):  # Quick training for demo
        for batch_idx, (data, target) in enumerate(train_loader):
            if batch_idx > 100:  # Limit training for demo
                break
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
        print(f"Epoch {epoch+1} completed")
    
    # Save the trained model
    model_path = os.path.join(output_dir, "simple_model.pth")
    torch.save(model.state_dict(), model_path)
    print(f"Model saved to {model_path}")
    
    return model

# -----------------------------------------------------------------------------
# Section 2: TorchScript - Tracing and Scripting
# -----------------------------------------------------------------------------

def demonstrate_torchscript(model):
    """Demonstrate TorchScript conversion methods."""
    print("\nSection 2: TorchScript - Tracing and Scripting")
    print("-" * 70)
    
    model.eval()
    
    # Method 1: Tracing
    print("\n--- Method 1: Tracing ---")
    example_input = torch.randn(1, 1, 28, 28).to(device)
    traced_model = torch.jit.trace(model, example_input)
    
    # Save traced model
    traced_path = os.path.join(output_dir, "traced_model.pt")
    traced_model.save(traced_path)
    print(f"Traced model saved to {traced_path}")
    
    # Method 2: Scripting
    print("\n--- Method 2: Scripting ---")
    scripted_model = torch.jit.script(model)
    
    # Save scripted model
    scripted_path = os.path.join(output_dir, "scripted_model.pt")
    scripted_model.save(scripted_path)
    print(f"Scripted model saved to {scripted_path}")
    
    # Compare inference times
    print("\n--- Performance Comparison ---")
    test_input = torch.randn(100, 1, 28, 28).to(device)
    
    # Original model
    start = time.time()
    with torch.no_grad():
        _ = model(test_input)
    original_time = time.time() - start
    
    # Traced model
    start = time.time()
    with torch.no_grad():
        _ = traced_model(test_input)
    traced_time = time.time() - start
    
    print(f"Original model inference time: {original_time:.4f}s")
    print(f"Traced model inference time: {traced_time:.4f}s")
    print(f"Speedup: {original_time/traced_time:.2f}x")
    
    return traced_model, scripted_model

# -----------------------------------------------------------------------------
# Section 3: ONNX Export
# -----------------------------------------------------------------------------

def demonstrate_onnx_export(model):
    """Export model to ONNX format."""
    print("\nSection 3: ONNX Export")
    print("-" * 70)
    
    model.eval()
    
    # Prepare dummy input
    dummy_input = torch.randn(1, 1, 28, 28).to(device)
    
    # Export to ONNX
    onnx_path = os.path.join(output_dir, "model.onnx")
    
    torch.onnx.export(
        model,                       # model
        dummy_input,                 # model input
        onnx_path,                   # output path
        export_params=True,          # store trained params
        opset_version=11,            # ONNX version
        do_constant_folding=True,    # optimize constant folding
        input_names=['input'],       # input names
        output_names=['output'],     # output names
        dynamic_axes={               # variable length axes
            'input': {0: 'batch_size'},
            'output': {0: 'batch_size'}
        }
    )
    
    print(f"Model exported to ONNX: {onnx_path}")
    
    # Verify ONNX model
    try:
        import onnx
        onnx_model = onnx.load(onnx_path)
        onnx.checker.check_model(onnx_model)
        print("ONNX model verified successfully!")
        
        # Print model info
        print(f"ONNX model inputs: {[inp.name for inp in onnx_model.graph.input]}")
        print(f"ONNX model outputs: {[out.name for out in onnx_model.graph.output]}")
    except ImportError:
        print("ONNX not installed. Install with: pip install onnx")
    
    return onnx_path

# -----------------------------------------------------------------------------
# Section 4: Quantization
# -----------------------------------------------------------------------------

class QuantizableConvNet(nn.Module):
    """Quantization-friendly version of SimpleConvNet."""
    def __init__(self):
        super(QuantizableConvNet, self).__init__()
        self.quant = torch.quantization.QuantStub()
        self.dequant = torch.quantization.DeQuantStub()
        
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(2)
        
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(2)
        
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.relu3 = nn.ReLU()
        self.dropout = nn.Dropout(0.25)
        self.fc2 = nn.Linear(128, 10)
        
    def forward(self, x):
        x = self.quant(x)
        
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.pool1(x)
        
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.pool2(x)
        
        x = x.reshape(x.size(0), -1)
        x = self.fc1(x)
        x = self.relu3(x)
        x = self.dropout(x)
        x = self.fc2(x)
        
        x = self.dequant(x)
        return x

def demonstrate_quantization():
    """Demonstrate model quantization techniques."""
    print("\nSection 4: Model Quantization")
    print("-" * 70)
    
    # Create and prepare model
    model = QuantizableConvNet()
    model.eval()
    
    # Load some data for calibration
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    calibration_dataset = torchvision.datasets.MNIST(
        root='./data', train=False, download=True, transform=transform
    )
    calibration_loader = DataLoader(calibration_dataset, batch_size=32)
    
    # Dynamic Quantization
    print("\n--- Dynamic Quantization ---")
    dynamic_quantized_model = torch.quantization.quantize_dynamic(
        model, {nn.Linear}, dtype=torch.qint8
    )
    
    # Static Quantization
    print("\n--- Static Quantization ---")
    # Configure quantization
    model.qconfig = torch.quantization.get_default_qconfig('fbgemm')
    
    # Prepare model
    torch.quantization.prepare(model, inplace=True)
    
    # Calibrate with representative data
    print("Calibrating model...")
    with torch.no_grad():
        for batch_idx, (data, _) in enumerate(calibration_loader):
            if batch_idx > 10:  # Use limited data for calibration
                break
            model(data)
    
    # Convert to quantized model
    quantized_model = torch.quantization.convert(model, inplace=True)
    
    # Compare model sizes
    print("\n--- Model Size Comparison ---")
    
    # Original model size
    original_path = os.path.join(output_dir, "original_model.pth")
    torch.save(model.state_dict(), original_path)
    original_size = os.path.getsize(original_path) / 1024 / 1024  # MB
    
    # Quantized model size
    quantized_path = os.path.join(output_dir, "quantized_model.pth")
    torch.save(quantized_model.state_dict(), quantized_path)
    quantized_size = os.path.getsize(quantized_path) / 1024 / 1024  # MB
    
    print(f"Original model size: {original_size:.2f} MB")
    print(f"Quantized model size: {quantized_size:.2f} MB")
    print(f"Size reduction: {(1 - quantized_size/original_size)*100:.1f}%")
    
    # Compare inference speed
    print("\n--- Inference Speed Comparison ---")
    test_input = torch.randn(100, 1, 28, 28)
    
    # Original model
    start = time.time()
    with torch.no_grad():
        for _ in range(10):
            _ = model(test_input)
    original_time = time.time() - start
    
    # Quantized model
    start = time.time()
    with torch.no_grad():
        for _ in range(10):
            _ = quantized_model(test_input)
    quantized_time = time.time() - start
    
    print(f"Original model inference time: {original_time:.4f}s")
    print(f"Quantized model inference time: {quantized_time:.4f}s")
    print(f"Speedup: {original_time/quantized_time:.2f}x")
    
    return quantized_model

# -----------------------------------------------------------------------------
# Section 5: Mobile Deployment Preparation
# -----------------------------------------------------------------------------

def prepare_mobile_deployment(model):
    """Prepare model for mobile deployment."""
    print("\nSection 5: Mobile Deployment Preparation")
    print("-" * 70)
    
    model.eval()
    
    # Optimize for mobile
    print("Optimizing model for mobile...")
    example_input = torch.randn(1, 1, 28, 28)
    traced_model = torch.jit.trace(model, example_input)
    
    # Optimize for mobile
    from torch.utils.mobile_optimizer import optimize_for_mobile
    optimized_model = optimize_for_mobile(traced_model)
    
    # Save for mobile
    mobile_path = os.path.join(output_dir, "model_mobile.ptl")
    optimized_model._save_for_lite_interpreter(mobile_path)
    print(f"Mobile-optimized model saved to {mobile_path}")
    
    # Print mobile model info
    print("\nMobile deployment steps:")
    print("1. Add the .ptl file to your mobile app's assets")
    print("2. Use PyTorch Mobile SDK to load and run the model")
    print("3. Example Android code:")
    print("""
    Module module = Module.load(assetFilePath(this, "model_mobile.ptl"));
    Tensor inputTensor = Tensor.fromBlob(inputArray, new long[]{1, 1, 28, 28});
    Tensor outputTensor = module.forward(IValue.from(inputTensor)).toTensor();
    float[] scores = outputTensor.getDataAsFloatArray();
    """)
    
    return optimized_model

# -----------------------------------------------------------------------------
# Section 6: Model Serving with Flask (Example)
# -----------------------------------------------------------------------------

def create_flask_app_example():
    """Create example Flask app code for model serving."""
    print("\nSection 6: Model Serving Example")
    print("-" * 70)
    
    flask_code = '''
# app.py - Flask app for serving PyTorch model

from flask import Flask, request, jsonify
import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import io
import base64

app = Flask(__name__)

# Load model
model = torch.jit.load('traced_model.pt')
model.eval()

# Define preprocessing
transform = transforms.Compose([
    transforms.Resize((28, 28)),
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get image from request
        data = request.get_json()
        image_base64 = data['image']
        
        # Decode and preprocess image
        image_bytes = base64.b64decode(image_base64)
        image = Image.open(io.BytesIO(image_bytes)).convert('L')
        tensor = transform(image).unsqueeze(0)
        
        # Make prediction
        with torch.no_grad():
            output = model(tensor)
            probabilities = F.softmax(output, dim=1)
            prediction = torch.argmax(probabilities, dim=1).item()
            confidence = probabilities[0][prediction].item()
        
        return jsonify({
            'prediction': prediction,
            'confidence': confidence,
            'probabilities': probabilities[0].tolist()
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
'''
    
    # Save Flask app code
    flask_path = os.path.join(output_dir, "app.py")
    with open(flask_path, 'w') as f:
        f.write(flask_code)
    
    print(f"Flask app example saved to {flask_path}")
    print("\nTo run the Flask server:")
    print("1. Install Flask: pip install flask pillow")
    print("2. Run: python app.py")
    print("3. Send POST requests to http://localhost:5000/predict")
    
    # Create client example
    client_code = '''
# client.py - Example client for testing the Flask API

import requests
import base64
from PIL import Image
import io

def predict_image(image_path, server_url='http://localhost:5000/predict'):
    # Load and encode image
    with open(image_path, 'rb') as f:
        image_bytes = f.read()
    image_base64 = base64.b64encode(image_bytes).decode('utf-8')
    
    # Send request
    response = requests.post(
        server_url,
        json={'image': image_base64}
    )
    
    return response.json()

# Example usage
if __name__ == '__main__':
    result = predict_image('test_image.png')
    print(f"Prediction: {result['prediction']}")
    print(f"Confidence: {result['confidence']:.2%}")
'''
    
    client_path = os.path.join(output_dir, "client.py")
    with open(client_path, 'w') as f:
        f.write(client_code)
    
    print(f"Client example saved to {client_path}")

# -----------------------------------------------------------------------------
# Section 7: Best Practices and Tips
# -----------------------------------------------------------------------------

def deployment_best_practices():
    """Print deployment best practices and tips."""
    print("\nSection 7: Deployment Best Practices")
    print("-" * 70)
    
    practices = """
1. Model Optimization:
   - Use TorchScript for production deployment
   - Apply quantization for mobile/edge devices
   - Consider model pruning for size reduction
   
2. Input Preprocessing:
   - Ensure consistent preprocessing between training and deployment
   - Document input formats and expected ranges
   - Handle edge cases and validate inputs
   
3. Performance Considerations:
   - Batch inference requests when possible
   - Use GPU acceleration where available
   - Profile and optimize bottlenecks
   
4. Monitoring and Logging:
   - Log prediction metrics and latencies
   - Monitor model performance over time
   - Implement A/B testing for model updates
   
5. Security:
   - Validate and sanitize all inputs
   - Use HTTPS for API endpoints
   - Implement rate limiting and authentication
   
6. Version Control:
   - Version your models with semantic versioning
   - Maintain backward compatibility
   - Document model changes and performance metrics
   
7. Testing:
   - Test with representative data
   - Verify numerical accuracy after optimization
   - Test on target deployment hardware
"""
    
    print(practices)

# -----------------------------------------------------------------------------
# Main function
# -----------------------------------------------------------------------------

def main():
    """Main function to run all deployment examples."""
    print("=" * 80)
    print("PyTorch Model Deployment Tutorial")
    print("=" * 80)
    
    # Train a simple model
    model = train_simple_model()
    
    # Demonstrate TorchScript
    traced_model, scripted_model = demonstrate_torchscript(model)
    
    # Demonstrate ONNX export
    onnx_path = demonstrate_onnx_export(model)
    
    # Demonstrate quantization
    quantized_model = demonstrate_quantization()
    
    # Prepare for mobile deployment
    if device.type == 'cpu':  # Mobile optimization works best on CPU
        prepare_mobile_deployment(model)
    else:
        print("\nSkipping mobile deployment (requires CPU mode)")
    
    # Create Flask app example
    create_flask_app_example()
    
    # Best practices
    deployment_best_practices()
    
    print(f"\nAll deployment artifacts saved to '{output_dir}' directory")
    print("Tutorial complete!")

if __name__ == '__main__':
    main()