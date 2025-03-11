# PyTorch Tutorials

A comprehensive collection of tutorials covering essential PyTorch concepts and applications.

## Overview

This repository contains a series of tutorials designed to help you learn PyTorch from the ground up. Each tutorial includes detailed explanations, code examples, and practical applications. The tutorials are structured to progress from basic concepts to advanced techniques, making it suitable for beginners and experienced practitioners alike.

## Table of Contents

### Fundamentals
1. [PyTorch Basics](01_pytorch_basics/README.md)
   - Tensors, operations, and computational graphs
   - NumPy integration
   - GPU acceleration

2. [Autograd and Optimization](02_autograd_optimization/README.md)
   - Automatic differentiation
   - Gradient computation
   - Optimizers (SGD, Adam, etc.)

3. [Neural Network Basics](03_neural_network_basics/README.md)
   - Linear layers
   - Activation functions
   - Building a simple neural network
   - nn.Module and nn.Sequential

### Data Handling
4. [Data Loading and Processing](04_data_loading/README.md)
   - Dataset and DataLoader
   - Transforms and augmentation
   - Custom datasets
   - Batch processing

5. [Data Preprocessing](05_data_preprocessing/README.md)
   - Normalization techniques
   - Feature scaling
   - One-hot encoding
   - Handling missing data

### Computer Vision
6. [Convolutional Neural Networks](06_convolutional_networks/README.md)
   - CNN architecture
   - Convolution, pooling, and fully connected layers
   - Image classification
   - Transfer learning with pre-trained models

7. [Computer Vision Applications](07_computer_vision_applications/README.md)
   - Object detection
   - Semantic segmentation
   - Instance segmentation
   - Image generation

### Natural Language Processing
8. [Recurrent Neural Networks](08_recurrent_networks/README.md)
   - RNN architecture
   - LSTM and GRU
   - Sequence modeling
   - Text classification

9. [Transformers and Attention](09_transformers_attention/README.md)
   - Self-attention mechanism
   - Transformer architecture
   - BERT and GPT models
   - Fine-tuning pre-trained language models

### Advanced Topics
10. [Generative Models](10_generative_models/README.md)
    - Variational Autoencoders (VAEs)
    - Generative Adversarial Networks (GANs)
    - Diffusion models
    - Style transfer

11. [Reinforcement Learning](11_reinforcement_learning/README.md)
    - Policy gradients
    - Deep Q-Networks (DQN)
    - Actor-Critic methods
    - PyTorch integration with RL environments

12. [Model Optimization](12_model_optimization/README.md)
    - Quantization
    - Pruning
    - Knowledge distillation
    - Mixed precision training

### Deployment and Production
13. [Model Deployment](13_model_deployment/README.md)
    - TorchScript
    - ONNX export
    - Mobile deployment
    - Web deployment

14. [Distributed Training](14_distributed_training/README.md)
    - Data parallelism
    - Model parallelism
    - DistributedDataParallel
    - Multi-GPU training

15. [PyTorch Lightning](15_pytorch_lightning/README.md)
    - Lightning modules
    - Trainers and callbacks
    - Logging and checkpointing
    - Hyperparameter tuning

## Requirements

- Python 3.8+
- PyTorch 2.0+
- torchvision
- torchaudio (for audio tutorials)
- matplotlib
- numpy
- pandas
- scikit-learn
- Jupyter Notebook/Lab

You can install the required packages using:

```bash
pip install -r requirements.txt
```

## Getting Started

1. Clone this repository:
```bash
git clone https://github.com/niconielsen32/pytorch-tutorials.git
cd pytorch-tutorials
```

2. Install the requirements:
```bash
pip install -r requirements.txt
```

3. Navigate to any tutorial directory and open the Jupyter notebooks or Python scripts.

## How to Use This Repository

Each tutorial directory contains:
- A README.md file with explanations and theory
- Jupyter notebooks with code examples
- Python scripts for standalone execution
- Sample data or instructions to download datasets

You can follow the tutorials sequentially for a comprehensive learning experience or jump to specific topics based on your interests and needs.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- PyTorch team for the amazing framework
- The deep learning community for continuous innovation
- All contributors to this repository