# PyTorch Tutorials

A comprehensive collection of PyTorch tutorials from beginner to expert level. This repository aims to provide practical, hands-on examples and explanations for various PyTorch concepts and applications.

## 🚀 Quick Start

### Installation
```bash
git clone https://github.com/niconielsen32/pytorch-tutorials.git
cd pytorch-tutorials
pip install -r requirements.txt
```

### Running the Tutorials
```bash
# Run Python scripts directly
python 01_pytorch_basics/pytorch_basics.py

# Or use Jupyter notebooks for interactive learning
jupyter notebook
# Then navigate to any tutorial folder and open the .ipynb file
```

## 📚 Table of Contents

### **Fundamentals**

#### Beginner Level

1. **[PyTorch Basics](01_pytorch_basics/)**
   - Tensors, operations, and computational graphs
   - NumPy integration
   - GPU acceleration
   - Basic autograd operations

2. **[Neural Networks Fundamentals](02_neural_networks_fundamentals/)**
   - Linear layers, activation functions, loss functions, optimizers
   - Building your first neural network
   - Forward and backward propagation
   - nn.Module and nn.Sequential

3. **[Automatic Differentiation](03_automatic_differentiation/)**
   - Autograd mechanics
   - Computing gradients
   - Custom autograd functions
   - Higher-order derivatives

4. **[Training Neural Networks](04_training_neural_networks/)**
   - Training loop implementation
   - Validation techniques
   - Hyperparameter tuning
   - Learning rate scheduling
   - Early stopping

5. **[Data Loading and Preprocessing](05_data_loading_preprocessing/)**
   - Dataset and DataLoader classes
   - Custom datasets
   - Data transformations and augmentation
   - Efficient data loading techniques
   - Batch processing

### **Computer Vision**

#### Intermediate Level

6. **[Convolutional Neural Networks](06_convolutional_neural_networks/)**
   - CNN architecture components
   - Convolution, pooling, and fully connected layers
   - Image classification with CNNs
   - Transfer learning with pre-trained models
   - Feature visualization

#### Advanced Computer Vision Applications
- Object detection (YOLO, R-CNN)
- Semantic segmentation
- Instance segmentation
- Image generation
- Style transfer

### **Natural Language Processing**

7. **[Recurrent Neural Networks](07_recurrent_neural_networks/)**
   - RNN architecture
   - LSTM and GRU implementations
   - Sequence modeling
   - Text classification
   - Text generation
   - Time series forecasting

8. **[Transformers and Attention Mechanisms](08_transformers_and_attention_mechanisms/)**
   - Self-attention and multi-head attention
   - Transformer architecture
   - BERT and GPT model implementations
   - Fine-tuning pre-trained transformers
   - Positional encoding

### **Advanced Topics**

#### Advanced Level

9. **[Generative Models](09_generative_models/)**
   - Autoencoders
   - Variational Autoencoders (VAEs)
   - Generative Adversarial Networks (GANs)
   - Diffusion models
   - Style transfer

10. **[Model Deployment](10_model_deployment/)**
    - TorchScript and tracing
    - ONNX export
    - Quantization techniques
    - Mobile deployment (PyTorch Mobile)
    - Web deployment (ONNX.js)
    - Model serving

11. **[PyTorch Lightning](11_pytorch_lightning/)**
    - Lightning modules
    - Trainers and callbacks
    - Multi-GPU training
    - Experiment logging
    - Hyperparameter tuning with Lightning

12. **[Distributed Training](12_distributed_training/)**
    - Data Parallel (DP) for single-machine multi-GPU
    - Distributed Data Parallel (DDP) for multi-node training
    - Model Parallel for large models
    - Pipeline Parallelism for deep networks
    - Fully Sharded Data Parallel (FSDP) for extreme scale

### **Additional Advanced Topics**

13. **[Custom Extensions](13_custom_extensions/)**
    - C++ extensions for custom operations
    - CUDA kernels for GPU acceleration
    - Custom autograd functions
    - JIT compilation with TorchScript
    - Binding C++/CUDA code to Python

14. **[Performance Optimization](14_performance_optimization/)**
    - Memory optimization techniques
    - Mixed precision training with AMP
    - Profiling and benchmarking
    - Data loading optimization
    - Gradient accumulation and checkpointing

15. **[Advanced Model Architectures](15_advanced_model_architectures/)**
    - Graph Neural Networks (GNNs)
    - Vision Transformers (ViT)
    - EfficientNet and compound scaling
    - Neural ODEs
    - Capsule Networks

16. **[Reinforcement Learning](16_reinforcement_learning/)**
    - Deep Q-Networks (DQN)
    - Policy gradient methods (REINFORCE)
    - Actor-Critic and A2C
    - Proximal Policy Optimization (PPO)
    - Integration with OpenAI Gym

17. **[Model Optimization Techniques](17_model_optimization_techniques/)**
    - Quantization (dynamic and static)
    - Pruning (structured and unstructured)
    - Knowledge distillation
    - Model compression
    - Hardware-aware optimization

18. **[Meta-Learning and Few-Shot Learning](18_meta_learning/)**
    - Model-Agnostic Meta-Learning (MAML)
    - Prototypical Networks
    - Matching Networks
    - Reptile algorithm
    - Few-shot classification tasks

### **Expert Level Topics**

19. **[Neural Architecture Search](19_neural_architecture_search/)**
    - Random search and grid search
    - Evolutionary algorithms
    - Differentiable Architecture Search (DARTS)
    - Efficient Neural Architecture Search (ENAS)
    - Performance prediction

20. **[Bayesian Deep Learning](20_bayesian_deep_learning/)**
    - Bayesian Neural Networks
    - Variational inference
    - Monte Carlo Dropout
    - Deep ensembles
    - Uncertainty quantification

21. **[Advanced Research Topics](21_advanced_research_topics/)**
    - Self-supervised learning (SimCLR, BYOL)
    - Contrastive learning methods
    - Diffusion models
    - Neural Radiance Fields (NeRF)
    - Implicit neural representations

## 📋 Each Tutorial Includes

- **📖 README.md** - Detailed theory and concepts
- **🐍 Python Script** - Complete runnable code with comments
- **📓 Jupyter Notebook** - Interactive step-by-step learning

## 🛠️ Requirements

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

## 📖 How to Use This Repository

1. **Sequential Learning**: Follow the tutorials in order for a comprehensive learning experience
2. **Topic-Based**: Jump to specific topics based on your interests and needs
3. **Practice**: Each tutorial contains exercises and examples
4. **Experiment**: Modify the code and experiment with different parameters

### Getting Started

1. **Start with the README** in each folder for theoretical background
2. **Run the Python script** to see the complete implementation
3. **Open the Jupyter notebook** for interactive learning and experimentation

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes, please open an issue first to discuss what you would like to change.

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- PyTorch team for the amazing framework
- The deep learning community for continuous innovation
- All contributors to this repository

---

Perfect for both beginners starting their PyTorch journey and experts looking to deepen their understanding of advanced topics!