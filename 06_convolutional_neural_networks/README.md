# Convolutional Neural Networks (CNNs) in PyTorch

This tutorial provides a comprehensive guide to understanding and implementing Convolutional Neural Networks (CNNs) using PyTorch. CNNs are a class of deep neural networks most commonly applied to analyzing visual imagery, but also effective for other types of data like audio and text.

## Table of Contents
1. [Introduction to Convolutional Neural Networks](#introduction-to-convolutional-neural-networks)
   - What are CNNs and Why Use Them for Images?
   - Key Concepts: Local Receptive Fields, Shared Weights, Pooling
2. [Core CNN Layers and Components](#core-cnn-layers-and-components)
   - **Convolutional Layers (`nn.Conv2d`)**
     - Kernels (Filters): Size, Stride, Padding, Dilation
     - Input and Output Channels
     - Feature Maps
     - 2D Convolution Operation Explained
   - **Activation Functions (ReLU)**
     - Role in CNNs
   - **Pooling Layers (`nn.MaxPool2d`, `nn.AvgPool2d`)**
     - Purpose: Down-sampling, Dimensionality Reduction, Invariance
     - Max Pooling vs. Average Pooling
     - Kernel Size and Stride
   - **Fully Connected Layers (`nn.Linear`)**
     - Role in Classification/Regression after Convolutional Base
     - Flattening Feature Maps
   - **Batch Normalization (`nn.BatchNorm2d`)**
     - Normalizing Activations in CNNs
   - **Dropout (`nn.Dropout2d`, `nn.Dropout`)**
     - Regularization in CNNs
3. [Building a Basic CNN Architecture](#building-a-basic-cnn-architecture)
   - Stacking Convolutional, Activation, and Pooling Layers
   - Adding Fully Connected Layers for Classification
   - Example CNN for MNIST or CIFAR-10
4. [Training CNNs for Image Classification](#training-cnns-for-image-classification)
   - Data Preparation: Image Transforms and Augmentation specific to CNNs
   - Loss Function (e.g., `nn.CrossEntropyLoss`)
   - Optimizer (e.g., Adam, SGD)
   - The Training Loop (Revisiting with CNN context)
5. [Understanding and Implementing Famous CNN Architectures (Conceptual Overview)](#understanding-and-implementing-famous-cnn-architectures-conceptual-overview)
   - **LeNet-5:** A pioneering CNN.
   - **AlexNet:** Deepened the architecture, used ReLUs and Dropout.
   - **VGGNets:** Simplicity with deeper stacks of small (3x3) convolutions.
   - **GoogLeNet (Inception):** Introduced Inception modules for efficiency and multi-scale processing.
   - **ResNet (Residual Networks):** Introduced residual connections to train very deep networks.
   - (Implementation of one simple architecture like LeNet-5 will be in the .py script)
6. [Transfer Learning with Pre-trained CNN Models](#transfer-learning-with-pre-trained-cnn-models)
   - What is Transfer Learning?
   - Benefits: Reduced training time, better performance with less data.
   - Using Pre-trained Models from `torchvision.models` (e.g., ResNet, VGG).
   - **Feature Extraction:** Using the pre-trained CNN as a fixed feature extractor by freezing its weights and replacing the classifier head.
   - **Fine-tuning:** Unfreezing some of the later layers of the pre-trained model and training them with a smaller learning rate on the new dataset.
7. [Visualizing What CNNs Learn (Feature Visualization - Conceptual)](#visualizing-what-cnns-learn-feature-visualization---conceptual)
   - Understanding intermediate feature maps.
   - Visualizing Convolutional Filters (first layer).
   - Techniques like Saliency Maps, Class Activation Maps (CAM), Grad-CAM (Conceptual Overview).
8. [Practical Tips for Training CNNs](#practical-tips-for-training-cnns)
   - Data Augmentation is Key
   - Appropriate Learning Rates and Schedulers
   - Choosing Batch Size (considering GPU memory)
   - Regularization (Dropout, Weight Decay)
   - Monitoring Validation Performance

## Introduction to Convolutional Neural Networks

- **What are CNNs and Why Use Them for Images?**
  CNNs are specialized neural networks designed to process data with a grid-like topology, such as images (2D grid of pixels) or audio (1D grid of time samples). They are highly effective for image-related tasks because they can automatically and adaptively learn spatial hierarchies of features from low-level edges and textures to high-level object parts and concepts.
- **Key Concepts:**
  - **Local Receptive Fields:** Each neuron in a convolutional layer is connected to only a small region of the input volume (its local receptive field), allowing it to learn local features.
  - **Shared Weights (Parameter Sharing):** The same set of weights (kernel/filter) is used across different spatial locations in the input. This drastically reduces the number of parameters and makes the model equivariant to translations of features.
  - **Pooling:** Summarizes features in a neighborhood, providing a degree of translation invariance and reducing dimensionality.

## Core CNN Layers and Components

- **Convolutional Layers (`nn.Conv2d`)**
  The core building block of a CNN. It performs a convolution operation, sliding a learnable filter (kernel) over the input.
  - **Kernels (Filters):** Small matrices of learnable parameters. Each kernel is responsible for detecting a specific feature (e.g., an edge, a texture). The depth of the kernel matches the depth (number of channels) of its input.
  - **Input and Output Channels:** `in_channels` is the number of channels in the input volume (e.g., 3 for RGB images). `out_channels` is the number of filters applied, determining the depth of the output feature map.
  - **Feature Maps:** The output of a convolutional layer. Each channel in the output feature map corresponds to the response of a specific filter across the input.
  - **Parameters:**
    - `kernel_size (int or tuple)`: Size of the filter (e.g., 3 for 3x3, (3,5) for 3x5).
    - `stride (int or tuple, optional)`: Step size with which the filter slides over the input (default: 1).
    - `padding (int or tuple, optional)`: Amount of zero-padding added to the borders of the input (default: 0). Padding can help control the spatial size of the output feature map and preserve border information.
    - `dilation (int or tuple, optional)`: Spacing between kernel elements (default: 1).
  ```python
  import torch
  import torch.nn as nn

  # Example: Conv2d layer
  # Input: Batch of 16 images, 3 channels (RGB), 32x32 pixels
  # Output: 32 feature maps (output channels), spatial size depends on kernel, stride, padding
  conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, stride=1, padding=1)
  # input_tensor = torch.randn(16, 3, 32, 32) # Batch, Channels, Height, Width
  # output_feature_map = conv1(input_tensor)
  # print(f"Output feature map shape: {output_feature_map.shape}") # e.g., [16, 32, 32, 32]
  ```

- **Activation Functions (ReLU)**
  Typically, a non-linear activation function like ReLU (`nn.ReLU()`) is applied element-wise after each convolutional operation to introduce non-linearity.

- **Pooling Layers (`nn.MaxPool2d`, `nn.AvgPool2d`)**
  Reduce the spatial dimensions (height and width) of the feature maps, reducing computation and parameters, and providing a form of translation invariance.
  - `nn.MaxPool2d(kernel_size, stride=None)`: Selects the maximum value from each patch of the feature map covered by the pooling window.
  - `nn.AvgPool2d(kernel_size, stride=None)`: Computes the average value.
  ```python
  # pool = nn.MaxPool2d(kernel_size=2, stride=2) # Reduces H and W by factor of 2
  # pooled_output = pool(output_feature_map) # Assuming output_feature_map from conv1
  # print(f"Pooled output shape: {pooled_output.shape}") # e.g., [16, 32, 16, 16]
  ```

- **Fully Connected Layers (`nn.Linear`)**
  After several convolutional and pooling layers, the high-level features are typically flattened and fed into one or more fully connected layers for classification or regression.
  - **Flattening:** Converting the 3D feature maps (Channels x Height x Width) into a 1D vector.

- **Batch Normalization (`nn.BatchNorm2d`)**
  Applied after convolutional layers (and before or after activation) to normalize the activations across the batch. Helps stabilize training, allows higher learning rates, and can act as a regularizer.

- **Dropout (`nn.Dropout2d`, `nn.Dropout`)**
  `nn.Dropout2d` randomly zeros out entire channels during training. `nn.Dropout` (1D dropout) is used for fully connected layers. Helps prevent overfitting.

## Building a Basic CNN Architecture

A typical CNN architecture pattern:
`INPUT -> [[CONV -> ACT -> POOL] * N -> FLATTEN -> [FC -> ACT] * M -> FC (Output)]`

```python
class SimpleCNN(nn.Module):
    def __init__(self, num_classes=10):
        super(SimpleCNN, self).__init__()
        self.conv_block1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=16, kernel_size=5, stride=1, padding=2), # MNIST: 1 channel
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2) # Output: 16 x 14 x 14 (for 28x28 input)
        )
        self.conv_block2 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2) # Output: 32 x 7 x 7
        )
        # After two max pooling layers of stride 2, a 28x28 image becomes 7x7.
        # So, the flattened size is 32 (channels) * 7 * 7.
        self.fc = nn.Linear(32 * 7 * 7, num_classes)

    def forward(self, x): # Input x shape: [batch_size, 1, 28, 28] for MNIST
        x = self.conv_block1(x)
        x = self.conv_block2(x)
        x = x.view(x.size(0), -1) # Flatten the feature maps: [batch_size, 32*7*7]
        x = self.fc(x)
        return x # Raw logits for classification

# model_cnn = SimpleCNN(num_classes=10) # For MNIST (10 digits)
# print(model_cnn)
```

## Training CNNs for Image Classification

Training involves the same general steps as other neural networks, but with data and augmentations tailored for images.
- **Data Preparation:** Use `torchvision.transforms` for normalization, resizing, and data augmentation (random flips, rotations, crops, color jitter, etc.).
- **Loss Function:** `nn.CrossEntropyLoss` is standard for multi-class image classification.
- **Optimizer:** Adam or SGD with momentum are common choices.

## Understanding and Implementing Famous CNN Architectures (Conceptual Overview)

- **LeNet-5:** One of the earliest successful CNNs, designed for digit recognition.
- **AlexNet:** Won the ImageNet LSVRC-2012. Deeper than LeNet, used ReLU, Dropout, and data augmentation extensively.
- **VGGNets:** Showed that depth is critical. Used very small (3x3) convolutional filters stacked deeply.
- **GoogLeNet (Inception):** Introduced the "Inception module," which performs convolutions at multiple scales in parallel and concatenates their outputs, improving performance and computational efficiency.
- **ResNet (Residual Networks):** Enabled training of extremely deep networks (hundreds of layers) by introducing "residual connections" (skip connections) that allow gradients to propagate more easily.

## Transfer Learning with Pre-trained CNN Models

Leveraging models pre-trained on large datasets (like ImageNet) can significantly boost performance on smaller, related datasets.

- **`torchvision.models`:** Provides access to many pre-trained models (ResNet, VGG, Inception, MobileNet, etc.).
  ```python
  import torchvision.models as models
  # resnet18_pretrained = models.resnet18(pretrained=True) # PyTorch < 0.13
  # resnet18_pretrained = models.resnet18(weights=models.ResNet18_Weights.DEFAULT) # PyTorch >= 0.13
  ```
- **Feature Extraction:** Freeze the weights of the convolutional base of the pre-trained model and replace its final classification layer with a new one suited to your task. Train only the new classifier.
- **Fine-tuning:** Unfreeze some of the top layers of the pre-trained model in addition to training the new classifier. Use a small learning rate to avoid catastrophically forgetting the learned features.

## Visualizing What CNNs Learn (Feature Visualization - Conceptual)

Understanding the internal workings of CNNs can be aided by visualizing:
- **Filters:** Especially in the first layer, filters often learn to detect simple patterns like edges, corners, and color blobs.
- **Feature Maps (Activations):** Show which regions of an image activate certain filters/channels at different layers, revealing the hierarchical feature extraction process.
- **Saliency Maps/Class Activation Maps (CAM/Grad-CAM):** Highlight the image regions most influential in a model's prediction for a specific class.

## Practical Tips for Training CNNs
- Start with a standard architecture (e.g., ResNet variant) and pre-trained weights if applicable.
- Aggressive data augmentation is often very beneficial.
- Use appropriate learning rates, often starting higher and decaying (e.g., with a scheduler).
- Batch Normalization is generally helpful.
- Monitor training and validation metrics closely.

## Running the Tutorial

To run the Python script associated with this tutorial:
```bash
python convolutional_neural_networks.py
```
We recommend you manually create a `convolutional_neural_networks.ipynb` notebook and copy the code from the Python script into it for an interactive experience.

## Prerequisites
- Python 3.7+
- PyTorch 1.10+
- Torchvision
- NumPy
- Matplotlib (for visualization)

## Related Tutorials
1. [Data Loading and Preprocessing](../05_data_loading_preprocessing/README.md)
2. [Training Neural Networks](../04_training_neural_networks/README.md)
3. [Recurrent Neural Networks](../07_recurrent_neural_networks/README.md) (for sequence data) 