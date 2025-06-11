# Data Loading, Preprocessing, and Augmentation in PyTorch

This tutorial provides a comprehensive guide to efficiently loading, preprocessing, and augmenting data in PyTorch. Effective data handling is a critical step in any machine learning pipeline, ensuring that your model receives data in the correct format and benefits from techniques that can improve generalization.

## Table of Contents
1. [Introduction: The Importance of Data Handling](#introduction-the-importance-of-data-handling)
2. [PyTorch `Dataset` Class](#pytorch-dataset-class)
   - Role and Purpose
   - Key Methods: `__init__`, `__len__`, `__getitem__`
   - Using Built-in Datasets (e.g., `torchvision.datasets.MNIST`, `CIFAR10`)
3. [Creating Custom `Dataset`s](#creating-custom-datasets)
   - For Image Data (e.g., from a folder of images, from a CSV file with paths)
   - For Text Data (e.g., loading text files, tokenization basics)
   - For Other Data Types (e.g., CSV, time series)
4. [PyTorch `DataLoader` Class](#pytorch-dataloader-class)
   - Purpose: Batching, Shuffling, Parallel Loading
   - Key Parameters: `dataset`, `batch_size`, `shuffle`, `num_workers`, `pin_memory`
   - Iterating Through a `DataLoader`
5. [Data Transformations (`torchvision.transforms`)](#data-transformations-torchvisiontransforms)
   - Common Transformations for Images:
     - `transforms.ToTensor()`: Converting PIL Images/NumPy arrays to Tensors.
     - `transforms.Normalize()`: Normalizing tensor images.
     - Resizing, Cropping (`transforms.Resize`, `transforms.CenterCrop`, `transforms.RandomResizedCrop`)
     - `transforms.Compose()`: Chaining multiple transformations.
   - Creating Custom Transformations
6. [Data Augmentation](#data-augmentation)
   - Why Augment Data? Improving Model Robustness and Generalization.
   - Image Augmentation Techniques (using `torchvision.transforms`):
     - Random Flips (`transforms.RandomHorizontalFlip`, `transforms.RandomVerticalFlip`)
     - Random Rotations (`transforms.RandomRotation`)
     - Color Jitter (`transforms.ColorJitter`)
     - Random Affine Transformations (`transforms.RandomAffine`)
   - Integrating Augmentations into the `Dataset` or `DataLoader` Flow
   - Advanced Augmentation Libraries (e.g., Albumentations - conceptual mention)
7. [Working with Different Data Types](#working-with-different-data-types)
   - **Image Data:** Loading, common formats, channel orders.
   - **Text Data:** Tokenization, padding, creating vocabulary, embedding lookups (conceptual).
   - **Tabular Data:** Loading from CSV/Pandas, feature engineering, encoding categorical features (conceptual).
8. [Efficient Data Loading Techniques](#efficient-data-loading-techniques)
   - `num_workers` in `DataLoader`: Parallelizing data loading.
   - `pin_memory=True` in `DataLoader`: Faster CPU-to-GPU data transfer.
   - Pre-fetching and Caching Strategies (Conceptual)
   - Considerations for Large Datasets that Don't Fit in Memory
9. [Practical Example: Image Classification Dataset](#practical-example-image-classification-dataset)
   - Setting up a custom image folder dataset.
   - Applying transformations and augmentations.
   - Using `DataLoader` for training.

## Introduction: The Importance of Data Handling

Raw data is rarely in a format suitable for direct input into a neural network. Data loading and preprocessing involve several steps:
- **Loading:** Reading data from various sources (files, databases).
- **Preprocessing:** Cleaning, transforming, and structuring data (e.g., resizing images, tokenizing text, normalizing features).
- **Augmentation:** Artificially expanding the dataset by creating modified versions of existing data (e.g., rotating images, paraphrasing text) to improve model generalization and reduce overfitting.
Efficient data handling is crucial for training performance, as data loading can become a bottleneck if not optimized.

## PyTorch `Dataset` Class

- **Role and Purpose:** `torch.utils.data.Dataset` is an abstract class representing a dataset. All datasets in PyTorch that interact with `DataLoader` should inherit from this class.
- **Key Methods:**
  - `__init__(self, ...)`: Initializes the dataset (e.g., loads data paths, labels, performs initial setup).
  - `__len__(self)`: Returns the total number of samples in the dataset.
  - `__getitem__(self, idx)`: Loads and returns a single sample from the dataset at the given index `idx`. This is where transformations are often applied.
- **Using Built-in Datasets:** `torchvision.datasets` provides many common datasets like MNIST, CIFAR10, ImageNet, which are subclasses of `Dataset`.

```python
import torchvision
import torchvision.transforms as transforms

# Example: Using torchvision.datasets.MNIST
mnist_train_raw = torchvision.datasets.MNIST(root='./data', train=True, download=True)
sample_raw, label_raw = mnist_train_raw[0]
print(f"MNIST raw sample type: {type(sample_raw)}, Label: {label_raw}")

# Applying a transform to convert PIL Image to Tensor
mnist_train_transformed = torchvision.datasets.MNIST(
    root='./data', 
    train=True, 
    download=True, 
    transform=transforms.ToTensor() # Converts PIL Image to FloatTensor
)
sample_tensor, label_tensor = mnist_train_transformed[0]
print(f"MNIST transformed sample type: {type(sample_tensor)}, shape: {sample_tensor.shape}, Label: {label_tensor}")
```

## Creating Custom `Dataset`s

For most real-world applications, you'll need to create your own custom `Dataset`.

- **For Image Data:** Often involves reading image files (e.g., JPEG, PNG) and their corresponding labels.
  ```python
  from torch.utils.data import Dataset
  from PIL import Image # Pillow library for image manipulation
  import os

  class CustomImageDataset(Dataset):
      def __init__(self, img_dir, transform=None, target_transform=None):
          # Example: img_dir contains subfolders for each class (e.g., img_dir/cat/cat1.jpg)
          self.img_labels = [] # List of (image_path, class_index)
          self.classes = sorted(entry.name for entry in os.scandir(img_dir) if entry.is_dir())
          self.class_to_idx = {cls_name: i for i, cls_name in enumerate(self.classes)}
          
          for class_name in self.classes:
              class_dir = os.path.join(img_dir, class_name)
              for img_name in os.listdir(class_dir):
                  self.img_labels.append((os.path.join(class_dir, img_name), self.class_to_idx[class_name]))
          
          self.transform = transform
          self.target_transform = target_transform

      def __len__(self):
          return len(self.img_labels)

      def __getitem__(self, idx):
          img_path, label = self.img_labels[idx]
          image = Image.open(img_path).convert("RGB") # Ensure 3 channels
          if self.transform:
              image = self.transform(image)
          if self.target_transform:
              label = self.target_transform(label)
          return image, label
  ```
- **For Text Data:** Might involve reading lines from files, tokenizing text into numerical representations, and padding sequences.

## PyTorch `DataLoader` Class

- **Purpose:** `torch.utils.data.DataLoader` takes a `Dataset` object and provides an iterable to easily access batches of data. It automates batching, shuffling, and can use multiple worker processes for parallel data loading.
- **Key Parameters:**
  - `dataset`: The `Dataset` object from which to load the data.
  - `batch_size (int, optional)`: How many samples per batch to load (default: 1).
  - `shuffle (bool, optional)`: Set to `True` to have the data reshuffled at every epoch (default: `False`).
  - `num_workers (int, optional)`: How many subprocesses to use for data loading. 0 means that the data will be loaded in the main process (default: 0).
  - `pin_memory (bool, optional)`: If `True`, the `DataLoader` will copy Tensors into CUDA pinned memory before returning them. Useful for faster CPU to GPU transfers.

```python
from torch.utils.data import DataLoader

# Assuming mnist_train_transformed is an instance of a Dataset
# train_loader = DataLoader(mnist_train_transformed, batch_size=64, shuffle=True, num_workers=2)

# Iterating through a DataLoader
# for epoch in range(num_epochs):
#     for i, (inputs, labels) in enumerate(train_loader):
#         # inputs and labels are now batches of data
#         # Move to device: inputs, labels = inputs.to(device), labels.to(device)
#         # ... training logic ...
#         if i % 100 == 0:
#             print(f"Epoch {epoch}, Batch {i}, Input shape: {inputs.shape}")
```

## Data Transformations (`torchvision.transforms`)

`torchvision.transforms` provides common image transformations. They can be chained together using `transforms.Compose()`.

- **Common Transformations:**
  - `transforms.ToTensor()`: Converts a PIL Image or `numpy.ndarray` (H x W x C) in the range [0, 255] to a `torch.FloatTensor` of shape (C x H x W) in the range [0.0, 1.0].
  - `transforms.Normalize(mean, std)`: Normalizes a tensor image with mean and standard deviation. `output[channel] = (input[channel] - mean[channel]) / std[channel]`.
  - `transforms.Resize(size)`: Resizes the input PIL Image to the given size.
  - `transforms.CenterCrop(size)`: Crops the given PIL Image at the center.

```python
# Example of composing transformations
image_transforms = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) # ImageNet stats
])

# my_dataset = CustomImageDataset(..., transform=image_transforms)
```

## Data Augmentation

Data augmentation artificially increases the training set size by creating modified copies of its data. This helps the model become more robust to variations and reduces overfitting.

- **Image Augmentation Techniques:**
  - `transforms.RandomHorizontalFlip(p=0.5)`
  - `transforms.RandomRotation(degrees)`
  - `transforms.ColorJitter(brightness=0, contrast=0, saturation=0, hue=0)`
  - `transforms.RandomResizedCrop(size)`: Crops a random part of an image and resizes it.

```python
# Example augmentation pipeline for training
train_transforms_augmented = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    transforms.RandomRotation(degrees=15),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
# For validation/testing, typically only use non-random transformations like Resize, CenterCrop, ToTensor, Normalize.
```

## Working with Different Data Types
Conceptual overview; detailed implementations depend on the specific task.
- **Image Data:** Use PIL/OpenCV for loading, `torchvision.transforms` for preprocessing/augmentation. Pay attention to channel order (e.g., RGB vs BGR) and normalization.
- **Text Data:** Involves tokenization (splitting text into words/subwords), numericalization (mapping tokens to integers), padding sequences to the same length, and often using pre-trained embeddings or an `nn.Embedding` layer.
- **Tabular Data:** Often loaded using Pandas. Numerical features might need scaling/normalization. Categorical features need encoding (e.g., one-hot encoding, label encoding, or embedding layers).

## Efficient Data Loading Techniques

- **`num_workers > 0`:** Spawns multiple subprocesses to load data in parallel, preventing the main training process from waiting for data I/O.
- **`pin_memory=True`:** If using GPUs, setting this to `True` in `DataLoader` tells PyTorch to put fetched data Tensors in pinned (page-locked) memory. This enables faster data transfer from CPU to GPU memory via Direct Memory Access (DMA).
- **Caching/Pre-fetching:** For very large datasets or slow storage, caching frequently accessed data or pre-fetching next batches can help.

## Practical Example: Image Classification Dataset

This section will be detailed in the accompanying Python script (`data_loading_preprocessing.py`) and Jupyter Notebook, showing an end-to-end example of loading an image dataset from folders, applying transformations, and using `DataLoader`.

## Running the Tutorial

To run the Python script associated with this tutorial:
```bash
python data_loading_preprocessing.py
```
We recommend you manually create a `data_loading_preprocessing.ipynb` notebook and copy the code from the Python script into it for an interactive experience.

## Prerequisites
- Python 3.7+
- PyTorch 1.10+
- Torchvision (for built-in datasets and transforms)
- Pillow (PIL Fork, usually a dependency of Torchvision: `pip install Pillow`)
- NumPy

## Related Tutorials
1. [PyTorch Basics](../01_pytorch_basics/README.md)
2. [Training Neural Networks](../04_training_neural_networks/README.md)
3. [Convolutional Neural Networks](../06_convolutional_neural_networks/README.md) 