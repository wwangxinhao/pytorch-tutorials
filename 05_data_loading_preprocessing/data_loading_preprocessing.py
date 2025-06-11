#!/usr/bin/env python
# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data import Dataset, DataLoader, random_split
from PIL import Image
import os
import pandas as pd
from pathlib import Path
import glob
import time

"""
Data Loading, Preprocessing, and Augmentation in PyTorch

This script demonstrates how to use PyTorch's Dataset and DataLoader classes,
create custom datasets, apply transformations and augmentations for images,
and discusses efficient data loading techniques.
"""

# Set random seed for reproducibility (though less critical for data loading demo itself)
torch.manual_seed(42)
np.random.seed(42)

# Device configuration (not heavily used in this script, but good practice)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Create output directory for plots or temporary data if needed
output_dir = "05_data_loading_preprocessing_outputs"
os.makedirs(output_dir, exist_ok=True)

# ====================================
# 1. Introduction: The Importance of Data Handling (Conceptual - in README)
# ====================================
def intro_data_handling():
    print("\nSection 1: Introduction to Data Handling")
    print("-" * 70)
    print("This section is conceptual and detailed in the README.md.")
    print("Covers: Loading, Preprocessing, Augmentation, Efficiency.")

# ====================================
# 2. PyTorch `Dataset` Class - Using Built-in Datasets
# ====================================
def demonstrate_builtin_datasets():
    print("\nSection 2: PyTorch `Dataset` Class - Using Built-in Datasets")
    print("-" * 70)
    print("PyTorch provides many built-in datasets in `torchvision.datasets`.")

    # Example 1: MNIST without any transformations (returns PIL Image)
    print("\n--- MNIST Dataset (Raw PIL Images) ---")
    mnist_train_raw = torchvision.datasets.MNIST(
        root='./data', 
        train=True, 
        download=True, 
        transform=None # No transform, returns PIL Image
    )
    raw_sample, raw_label = mnist_train_raw[0] # Get the first sample
    print(f"Type of raw MNIST sample: {type(raw_sample)}")
    print(f"Mode of raw PIL Image: {raw_sample.mode}, Size: {raw_sample.size}")
    print(f"Label of first sample: {raw_label} (type: {type(raw_label)})")
    # raw_sample.save(os.path.join(output_dir, "mnist_raw_sample.png")) # Optionally save

    # Example 2: MNIST with ToTensor transformation
    print("\n--- MNIST Dataset (Transformed to Tensors) ---")
    mnist_train_totensor = torchvision.datasets.MNIST(
        root='./data', 
        train=True, 
        download=True, 
        transform=transforms.ToTensor() # Converts PIL Image or numpy.ndarray to torch.FloatTensor
    )
    tensor_sample, tensor_label = mnist_train_totensor[0]
    print(f"Type of transformed MNIST sample: {type(tensor_sample)}")
    print(f"Shape of tensor sample: {tensor_sample.shape} (C x H x W)")
    print(f"Data type of tensor sample: {tensor_sample.dtype}")
    print(f"Min value in tensor: {tensor_sample.min():.4f}, Max value: {tensor_sample.max():.4f}") # Should be [0,1]
    print(f"Label of first transformed sample: {tensor_label}")
    print("`Dataset` subclasses must implement `__len__` and `__getitem__`.")

# ====================================
# 3. Creating Custom `Dataset`s
# ====================================

# --- Custom Image Dataset Example (from folder structure) ---
# Create dummy image data for the custom dataset example
def create_dummy_image_folder_dataset(base_path, num_classes=2, imgs_per_class=5):
    if os.path.exists(base_path) and len(os.listdir(base_path)) > 0:
        print(f"Dummy image dataset already exists at '{base_path}'. Skipping creation.")
        return
    print(f"Creating dummy image dataset at '{base_path}'...")
    os.makedirs(base_path, exist_ok=True)
    for i in range(num_classes):
        class_name = f"class_{i}"
        class_dir = os.path.join(base_path, class_name)
        os.makedirs(class_dir, exist_ok=True)
        for j in range(imgs_per_class):
            # Create a small random PIL image
            try:
                img = Image.fromarray((np.random.rand(32, 32, 3) * 255).astype(np.uint8))
                img.save(os.path.join(class_dir, f"img_{j}.png"))
            except Exception as e:
                print(f"Could not create/save dummy image: {e}")
    print("Dummy image dataset created.")

class CustomImageFolderDataset(Dataset):
    """Custom Dataset for loading images from a folder structure (e.g., root/class_a/img1.png)."""
    def __init__(self, img_dir, transform=None, target_transform=None):
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform
        
        self.classes = sorted(entry.name for entry in os.scandir(img_dir) if entry.is_dir())
        self.class_to_idx = {cls_name: i for i, cls_name in enumerate(self.classes)}
        
        self.img_paths_labels = []
        for class_name in self.classes:
            class_path = os.path.join(img_dir, class_name)
            for img_name in glob.glob(os.path.join(class_path, "*.png")):
                 self.img_paths_labels.append((img_name, self.class_to_idx[class_name]))
        print(f"CustomImageFolderDataset: Found {len(self.img_paths_labels)} images in {len(self.classes)} classes.")

    def __len__(self):
        return len(self.img_paths_labels)

    def __getitem__(self, idx):
        img_path, label = self.img_paths_labels[idx]
        try:
            image = Image.open(img_path).convert("RGB") # Ensure 3 channels, robust loading
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
            # Return a placeholder or skip, depending on strategy
            return None, None 

        if self.transform:
            image = self.transform(image)
        
        if self.target_transform:
            label = self.target_transform(label) # Usually labels are already numbers
            
        return image, torch.tensor(label, dtype=torch.long)

# --- Custom Text Dataset Example (Conceptual) ---
class CustomTextDataset(Dataset):
    """Conceptual custom Dataset for text data."""
    def __init__(self, texts, labels, tokenizer, max_length):
        self.texts = texts # List of text strings
        self.labels = labels # List of corresponding labels
        self.tokenizer = tokenizer # e.g., from Hugging Face or custom
        self.max_length = max_length
        print(f"CustomTextDataset: Loaded {len(texts)} text samples.")

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        
        # Tokenization example (actual implementation depends on tokenizer)
        # encoding = self.tokenizer(text, truncation=True, padding='max_length', max_length=self.max_length, return_tensors='pt')
        # input_ids = encoding['input_ids'].squeeze(0) # Remove batch dimension
        # attention_mask = encoding['attention_mask'].squeeze(0)
        
        # For this demo, let's simulate tokenized output
        simulated_input_ids = torch.randint(0, 1000, (self.max_length,)) # vocab size 1000
        simulated_attention_mask = torch.ones(self.max_length)

        return {
            'input_ids': simulated_input_ids,
            'attention_mask': simulated_attention_mask,
            'label': torch.tensor(label, dtype=torch.long)
        }

def demonstrate_custom_datasets():
    print("\nSection 3: Creating Custom `Dataset`s")
    print("-" * 70)

    # --- Image Dataset Demo ---
    print("\n--- Custom Image Folder Dataset Demo ---")
    dummy_image_root = os.path.join(output_dir, "dummy_image_dataset")
    create_dummy_image_folder_dataset(dummy_image_root, num_classes=2, imgs_per_class=3)
    
    # Define a simple transform for the custom image dataset
    custom_image_transform = transforms.Compose([
        transforms.Resize((64, 64)), # Resize images
        transforms.ToTensor()         # Convert to tensor
    ])
    custom_image_ds = CustomImageFolderDataset(img_dir=dummy_image_root, transform=custom_image_transform)
    if len(custom_image_ds) > 0:
        img_sample, img_label = custom_image_ds[0]
        if img_sample is not None:
             print(f"Custom image dataset - first sample shape: {img_sample.shape}, label: {img_label}")
        else:
            print("Failed to load sample from custom image dataset.")
    else:
        print("Custom image dataset is empty or failed to initialize properly.")

    # --- Text Dataset Demo (Conceptual) ---
    print("\n--- Custom Text Dataset Demo (Conceptual) ---")
    dummy_texts = ["This is a great movie!", "I did not like this film.", "PyTorch is fun."]
    dummy_labels = [1, 0, 1] # Positive, Negative, Positive
    # In a real scenario, tokenizer would be from HuggingFace, SpaCy, etc.
    dummy_tokenizer = lambda x: x.lower().split() # Very simple tokenizer for concept
    custom_text_ds = CustomTextDataset(dummy_texts, dummy_labels, dummy_tokenizer, max_length=10)
    if len(custom_text_ds) > 0:
        text_sample_dict = custom_text_ds[0]
        print(f"Custom text dataset - first sample input_ids shape: {text_sample_dict['input_ids'].shape}")
        print(f"Custom text dataset - first sample label: {text_sample_dict['label']}")
    else:
        print("Custom text dataset is empty.")
    print("Custom Datasets allow flexible data loading for diverse data types.")

# ====================================
# 4. PyTorch `DataLoader` Class
# ====================================
def demonstrate_dataloader():
    print("\nSection 4: PyTorch `DataLoader` Class")
    print("-" * 70)
    print("`DataLoader` provides iterable batches, shuffling, and parallel loading.")

    mnist_dataset = torchvision.datasets.MNIST(
        root='./data', train=True, download=False, transform=transforms.ToTensor()
    ) # Assumes downloaded by demonstrate_builtin_datasets
    
    print(f"Total MNIST training samples: {len(mnist_dataset)}")

    # DataLoader with batch_size and shuffle
    print("\n--- DataLoader with batch_size=64, shuffle=True ---")
    train_loader_shuffle = DataLoader(mnist_dataset, batch_size=64, shuffle=True)
    print(f"Number of batches with shuffle: {len(train_loader_shuffle)}")
    first_batch_inputs, first_batch_labels = next(iter(train_loader_shuffle))
    print(f"First batch input shape: {first_batch_inputs.shape}") # [batch_size, C, H, W]
    print(f"First batch labels shape: {first_batch_labels.shape}")

    # DataLoader with num_workers for parallel loading
    # Note: num_workers > 0 might not work well in some interactive environments (like some notebooks on Windows)
    # It's best used in scripts. For demo, we can set it to 0 or 1.
    num_workers_demo = 0 # Set to 2 or more to see effect, if environment supports it.
    if os.name == 'posix': # Multi-processing works better on Linux/macOS for PyTorch DataLoader
        num_workers_demo = 2
        
    print(f"\n--- DataLoader with num_workers={num_workers_demo} (Parallel Loading) ---")
    start_time = time.time()
    # Iterate through a few batches to demonstrate loading time (will be very fast for MNIST anyway)
    train_loader_workers = DataLoader(mnist_dataset, batch_size=256, shuffle=True, num_workers=num_workers_demo, pin_memory=(device.type=='cuda'))
    for i, (batch_inputs, _) in enumerate(train_loader_workers):
        if i >= 5: # Load first 5 batches
            break
        # Simulate moving to device if applicable (pin_memory helps here for GPU)
        # batch_inputs = batch_inputs.to(device, non_blocking=pin_memory)
    end_time = time.time()
    print(f"Time to load 5 batches with num_workers={num_workers_demo}: {end_time - start_time:.4f}s")
    print("`num_workers` can significantly speed up data loading by using multiple CPU cores.")
    print("`pin_memory=True` can speed up CPU to GPU transfers.")

# ====================================
# 5 & 6. Data Transformations and Augmentation (`torchvision.transforms`)
# ====================================
def demonstrate_transforms_augmentation():
    print("\nSection 5 & 6: Data Transformations and Augmentation")
    print("-" * 70)
    print("`torchvision.transforms` provides common image transformations and augmentations.")

    # Load a sample MNIST image as PIL to apply transforms
    mnist_pil_dataset = torchvision.datasets.MNIST(root='./data', train=True, download=False, transform=None)
    pil_image, _ = mnist_pil_dataset[0]
    # pil_image.save(os.path.join(output_dir, "mnist_sample_for_transform.png"))

    print(f"Original PIL Image size: {pil_image.size}")

    # --- Common Transformations ---
    print("\n--- Common Transformations ---")
    # 1. Resize
    resize_transform = transforms.Resize((100, 100))
    resized_image = resize_transform(pil_image)
    print(f"Resized image size: {resized_image.size}")

    # 2. CenterCrop
    center_crop_transform = transforms.CenterCrop(60)
    cropped_image = center_crop_transform(resized_image)
    print(f"Center-cropped image size: {cropped_image.size}")

    # 3. ToTensor (converts PIL to Tensor [0,1] and C,H,W format)
    to_tensor_transform = transforms.ToTensor()
    tensor_image = to_tensor_transform(cropped_image)
    print(f"ToTensor output shape: {tensor_image.shape}, dtype: {tensor_image.dtype}, min: {tensor_image.min()}, max: {tensor_image.max()}")

    # 4. Normalize (mean, std for each channel - MNIST is grayscale, so 1 channel)
    # For MNIST, mean=0.1307, std=0.3081 (approx)
    normalize_transform = transforms.Normalize(mean=(0.1307,), std=(0.3081,))
    normalized_image_tensor = normalize_transform(tensor_image)
    print(f"Normalized tensor mean: {normalized_image_tensor.mean():.4f}, std: {normalized_image_tensor.std():.4f} (approx)")

    # 5. Compose: Chaining transformations
    composed_transform = transforms.Compose([
        transforms.Resize(32),       # Resize to 32x32
        transforms.ToTensor(),       # Convert to tensor
        transforms.Normalize((0.1307,), (0.3081,)) # Normalize
    ])
    final_tensor = composed_transform(pil_image)
    print(f"Composed transform output shape: {final_tensor.shape}")

    # --- Data Augmentation Techniques ---
    print("\n--- Data Augmentation ---")
    # Typically applied only to the training set
    augmentation_pipeline = transforms.Compose([
        transforms.RandomResizedCrop(size=28, scale=(0.8, 1.0)), # Crop randomly and resize
        transforms.RandomHorizontalFlip(p=0.5), # Flip horizontally with 50% prob
        transforms.RandomRotation(degrees=15),    # Rotate by +/- 15 degrees
        transforms.ColorJitter(brightness=0.2, contrast=0.2), # Adjust brightness/contrast
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    # Show a few augmented versions of the same image
    plt.figure(figsize=(12, 3))
    plt.subplot(1, 5, 1)
    plt.imshow(pil_image, cmap='gray'); plt.title("Original"); plt.axis('off')
    for i in range(4):
        augmented_img_tensor = augmentation_pipeline(pil_image)
        plt.subplot(1, 5, i + 2)
        # To display, we need to un-normalize and convert back if needed, or just show tensor
        # For simplicity, showing permuted tensor (matplotlib expects H,W,C or H,W)
        plt.imshow(augmented_img_tensor.squeeze().numpy(), cmap='gray') 
        plt.title(f"Augmented {i+1}"); plt.axis('off')
    aug_plot_path = os.path.join(output_dir, "augmentation_examples.png")
    plt.savefig(aug_plot_path); plt.close()
    print(f"Augmentation examples plot saved to '{aug_plot_path}'")
    print("Augmentations increase dataset diversity and improve model robustness.")

# ====================================
# 7. Working with Different Data Types (Conceptual - in README)
# ====================================
def discuss_different_data_types():
    print("\nSection 7: Working with Different Data Types")
    print("-" * 70)
    print("This section is conceptual and detailed in the README.md.")
    print("Covers: Specifics for Image, Text, and Tabular data.")
    print("  - Image: PIL/OpenCV, torchvision.transforms, channel order, normalization.")
    print("  - Text: Tokenization, numericalization, padding, nn.Embedding.")
    print("  - Tabular: Pandas, feature engineering, scaling, encoding categoricals.")

# ====================================
# 8. Efficient Data Loading Techniques (Covered conceptually in DataLoader demo)
# ====================================
def recap_efficient_loading():
    print("\nSection 8: Efficient Data Loading Techniques")
    print("-" * 70)
    print("Techniques for speeding up data loading include:")
    print("  - `num_workers > 0` in DataLoader for parallel processing.")
    print("  - `pin_memory=True` in DataLoader for faster CPU-to-GPU transfers.")
    print("  - Pre-fetching, caching (advanced). Optimized file formats (e.g., HDF5, TFRecord concept). ")
    print("  (num_workers and pin_memory demonstrated in Section 4 with DataLoader).")

# ====================================
# 9. Practical Example: Image Classification Dataset
# (Covered by CustomImageFolderDataset and its usage with DataLoader/Transforms)
# ====================================
def recap_practical_image_example():
    print("\nSection 9: Practical Example - Image Classification Dataset")
    print("-" * 70)
    print("A practical image classification dataset pipeline involves:")
    print("  1. Organizing images into class folders (e.g., 'root/class_A/img1.jpg').")
    print("  2. Creating a Custom `Dataset` (like `CustomImageFolderDataset` shown in Sec 3).")
    print("     Or using `torchvision.datasets.ImageFolder` for this structure.")
    print("  3. Defining separate `transforms.Compose` pipelines for training (with augmentation) and validation/testing (without augmentation but with normalization).")
    print("  4. Wrapping these datasets in `DataLoader` instances for batching and shuffling.")
    # Example using ImageFolder for simplicity here
    dummy_image_root = os.path.join(output_dir, "dummy_image_dataset_imagefolder")
    create_dummy_image_folder_dataset(dummy_image_root, num_classes=2, imgs_per_class=2)
    
    train_transform_example = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    val_transform_example = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    # Check if dummy_image_root has subdirectories (classes)
    if any(os.path.isdir(os.path.join(dummy_image_root, i)) for i in os.listdir(dummy_image_root)):
        try:
            example_imagefolder_dataset = torchvision.datasets.ImageFolder(root=dummy_image_root, transform=train_transform_example)
            if len(example_imagefolder_dataset) > 0:
                example_loader = DataLoader(example_imagefolder_dataset, batch_size=2, shuffle=True)
                print(f"ImageFolder: Found {len(example_imagefolder_dataset)} images in {len(example_imagefolder_dataset.classes)} classes.")
                ex_imgs, ex_lbls = next(iter(example_loader))
                print(f"Sample batch from ImageFolder DataLoader: images shape {ex_imgs.shape}, labels {ex_lbls}")
            else:
                print("ImageFolder dataset initialized but is empty.")
        except Exception as e:
            print(f"Could not initialize ImageFolder (ensure subdirectories exist and are not empty): {e}")
    else:
        print(f"Skipping ImageFolder example: No class subdirectories found in {dummy_image_root}")

# ====================================
# Main function to run all sections
# ====================================
def main():
    """Main function to run all Data Loading and Preprocessing tutorial sections."""
    print("=" * 80)
    print("PyTorch Data Loading, Preprocessing, and Augmentation Tutorial")
    print("=" * 80)
    
    intro_data_handling() # Section 1
    demonstrate_builtin_datasets() # Section 2
    demonstrate_custom_datasets() # Section 3
    demonstrate_dataloader() # Section 4
    demonstrate_transforms_augmentation() # Section 5 & 6
    discuss_different_data_types() # Section 7 (Conceptual)
    recap_efficient_loading() # Section 8 (Recap)
    recap_practical_image_example() # Section 9 (Recap/Example)
    
    print("\nTutorial complete! Outputs (like plots) are in '05_data_loading_preprocessing_outputs' directory.")

if __name__ == '__main__':
    main() 