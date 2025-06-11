# Transformers and Attention Mechanisms

This tutorial delves into Transformers and Attention Mechanisms, pivotal concepts in modern deep learning, especially for natural language processing and beyond.

## Table of Contents
1. [Introduction to Attention Mechanisms](#introduction-to-attention-mechanisms)
   - What is Attention?
   - Types of Attention (Bahdanau, Luong)
2. [Self-Attention](#self-attention)
   - Concept and Motivation
   - Scaled Dot-Product Attention
3. [Multi-Head Attention](#multi-head-attention)
   - Purpose and Architecture
   - Implementation Details
4. [The Transformer Architecture](#the-transformer-architecture)
   - Encoder-Decoder Structure
   - Positional Encoding
   - Feed-Forward Networks
   - Layer Normalization and Residual Connections
5. [Building a Transformer Block](#building-a-transformer-block)
   - Encoder Block
   - Decoder Block
6. [Applications of Transformers](#applications-of-transformers)
   - Natural Language Processing (e.g., Translation, Summarization)
   - Vision Transformers (ViT)
7. [Implementing a Simple Transformer with PyTorch](#implementing-a-simple-transformer-with-pytorch)
   - Step-by-step guide
8. [Pre-trained Transformer Models (BERT, GPT)](#pre-trained-transformer-models)
   - Overview of popular models
   - Using Hugging Face Transformers library
9. [Fine-tuning Pre-trained Transformers](#fine-tuning-pre-trained-transformers)
   - Concepts and techniques
   - Example: Text classification

## Introduction to Attention Mechanisms

Attention mechanisms in deep learning are inspired by human visual attention – the ability to focus on specific parts of an image while perceiving the whole. In the context of neural networks, attention allows a model to dynamically focus on different parts of the input sequence when producing an output.

- **What is Attention?**
  - A mechanism that allows the model to assign different weights (importance scores) to different parts of the input.
  - Helps in handling long sequences and capturing long-range dependencies.
- **Types of Attention:**
  - **Bahdanau Attention (Additive Attention):** Uses a feed-forward network to compute alignment scores.
  - **Luong Attention (Multiplicative Attention):** Uses dot-product based alignment scores.

## Self-Attention

Self-attention, also known as intra-attention, is an attention mechanism relating different positions of a single sequence in order to compute a representation of the sequence. It is a key component of Transformers.

- **Concept and Motivation:**
  - Allows the model to weigh the importance of other words in the *same* sentence when encoding a particular word.
  - Example: "The animal didn't cross the street because **it** was too tired." Self-attention helps determine if "it" refers to "animal" or "street".
- **Scaled Dot-Product Attention:**
  - The core of self-attention.
  - Queries (Q), Keys (K), and Values (V) are computed from the input embeddings.
  - Attention Score = `softmax((Q * K^T) / sqrt(d_k)) * V`
  - `d_k` is the dimension of the key vectors, used for scaling to prevent overly small gradients.

## Multi-Head Attention

Instead of performing a single attention function, Multi-Head Attention runs multiple attention mechanisms in parallel and concatenates their outputs.

- **Purpose and Architecture:**
  - Allows the model to jointly attend to information from different representation subspaces at different positions.
  - Each "head" can learn different aspects of the input.
  - Input Q, K, V are linearly projected `h` times with different, learned linear projections.
  - Attention is applied by each head in parallel.
  - Outputs are concatenated and linearly projected again.

## The Transformer Architecture

The Transformer model, introduced in "Attention Is All You Need," relies entirely on attention mechanisms, dispensing with recurrence and convolutions.

- **Encoder-Decoder Structure:**
  - **Encoder:** Maps an input sequence of symbol representations (x1, ..., xn) to a sequence of continuous representations (z1, ..., zn). Composed of a stack of N identical layers.
  - **Decoder:** Given z, generates an output sequence (y1, ..., ym) one symbol at a time. Also composed of a stack of N identical layers. The decoder incorporates an additional multi-head attention over the output of the encoder stack.
- **Positional Encoding:**
  - Since Transformers contain no recurrence or convolution, positional encodings are added to the input embeddings to give the model information about the relative or absolute position of tokens in the sequence.
  - Sine and cosine functions of different frequencies are typically used.
- **Feed-Forward Networks:**
  - Each layer in the encoder and decoder contains a fully connected feed-forward network, applied to each position separately and identically. This consists of two linear transformations with a ReLU activation in between.
- **Layer Normalization and Residual Connections:**
  - Each sub-layer (self-attention, feed-forward network) in the encoder and decoder has a residual connection around it, followed by layer normalization.

## Building a Transformer Block

- **Encoder Block:**
  - Multi-Head Self-Attention layer
  - Add & Norm (Residual Connection + Layer Normalization)
  - Position-wise Feed-Forward Network
  - Add & Norm
- **Decoder Block:**
  - Masked Multi-Head Self-Attention layer (to prevent attending to future positions)
  - Add & Norm
  - Multi-Head Attention (over encoder output)
  - Add & Norm
  - Position-wise Feed-Forward Network
  - Add & Norm

## Applications of Transformers

- **Natural Language Processing (NLP):**
  - Machine Translation (original application)
  - Text Summarization
  - Question Answering
  - Sentiment Analysis
  - Text Generation
- **Vision Transformers (ViT):**
  - Apply Transformer architecture directly to sequences of image patches for image classification.

## Implementing a Simple Transformer with PyTorch

This section will provide code examples for building the core components of a Transformer, such as Scaled Dot-Product Attention, Multi-Head Attention, Positional Encoding, and a basic Encoder-Decoder structure using PyTorch.

```python
import torch
import torch.nn as nn
import math

# Example: Scaled Dot-Product Attention
class ScaledDotProductAttention(nn.Module):
    def forward(self, query, key, value, mask=None):
        dk = query.size(-1)
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(dk)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        attention = torch.softmax(scores, dim=-1)
        return torch.matmul(attention, value)

# Further components like MultiHeadAttention, PositionalEncoding, EncoderLayer, DecoderLayer will be shown.
```

## Pre-trained Transformer Models (BERT, GPT)

- **BERT (Bidirectional Encoder Representations from Transformers):**
  - Developed by Google.
  - Designed to pre-train deep bidirectional representations by jointly conditioning on both left and right context in all layers.
  - Used for tasks like question answering, language inference.
- **GPT (Generative Pre-trained Transformer):**
  - Developed by OpenAI.
  - Uses a decoder-only transformer architecture.
  - Excels at text generation tasks.
- **Hugging Face Transformers Library:**
  - Provides thousands of pre-trained models for a wide range of tasks in NLP, vision, and audio.
  - Simplifies downloading and using state-of-the-art models.

```python
# Example using Hugging Face Transformers
# from transformers import BertTokenizer, BertModel
# tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
# model = BertModel.from_pretrained('bert-base-uncased')
# inputs = tokenizer("Hello, my dog is cute", return_tensors="pt")
# outputs = model(**inputs)
# last_hidden_states = outputs.last_hidden_state
```

## Fine-tuning Pre-trained Transformers

- **Concepts:** Instead of training a large model from scratch, take a pre-trained model and adapt it to a specific downstream task using a smaller, task-specific dataset.
- **Techniques:**
  - Add a task-specific layer (e.g., a classification head) on top of the pre-trained model.
  - Unfreeze some of the top layers of the pre-trained model and train them with the task-specific layer.
  - Or, unfreeze and train the entire model, but with a much smaller learning rate.

## Running the Tutorial

To run the Python script associated with this tutorial:
```bash
python transformers_and_attention_mechanisms.py
```
Alternatively, you can follow along with the Jupyter notebook `transformers_and_attention_mechanisms.ipynb` for an interactive experience.

## Prerequisites
- Python 3.7+
- PyTorch 1.10+
- (Optionally) Hugging Face Transformers library: `pip install transformers`

## Next Steps
Explore building and training a full Transformer model for a specific task, or dive deeper into the mathematics and variations of attention mechanisms. 