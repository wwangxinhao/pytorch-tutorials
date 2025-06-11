# Recurrent Neural Networks (RNNs) in PyTorch: A Comprehensive Guide

This tutorial provides an in-depth guide to understanding, implementing, and applying Recurrent Neural Networks (RNNs), Long Short-Term Memory (LSTM) networks, and Gated Recurrent Units (GRUs) using PyTorch. These models are fundamental for processing sequential data such as text, time series, and audio.

## Table of Contents
1. [Introduction to Recurrent Neural Networks](#introduction-to-recurrent-neural-networks)
   - What are RNNs and Why Sequential Data?
   - The Concept of a Hidden State (Memory)
   - Basic RNN Cell Structure and Unrolling
   - Challenges: Vanishing and Exploding Gradients
2. [Core RNN Layer Implementations in PyTorch](#core-rnn-layer-implementations-in-pytorch)
   - **`nn.RNN`**: The basic Elman RNN.
     - Key Parameters: `input_size`, `hidden_size`, `num_layers`, `batch_first`, `bidirectional`.
     - Input and Output Shapes.
   - **`nn.LSTM` (Long Short-Term Memory)**
     - Addressing Vanishing Gradients with Gates (Forget, Input, Output Gates, Cell State).
     - Key Parameters and Shapes.
   - **`nn.GRU` (Gated Recurrent Unit)**
     - Simplified Gating Mechanism (Update, Reset Gates).
     - Key Parameters and Shapes.
   - Multi-layer (Stacked) RNNs
   - Bidirectional RNNs
3. [Sequence Modeling with RNNs](#sequence-modeling-with-rnns)
   - Many-to-One, One-to-Many, Many-to-Many Architectures (Conceptual)
   - **Sequence Classification (Many-to-One):** e.g., Sentiment Analysis.
     - Using the final hidden state or pooling outputs for classification.
   - Handling Variable-Length Sequences: Padding, Packing (`torch.nn.utils.rnn.pack_padded_sequence`, `pad_packed_sequence`).
4. [Application: Text Generation (Character-level RNN)](#application-text-generation-character-level-rnn)
   - Representing Text Data (Character Encoding).
   - Preparing Input-Target Sequences for Language Modeling.
   - Building a Character-level RNN/LSTM Model.
   - Training the Language Model.
   - Generating New Text (Sampling Strategies, Temperature).
5. [Application: Time Series Forecasting](#application-time-series-forecasting)
   - Preparing Time Series Data (Windowing/Sliding Windows).
   - Univariate vs. Multivariate Time Series.
   - Building an RNN/LSTM Model for Forecasting.
   - Sequence-to-Sequence vs. Sequence-to-Value Forecasting.
6. [Advanced RNN Techniques (Conceptual Overview)](#advanced-rnn-techniques-conceptual-overview)
   - **Attention Mechanisms:** Allowing the model to focus on relevant parts of the input sequence.
   - **Teacher Forcing:** Using ground truth outputs as inputs during training for faster convergence.
   - **Beam Search:** A more advanced decoding strategy for generation tasks.
   - **Encoder-Decoder Architecture (Seq2Seq):** For tasks like machine translation.
7. [Practical Tips for Training RNNs](#practical-tips-for-training-rnns)
   - Gradient Clipping to prevent exploding gradients.
   - Proper Initialization.
   - Choosing between RNN, LSTM, GRU.
   - Regularization (Dropout on non-recurrent connections).

## Introduction to Recurrent Neural Networks

- **What are RNNs and Why Sequential Data?**
  RNNs are a class of neural networks designed to recognize patterns in sequences of data, such as text, speech, time series, or genomes. Unlike feedforward networks, RNNs have loops, allowing information to persist from one step of the sequence to the next.
- **The Concept of a Hidden State (Memory):**
  The core idea of an RNN is its hidden state, which acts as a form of memory. The hidden state at timestep `t` captures information from all previous timesteps up to `t-1`. This hidden state is updated at each step based on the current input and the previous hidden state.
  `h_t = f(W_hh * h_{t-1} + W_xh * x_t + b_h)`
  `output_t = g(W_hy * h_t + b_y)`
- **Basic RNN Cell Structure and Unrolling:** An RNN can be thought of as multiple copies of the same network, each passing a message to a successor. Unrolling the RNN visualizes this chain-like structure.
- **Challenges: Vanishing and Exploding Gradients:** Standard RNNs struggle to learn long-range dependencies due to the vanishing gradient problem (gradients shrink exponentially as they propagate back through time) or the exploding gradient problem (gradients grow exponentially).

## Core RNN Layer Implementations in PyTorch

PyTorch provides optimized implementations for common recurrent layers.

- **`nn.RNN`**: The basic Elman RNN.
  - **Key Parameters:**
    - `input_size`: The number of expected features in the input `x`.
    - `hidden_size`: The number of features in the hidden state `h`.
    - `num_layers`: Number of recurrent layers. Stacking RNNs can increase model capacity.
    - `nonlinearity`: `tanh` or `relu`. Default: `tanh`.
    - `batch_first (bool)`: If `True`, input and output tensors are provided as `(batch, seq, feature)` instead of `(seq, batch, feature)`. Default: `False`.
    - `dropout (float)`: If non-zero, introduces a Dropout layer on the outputs of each RNN layer except the last layer. Default: 0.
    - `bidirectional (bool)`: If `True`, becomes a bidirectional RNN. Default: `False`.
  - **Input Shapes (if `batch_first=False`):**
    - `input`: `(seq_len, batch_size, input_size)`
    - `h_0` (initial hidden state): `(num_layers * num_directions, batch_size, hidden_size)`
  - **Output Shapes (if `batch_first=False`):**
    - `output`: `(seq_len, batch_size, num_directions * hidden_size)` (all hidden states from the last layer)
    - `h_n` (final hidden state): `(num_layers * num_directions, batch_size, hidden_size)`
  ```python
  import torch
  import torch.nn as nn

  # Example nn.RNN
  rnn = nn.RNN(input_size=10, hidden_size=20, num_layers=2, batch_first=True)
  # input_tensor shape: (batch_size=5, seq_len=3, input_size=10)
  # input_tensor = torch.randn(5, 3, 10)
  # h0 shape: (num_layers*num_directions=2*1, batch_size=5, hidden_size=20)
  # h0 = torch.randn(2, 5, 20)
  # output, hn = rnn(input_tensor, h0)
  # print(f"RNN Output shape: {output.shape}") # (5, 3, 20)
  # print(f"RNN Hidden state shape: {hn.shape}") # (2, 5, 20)
  ```

- **`nn.LSTM` (Long Short-Term Memory)**
  LSTMs use a more complex cell structure with gates (input, forget, output) and a cell state (`c_t`) to better control information flow and capture long-range dependencies, mitigating vanishing gradients.
  - **Gates:** Sigmoid layers that control what information to keep or discard.
  - **Cell State (`c_t`):** A separate memory stream that information can be added to or removed from, regulated by gates.
  - **Input/Output Shapes:** Similar to `nn.RNN`, but `h_0` and `h_n` are tuples `(hidden_state, cell_state)`. Each state has shape `(num_layers * num_directions, batch, hidden_size)`. 
  ```python
  # lstm = nn.LSTM(input_size=10, hidden_size=20, num_layers=2, batch_first=True)
  # input_lstm = torch.randn(5, 3, 10)
  # h0_lstm = torch.randn(2, 5, 20) # Initial hidden state
  # c0_lstm = torch.randn(2, 5, 20) # Initial cell state
  # output_lstm, (hn_lstm, cn_lstm) = lstm(input_lstm, (h0_lstm, c0_lstm))
  # print(f"LSTM Output shape: {output_lstm.shape}")
  # print(f"LSTM Hidden state shape: {hn_lstm.shape}")
  # print(f"LSTM Cell state shape: {cn_lstm.shape}")
  ```

- **`nn.GRU` (Gated Recurrent Unit)**
  GRUs are a simpler alternative to LSTMs, combining the cell state and hidden state. They use update and reset gates.
  - **Input/Output Shapes:** Same as `nn.RNN`. 
  ```python
  # gru = nn.GRU(input_size=10, hidden_size=20, num_layers=2, batch_first=True)
  # input_gru = torch.randn(5, 3, 10)
  # h0_gru = torch.randn(2, 5, 20)
  # output_gru, hn_gru = gru(input_gru, h0_gru)
  # print(f"GRU Output shape: {output_gru.shape}")
  # print(f"GRU Hidden state shape: {hn_gru.shape}")
  ```

- **Multi-layer (Stacked) RNNs:** Set `num_layers > 1`. The output of one layer becomes the input to the next. Dropout can be applied between layers.
- **Bidirectional RNNs:** Set `bidirectional=True`. Processes the sequence in both forward and backward directions. The outputs are typically concatenated. Useful when context from both past and future is important.

## Sequence Modeling with RNNs

- **Architectures:** RNNs can be used for various sequence tasks:
  - **Many-to-One:** Input sequence, single output (e.g., sentiment classification of a sentence).
  - **One-to-Many:** Single input, output sequence (e.g., image captioning).
  - **Many-to-Many (Synchronized):** Input and output sequences have same length (e.g., part-of-speech tagging).
  - **Many-to-Many (Delayed/Encoder-Decoder):** Input and output sequences can have different lengths (e.g., machine translation).
- **Handling Variable-Length Sequences:** Real-world sequences often have different lengths. Techniques:
  - **Padding:** Pad shorter sequences to the length of the longest sequence in a batch using a special padding token.
  - **Packing (`torch.nn.utils.rnn.pack_padded_sequence`):** Before feeding padded sequences to an RNN, pack them to avoid computation on padding tokens. Use `torch.nn.utils.rnn.pad_packed_sequence` to unpack the output.

## Application: Text Generation (Character-level RNN)

- **Representing Text Data:** Convert characters to numerical indices (character encoding). Create a vocabulary of all unique characters.
- **Preparing Sequences:** For a sequence `s`, the input at timestep `t` is `s[t]` and the target is `s[t+1]`. The model learns to predict the next character.
- **Training:** Use Cross-Entropy Loss to compare predicted character probabilities with the actual next character.
- **Generating New Text:** Start with a seed character/sequence. Feed it to the model to get probabilities for the next character. Sample from this distribution (e.g., using `torch.multinomial` or `argmax`). Append the sampled character to the sequence and repeat.
  - **Temperature:** A hyperparameter to control the randomness of sampling. Higher temperature -> more random; lower temperature -> more deterministic.

## Application: Time Series Forecasting

- **Preparing Data (Windowing):** Create input-output pairs by sliding a window over the time series. Input: `(x_t, x_{t+1}, ..., x_{t+N-1})`. Target: `x_{t+N}` (for one-step ahead) or `(x_{t+N}, ..., x_{t+N+M-1})` (for multi-step ahead).
- **Univariate vs. Multivariate:** Forecasting a single variable vs. multiple interacting variables.
- **Model Output:** Can be a single value (next step) or a sequence (multiple future steps).

## Advanced RNN Techniques (Conceptual Overview)

- **Attention Mechanisms:** For long sequences, allows the model to selectively focus on important parts of the input sequence when producing an output at each timestep. Particularly useful in Seq2Seq models.
- **Teacher Forcing:** During training, instead of feeding the model's own (potentially incorrect) previous prediction as input for the next step, the ground truth from the previous step is used. Helps stabilize training but can lead to exposure bias (discrepancy between training and inference).
- **Beam Search:** A decoding algorithm used in generation tasks (like machine translation or text generation) that explores multiple hypotheses (beams) at each step, rather than just greedily picking the single best option.
- **Encoder-Decoder Architecture (Seq2Seq):** Consists of two RNNs: an encoder that processes the input sequence into a context vector, and a decoder that generates the output sequence from this context vector. Widely used in machine translation and text summarization.

## Practical Tips for Training RNNs

- **Gradient Clipping:** Crucial for RNNs/LSTMs/GRUs to prevent exploding gradients. Use `torch.nn.utils.clip_grad_norm_`.
- **Initialization:** Proper weight initialization (e.g., Xavier, Kaiming, or specific heuristics for RNNs) can be important.
- **Choice of Unit:** LSTMs and GRUs are generally preferred over vanilla RNNs for their ability to handle longer sequences. GRUs are simpler and sometimes faster than LSTMs with comparable performance.
- **Dropout:** Apply dropout between stacked RNN layers (using the `dropout` parameter in `nn.RNN/LSTM/GRU`) or on the non-recurrent connections (e.g., before/after the RNN block or between the RNN output and fully connected layers).

## Running the Tutorial

To run the Python script associated with this tutorial:
```bash
python recurrent_neural_networks.py
```
This will execute demonstrations of RNN, LSTM, GRU layers, a character-level text generation example, and a time series forecasting example.

## Prerequisites
- Python 3.7+
- PyTorch 1.10+
- NumPy
- Matplotlib (for visualization)

## Related Tutorials
1. [Training Neural Networks](../04_training_neural_networks/README.md)
2. [Transformers and Attention Mechanisms](../08_transformers_and_attention_mechanisms/README.md) (Modern alternative/successor to RNNs for many sequence tasks) 