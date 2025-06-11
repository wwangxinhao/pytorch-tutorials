#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
PyTorch Transformers and Attention Mechanisms Tutorial

This script provides a detailed introduction to Transformers and Attention Mechanisms
in PyTorch, covering their components, implementation, and applications.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
import time

# Set random seed for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# -----------------------------------------------------------------------------
# Section 1: Introduction to Attention Mechanisms
# -----------------------------------------------------------------------------

def intro_to_attention():
    print("\nSection 1: Introduction to Attention Mechanisms")
    print("-" * 70)
    print("Attention allows models to focus on relevant parts of the input.")
    print("Example: In machine translation, which input words are most relevant to predict the next output word.")
    # Basic concept: Query, Key, Value
    # Score(Query, Key_i) -> Attention_Weight_i -> Weighted_Sum(Attention_Weight_i * Value_i)
    print("Core idea: Compute attention scores, normalize to weights, then take weighted sum of values.")

# -----------------------------------------------------------------------------
# Section 2: Scaled Dot-Product Attention
# -----------------------------------------------------------------------------

class ScaledDotProductAttention(nn.Module):
    """ Scaled Dot-Product Attention """
    def __init__(self, temperature, attn_dropout=0.1):
        super().__init__()
        self.temperature = temperature
        self.dropout = nn.Dropout(attn_dropout)

    def forward(self, q, k, v, mask=None):
        # q, k, v: [batch_size, n_heads, seq_len, d_k or d_v]
        # temperature is sqrt(d_k)
        attn = torch.matmul(q / self.temperature, k.transpose(2, 3))

        if mask is not None:
            attn = attn.masked_fill(mask == 0, -1e9) # or float('-inf')

        attn = self.dropout(F.softmax(attn, dim=-1))
        output = torch.matmul(attn, v)
        return output, attn

def demonstrate_scaled_dot_product_attention():
    print("\nSection 2: Scaled Dot-Product Attention")
    print("-" * 70)
    batch_size, seq_len, d_k, d_v = 2, 5, 16, 20 # Example dimensions
    q = torch.randn(batch_size, seq_len, d_k).to(device)
    k = torch.randn(batch_size, seq_len, d_k).to(device)
    v = torch.randn(batch_size, seq_len, d_v).to(device)
    
    # Reshape for single head attention for this basic demo
    q_single_head = q.unsqueeze(1) # [batch, 1, seq_len, d_k]
    k_single_head = k.unsqueeze(1) # [batch, 1, seq_len, d_k]
    v_single_head = v.unsqueeze(1) # [batch, 1, seq_len, d_v]

    attention = ScaledDotProductAttention(temperature=np.power(d_k, 0.5)).to(device)
    output, attn_weights = attention(q_single_head, k_single_head, v_single_head)
    
    print("Query shape:", q_single_head.shape)
    print("Key shape:", k_single_head.shape)
    print("Value shape:", v_single_head.shape)
    print("Output shape:", output.shape)
    print("Attention weights shape:", attn_weights.shape)
    print("Sample output (first item, first head):")
    print(output[0, 0])
    print("Sample attention weights (first item, first head):")
    print(attn_weights[0, 0])

# -----------------------------------------------------------------------------
# Section 3: Multi-Head Attention
# -----------------------------------------------------------------------------

class MultiHeadAttention(nn.Module):
    """ Multi-Head Attention module """
    def __init__(self, n_head, d_model, d_k, d_v, dropout=0.1):
        super().__init__()
        self.n_head = n_head
        self.d_k = d_k
        self.d_v = d_v

        self.w_qs = nn.Linear(d_model, n_head * d_k, bias=False)
        self.w_ks = nn.Linear(d_model, n_head * d_k, bias=False)
        self.w_vs = nn.Linear(d_model, n_head * d_v, bias=False)
        self.fc = nn.Linear(n_head * d_v, d_model, bias=False)

        self.attention = ScaledDotProductAttention(temperature=d_k ** 0.5)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)

    def forward(self, q, k, v, mask=None):
        # q, k, v: [batch_size, seq_len, d_model]
        d_k, d_v, n_head = self.d_k, self.d_v, self.n_head
        sz_b, len_q, len_k, len_v = q.size(0), q.size(1), k.size(1), v.size(1)

        residual = q

        # Pass through the pre-attention projection: b x lq x (n*dv)
        # Separate different heads: b x lq x n x dv
        q = self.w_qs(q).view(sz_b, len_q, n_head, d_k)
        k = self.w_ks(k).view(sz_b, len_k, n_head, d_k)
        v = self.w_vs(v).view(sz_b, len_v, n_head, d_v)

        # Transpose for attention dot product: b x n x lq x dv
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)

        if mask is not None:
            mask = mask.unsqueeze(1)   # For head axis broadcasting.

        output, attn = self.attention(q, k, v, mask=mask)

        # Transpose to move the head dimension back: b x lq x n x dv
        # Combine the last two dimensions to concatenate all the heads together: b x lq x (n*dv)
        output = output.transpose(1, 2).contiguous().view(sz_b, len_q, -1)
        output = self.dropout(self.fc(output))
        output += residual

        output = self.layer_norm(output)
        return output, attn

def demonstrate_multi_head_attention():
    print("\nSection 3: Multi-Head Attention")
    print("-" * 70)
    batch_size, seq_len, d_model, n_head = 2, 5, 512, 8
    d_k = d_v = d_model // n_head # Ensure d_k * n_head = d_model

    q = torch.randn(batch_size, seq_len, d_model).to(device) # query, key, value are same for self-attention
    
    mha = MultiHeadAttention(n_head, d_model, d_k, d_v).to(device)
    output, attn_weights = mha(q, q, q) # self-attention case

    print("Input q shape:", q.shape)
    print("Output shape:", output.shape)
    print("Attention weights shape:", attn_weights.shape) # [batch_size, n_head, seq_len, seq_len]
    print("Sample output (first item):")
    print(output[0])
    print("Sample attention weights (first item, first head):")
    print(attn_weights[0,0])

# -----------------------------------------------------------------------------
# Section 4: Positional Encoding
# -----------------------------------------------------------------------------

class PositionalEncoding(nn.Module):
    def __init__(self, d_hid, n_position=200):
        super(PositionalEncoding, self).__init__()
        # Not a parameter
        self.register_buffer('pos_table', self._get_sinusoid_encoding_table(n_position, d_hid))

    def _get_sinusoid_encoding_table(self, n_position, d_hid):
        ''' Sinusoid position encoding table '''
        def get_position_angle_vec(position):
            return [position / np.power(10000, 2 * (hid_j // 2) / d_hid) for hid_j in range(d_hid)]

        sinusoid_table = np.array([get_position_angle_vec(pos_i) for pos_i in range(n_position)])
        sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # dim 2i
        sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # dim 2i+1
        return torch.FloatTensor(sinusoid_table).unsqueeze(0)

    def forward(self, x):
        # x: [batch_size, seq_len, d_model]
        return x + self.pos_table[:, :x.size(1)].clone().detach()

def demonstrate_positional_encoding():
    print("\nSection 4: Positional Encoding")
    print("-" * 70)
    batch_size, seq_len, d_model = 2, 10, 512
    x = torch.randn(batch_size, seq_len, d_model).to(device)
    
    pos_enc = PositionalEncoding(d_model, n_position=seq_len).to(device)
    x_with_pe = pos_enc(x)
    
    print("Input shape:", x.shape)
    print("Shape after adding positional encoding:", x_with_pe.shape)
    print("Difference between input and output (should be the PE values):")
    print((x_with_pe - x)[0, :, :5]) # Print first 5 dimensions for brevity

# -----------------------------------------------------------------------------
# Section 5: Position-wise Feed-Forward Network
# -----------------------------------------------------------------------------

class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_in, d_hid, dropout=0.1):
        super().__init__()
        self.w_1 = nn.Linear(d_in, d_hid)
        self.w_2 = nn.Linear(d_hid, d_in)
        self.layer_norm = nn.LayerNorm(d_in, eps=1e-6)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        residual = x
        x = self.w_2(F.relu(self.w_1(x)))
        x = self.dropout(x)
        x += residual
        x = self.layer_norm(x)
        return x

def demonstrate_ffn():
    print("\nSection 5: Position-wise Feed-Forward Network (FFN)")
    print("-" * 70)
    batch_size, seq_len, d_model, d_hid = 2, 5, 512, 2048
    x = torch.randn(batch_size, seq_len, d_model).to(device)

    ffn = PositionwiseFeedForward(d_model, d_hid).to(device)
    output = ffn(x)

    print("Input shape:", x.shape)
    print("Output shape:", output.shape)
    print("Sample output (first item):")
    print(output[0])

# -----------------------------------------------------------------------------
# Section 6: Encoder and Decoder Layers
# -----------------------------------------------------------------------------

class EncoderLayer(nn.Module):
    def __init__(self, d_model, d_inner, n_head, d_k, d_v, dropout=0.1):
        super(EncoderLayer, self).__init__()
        self.slf_attn = MultiHeadAttention(n_head, d_model, d_k, d_v, dropout=dropout)
        self.pos_ffn = PositionwiseFeedForward(d_model, d_inner, dropout=dropout)

    def forward(self, enc_input, slf_attn_mask=None):
        enc_output, enc_slf_attn = self.slf_attn(enc_input, enc_input, enc_input, mask=slf_attn_mask)
        enc_output = self.pos_ffn(enc_output)
        return enc_output, enc_slf_attn

class DecoderLayer(nn.Module):
    def __init__(self, d_model, d_inner, n_head, d_k, d_v, dropout=0.1):
        super(DecoderLayer, self).__init__()
        self.slf_attn = MultiHeadAttention(n_head, d_model, d_k, d_v, dropout=dropout)
        self.enc_attn = MultiHeadAttention(n_head, d_model, d_k, d_v, dropout=dropout)
        self.pos_ffn = PositionwiseFeedForward(d_model, d_inner, dropout=dropout)

    def forward(self, dec_input, enc_output, slf_attn_mask=None, dec_enc_attn_mask=None):
        dec_output, dec_slf_attn = self.slf_attn(dec_input, dec_input, dec_input, mask=slf_attn_mask)
        dec_output, dec_enc_attn = self.enc_attn(dec_output, enc_output, enc_output, mask=dec_enc_attn_mask)
        dec_output = self.pos_ffn(dec_output)
        return dec_output, dec_slf_attn, dec_enc_attn

def demonstrate_encoder_decoder_layers():
    print("\nSection 6: Encoder and Decoder Layers")
    print("-" * 70)
    batch_size, seq_len, d_model, d_inner, n_head = 2, 5, 512, 2048, 8
    d_k = d_v = d_model // n_head

    # Encoder Layer
    enc_input = torch.randn(batch_size, seq_len, d_model).to(device)
    encoder_layer = EncoderLayer(d_model, d_inner, n_head, d_k, d_v).to(device)
    enc_output, _ = encoder_layer(enc_input)
    print("Encoder Input Shape:", enc_input.shape)
    print("Encoder Output Shape:", enc_output.shape)

    # Decoder Layer
    dec_input = torch.randn(batch_size, seq_len, d_model).to(device)
    # enc_output from above is used here
    decoder_layer = DecoderLayer(d_model, d_inner, n_head, d_k, d_v).to(device)
    dec_output, _, _ = decoder_layer(dec_input, enc_output)
    print("\nDecoder Input Shape:", dec_input.shape)
    print("Decoder using Encoder Output Shape:", enc_output.shape)
    print("Decoder Output Shape:", dec_output.shape)

# -----------------------------------------------------------------------------
# Section 7: Building a Simple Transformer (Conceptual)
# -----------------------------------------------------------------------------

class Transformer(nn.Module):
    def __init__(self, n_src_vocab, n_tgt_vocab, src_pad_idx, tgt_pad_idx,
                 d_word_vec=512, d_model=512, d_inner=2048,
                 n_layers=6, n_head=8, d_k=64, d_v=64, dropout=0.1, n_position=200):
        super().__init__()

        self.src_word_emb = nn.Embedding(n_src_vocab, d_word_vec, padding_idx=src_pad_idx)
        self.tgt_word_emb = nn.Embedding(n_tgt_vocab, d_word_vec, padding_idx=tgt_pad_idx)
        self.position_enc = PositionalEncoding(d_model, n_position=n_position)
        self.dropout = nn.Dropout(p=dropout)

        self.encoder = nn.ModuleList([
            EncoderLayer(d_model, d_inner, n_head, d_k, d_v, dropout=dropout)
            for _ in range(n_layers)])

        self.decoder = nn.ModuleList([
            DecoderLayer(d_model, d_inner, n_head, d_k, d_v, dropout=dropout)
            for _ in range(n_layers)])

        self.tgt_word_prj = nn.Linear(d_model, n_tgt_vocab, bias=False)
        # Weight tying for embedding and projection
        self.tgt_word_prj.weight = self.tgt_word_emb.weight 
        # Xavier initialization for projection layer
        nn.init.xavier_normal_(self.tgt_word_prj.weight) 

        self.d_model = d_model

    def get_pad_mask(self, seq, pad_idx):
        # seq: [batch_size, seq_len]
        return (seq != pad_idx).unsqueeze(-2) # [batch_size, 1, seq_len]

    def get_subsequent_mask(self, seq):
        # seq: [batch_size, seq_len]
        sz_b, len_s = seq.size()
        subsequent_mask = (1 - torch.triu(
            torch.ones((1, len_s, len_s), device=seq.device), diagonal=1)).bool()
        return subsequent_mask # [1, seq_len, seq_len]

    def forward(self, src_seq, tgt_seq):
        # src_seq: [batch_size, src_len]
        # tgt_seq: [batch_size, tgt_len]

        src_mask = self.get_pad_mask(src_seq, self.src_word_emb.padding_idx)
        tgt_mask = self.get_pad_mask(tgt_seq, self.tgt_word_emb.padding_idx) & \
                   self.get_subsequent_mask(tgt_seq)

        enc_output = self.dropout(self.position_enc(self.src_word_emb(src_seq)))
        for enc_layer in self.encoder:
            enc_output, _ = enc_layer(enc_output, slf_attn_mask=src_mask)

        dec_output = self.dropout(self.position_enc(self.tgt_word_emb(tgt_seq)))
        for dec_layer in self.decoder:
            dec_output, _, _ = dec_layer(dec_output, enc_output, slf_attn_mask=tgt_mask, dec_enc_attn_mask=src_mask)
        
        seq_logit = self.tgt_word_prj(dec_output)
        return seq_logit.view(-1, seq_logit.size(2))

def demonstrate_transformer_model():
    print("\nSection 7: Building a Simple Transformer Model")
    print("-" * 70)
    src_vocab_size = 1000
    tgt_vocab_size = 1200
    src_pad_idx = 0
    tgt_pad_idx = 0
    batch_size = 2
    src_seq_len = 10
    tgt_seq_len = 12

    model = Transformer(n_src_vocab=src_vocab_size, n_tgt_vocab=tgt_vocab_size,
                        src_pad_idx=src_pad_idx, tgt_pad_idx=tgt_pad_idx).to(device)

    src_seq = torch.randint(1, src_vocab_size, (batch_size, src_seq_len)).to(device) # Avoid padding idx for simplicity
    tgt_seq = torch.randint(1, tgt_vocab_size, (batch_size, tgt_seq_len)).to(device)
    src_seq[0, -2:] = src_pad_idx # Add some padding
    tgt_seq[0, -3:] = tgt_pad_idx

    output_logits = model(src_seq, tgt_seq)
    print("Source sequence shape:", src_seq.shape)
    print("Target sequence shape:", tgt_seq.shape)
    print("Output logits shape (batch_size * tgt_seq_len, tgt_vocab_size):", output_logits.shape)
    print("Sample output logits (first 5 predictions for first item in batch):")
    print(output_logits.view(batch_size, tgt_seq_len, -1)[0, :5, :10]) # First 5 tokens, first 10 vocab scores

# -----------------------------------------------------------------------------
# Section 8: Using Hugging Face Transformers (Conceptual Example)
# -----------------------------------------------------------------------------

def demonstrate_huggingface_transformers():
    print("\nSection 8: Using Hugging Face Transformers (Conceptual)")
    print("-" * 70)
    print("The Hugging Face `transformers` library simplifies using pre-trained models.")
    print("Example: Using BERT for feature extraction.")
    try:
        from transformers import BertTokenizer, BertModel
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        model = BertModel.from_pretrained('bert-base-uncased').to(device)

        text = "Hello, this is a test sentence for BERT."
        inputs = tokenizer(text, return_tensors="pt").to(device)
        print("Input IDs:", inputs['input_ids'])
        print("Attention Mask:", inputs['attention_mask'])
        
        with torch.no_grad(): # Disable gradient calculations for inference
            outputs = model(**inputs)
        
        last_hidden_states = outputs.last_hidden_state
        pooler_output = outputs.pooler_output # Output of [CLS] token after linear layer + Tanh

        print("Last hidden states shape (batch, seq_len, hidden_size):", last_hidden_states.shape)
        print("Pooler output shape (batch, hidden_size):", pooler_output.shape)
        print("This demonstrates loading a pre-trained model and getting embeddings.")

    except ImportError:
        print("Hugging Face `transformers` library not installed. Skipping this demo.")
        print("Install with: pip install transformers")
    except Exception as e:
        print(f"An error occurred while demonstrating Hugging Face: {e}")
        print("This might be due to network issues or model availability.")

# -----------------------------------------------------------------------------
# Main function to run all sections
# -----------------------------------------------------------------------------

def main():
    """Main function to run all tutorial sections."""
    print("=" * 80)
    print("PyTorch Transformers and Attention Mechanisms Tutorial")
    print("=" * 80)

    intro_to_attention()
    demonstrate_scaled_dot_product_attention()
    demonstrate_multi_head_attention()
    demonstrate_positional_encoding()
    demonstrate_ffn()
    demonstrate_encoder_decoder_layers()
    demonstrate_transformer_model() # This will build a full Transformer model
    demonstrate_huggingface_transformers()

    print("\nTutorial complete!")

if __name__ == '__main__':
    main() 