"""
Tutorial 15: Advanced Model Architectures
=========================================

This tutorial explores cutting-edge neural network architectures including
Graph Neural Networks, Vision Transformers, and other state-of-the-art models.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from typing import Optional, Tuple, Union
import matplotlib.pyplot as plt
from torch_geometric.nn import GCNConv, GATConv, global_mean_pool
from torch_geometric.data import Data, Batch
import torchvision
from einops import rearrange, repeat
from einops.layers.torch import Rearrange

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")
print()

# Example 1: Graph Neural Networks (GNNs)
print("Example 1: Graph Neural Networks")
print("=" * 50)

class GraphConvolutionLayer(nn.Module):
    """Simple Graph Convolution Layer"""
    def __init__(self, in_features, out_features):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(in_features, out_features))
        self.bias = nn.Parameter(torch.zeros(out_features))
        
    def forward(self, x, adj):
        # x: [num_nodes, in_features]
        # adj: [num_nodes, num_nodes]
        support = torch.mm(x, self.weight)
        output = torch.sparse.mm(adj, support)
        return output + self.bias

class GCN(nn.Module):
    """Graph Convolutional Network"""
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=2):
        super().__init__()
        self.layers = nn.ModuleList()
        
        # Input layer
        self.layers.append(GraphConvolutionLayer(input_dim, hidden_dim))
        
        # Hidden layers
        for _ in range(num_layers - 2):
            self.layers.append(GraphConvolutionLayer(hidden_dim, hidden_dim))
        
        # Output layer
        self.layers.append(GraphConvolutionLayer(hidden_dim, output_dim))
        
        self.dropout = nn.Dropout(0.5)
        
    def forward(self, x, adj):
        for i, layer in enumerate(self.layers[:-1]):
            x = layer(x, adj)
            x = F.relu(x)
            x = self.dropout(x)
        
        x = self.layers[-1](x, adj)
        return F.log_softmax(x, dim=1)

# Create a simple graph
num_nodes = 100
num_features = 16
num_classes = 4

# Random features
x = torch.randn(num_nodes, num_features)

# Random adjacency matrix (sparse)
edge_prob = 0.1
adj_dense = torch.rand(num_nodes, num_nodes) < edge_prob
adj_dense = adj_dense.float()
adj_dense = (adj_dense + adj_dense.t()) / 2  # Make symmetric
adj = adj_dense.to_sparse()

# Create GCN model
gcn = GCN(num_features, 32, num_classes)
output = gcn(x, adj)
print(f"GCN output shape: {output.shape}")
print()

# Example 2: Graph Attention Networks (GAT)
print("Example 2: Graph Attention Networks")
print("-" * 30)

class GraphAttentionLayer(nn.Module):
    """Graph Attention Layer"""
    def __init__(self, in_features, out_features, dropout=0.6, alpha=0.2):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.dropout = dropout
        self.alpha = alpha
        
        self.W = nn.Parameter(torch.randn(in_features, out_features))
        self.a = nn.Parameter(torch.randn(2 * out_features, 1))
        
        self.leakyrelu = nn.LeakyReLU(self.alpha)
        
    def forward(self, x, adj):
        # x: [N, in_features]
        h = torch.mm(x, self.W)  # [N, out_features]
        N = h.size(0)
        
        # Attention mechanism
        a_input = torch.cat([h.repeat(1, N).view(N * N, -1), 
                           h.repeat(N, 1)], dim=1).view(N, -1, 2 * self.out_features)
        e = self.leakyrelu(torch.matmul(a_input, self.a).squeeze(2))
        
        # Mask attention scores
        zero_vec = -9e15 * torch.ones_like(e)
        attention = torch.where(adj.to_dense() > 0, e, zero_vec)
        attention = F.softmax(attention, dim=1)
        attention = F.dropout(attention, self.dropout, training=self.training)
        
        h_prime = torch.matmul(attention, h)
        return F.elu(h_prime)

class GAT(nn.Module):
    """Graph Attention Network"""
    def __init__(self, input_dim, hidden_dim, output_dim, num_heads=8):
        super().__init__()
        self.attention_heads = nn.ModuleList([
            GraphAttentionLayer(input_dim, hidden_dim) for _ in range(num_heads)
        ])
        self.out_att = GraphAttentionLayer(hidden_dim * num_heads, output_dim)
        
    def forward(self, x, adj):
        # Multi-head attention
        x = torch.cat([att(x, adj) for att in self.attention_heads], dim=1)
        x = F.dropout(x, 0.6, training=self.training)
        x = F.elu(self.out_att(x, adj))
        return F.log_softmax(x, dim=1)

# Test GAT
gat = GAT(num_features, 8, num_classes, num_heads=4)
output = gat(x, adj)
print(f"GAT output shape: {output.shape}")
print()

# Example 3: Vision Transformer (ViT)
print("Example 3: Vision Transformer")
print("=" * 50)

class PatchEmbedding(nn.Module):
    """Convert image into patches and embed them"""
    def __init__(self, img_size=224, patch_size=16, in_channels=3, embed_dim=768):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2
        
        self.projection = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', 
                     p1=patch_size, p2=patch_size),
            nn.Linear(patch_size * patch_size * in_channels, embed_dim)
        )
        
    def forward(self, x):
        return self.projection(x)

class MultiHeadSelfAttention(nn.Module):
    """Multi-Head Self Attention"""
    def __init__(self, embed_dim, num_heads=8, dropout=0.1):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        
        assert embed_dim % num_heads == 0
        
        self.qkv = nn.Linear(embed_dim, embed_dim * 3)
        self.attention_dropout = nn.Dropout(dropout)
        self.proj = nn.Linear(embed_dim, embed_dim)
        self.proj_dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        attn = (q @ k.transpose(-2, -1)) * (self.head_dim ** -0.5)
        attn = attn.softmax(dim=-1)
        attn = self.attention_dropout(attn)
        
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_dropout(x)
        return x

class TransformerBlock(nn.Module):
    """Transformer Block"""
    def __init__(self, embed_dim, num_heads, mlp_ratio=4.0, dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = MultiHeadSelfAttention(embed_dim, num_heads, dropout)
        self.norm2 = nn.LayerNorm(embed_dim)
        
        mlp_hidden_dim = int(embed_dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, mlp_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden_dim, embed_dim),
            nn.Dropout(dropout)
        )
        
    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x

class VisionTransformer(nn.Module):
    """Vision Transformer"""
    def __init__(self, img_size=224, patch_size=16, in_channels=3, 
                 num_classes=1000, embed_dim=768, depth=12, num_heads=12):
        super().__init__()
        self.patch_embed = PatchEmbedding(img_size, patch_size, in_channels, embed_dim)
        num_patches = self.patch_embed.num_patches
        
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        self.pos_drop = nn.Dropout(0.1)
        
        self.blocks = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads) for _ in range(depth)
        ])
        
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)
        
        # Initialize weights
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        
    def forward(self, x):
        B = x.shape[0]
        x = self.patch_embed(x)
        
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embed
        x = self.pos_drop(x)
        
        for block in self.blocks:
            x = block(x)
        
        x = self.norm(x)
        x = x[:, 0]  # Use CLS token
        x = self.head(x)
        return x

# Create a small ViT
vit = VisionTransformer(
    img_size=32, 
    patch_size=4, 
    in_channels=3, 
    num_classes=10, 
    embed_dim=192, 
    depth=6, 
    num_heads=6
).to(device)

# Test with random image
img = torch.randn(4, 3, 32, 32).to(device)
output = vit(img)
print(f"ViT output shape: {output.shape}")
print(f"ViT parameters: {sum(p.numel() for p in vit.parameters()) / 1e6:.2f}M")
print()

# Example 4: EfficientNet-style Architecture
print("Example 4: EfficientNet Architecture")
print("=" * 50)

class MBConvBlock(nn.Module):
    """Mobile Inverted Bottleneck Convolution Block"""
    def __init__(self, in_channels, out_channels, expand_ratio, stride, kernel_size=3):
        super().__init__()
        self.stride = stride
        self.use_residual = stride == 1 and in_channels == out_channels
        
        hidden_channels = in_channels * expand_ratio
        
        layers = []
        # Expansion
        if expand_ratio != 1:
            layers.append(nn.Conv2d(in_channels, hidden_channels, 1, bias=False))
            layers.append(nn.BatchNorm2d(hidden_channels))
            layers.append(nn.SiLU())
        
        # Depthwise conv
        layers.extend([
            nn.Conv2d(hidden_channels, hidden_channels, kernel_size, 
                     stride=stride, padding=kernel_size//2, groups=hidden_channels, bias=False),
            nn.BatchNorm2d(hidden_channels),
            nn.SiLU()
        ])
        
        # Squeeze and excitation
        squeeze_channels = max(1, in_channels // 4)
        layers.extend([
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(hidden_channels, squeeze_channels, 1),
            nn.SiLU(),
            nn.Conv2d(squeeze_channels, hidden_channels, 1),
            nn.Sigmoid()
        ])
        
        # Output projection
        layers.extend([
            nn.Conv2d(hidden_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels)
        ])
        
        self.conv = nn.Sequential(*layers)
        
    def forward(self, x):
        if self.use_residual:
            return x + self.conv(x)
        else:
            return self.conv(x)

class EfficientNet(nn.Module):
    """Simplified EfficientNet"""
    def __init__(self, width_mult=1.0, depth_mult=1.0, num_classes=1000):
        super().__init__()
        
        # Define base architecture
        base_widths = [32, 16, 24, 40, 80, 112, 192, 320]
        base_depths = [1, 2, 2, 3, 3, 4, 1]
        
        # Apply compound scaling
        widths = [int(w * width_mult) for w in base_widths]
        depths = [int(d * depth_mult) for d in base_depths]
        
        # Build network
        self.features = nn.ModuleList()
        
        # Stem
        self.features.append(
            nn.Sequential(
                nn.Conv2d(3, widths[0], 3, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(widths[0]),
                nn.SiLU()
            )
        )
        
        # MBConv blocks
        in_channels = widths[0]
        for i in range(len(depths)):
            out_channels = widths[i + 1]
            for j in range(depths[i]):
                stride = 2 if j == 0 and i > 0 else 1
                self.features.append(
                    MBConvBlock(in_channels, out_channels, expand_ratio=6, stride=stride)
                )
                in_channels = out_channels
        
        # Head
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(widths[-1], num_classes)
        
    def forward(self, x):
        for layer in self.features:
            x = layer(x)
        
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

# Create EfficientNet-B0
efficientnet = EfficientNet(width_mult=1.0, depth_mult=1.0, num_classes=10)
print(f"EfficientNet parameters: {sum(p.numel() for p in efficientnet.parameters()) / 1e6:.2f}M")
print()

# Example 5: Neural Ordinary Differential Equations (Neural ODEs)
print("Example 5: Neural ODEs")
print("-" * 30)

class ODEFunc(nn.Module):
    """ODE function for Neural ODE"""
    def __init__(self, dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim * 2),
            nn.Tanh(),
            nn.Linear(dim * 2, dim)
        )
        
    def forward(self, t, x):
        return self.net(x)

class NeuralODE(nn.Module):
    """Neural ODE Block"""
    def __init__(self, func, solver='euler', step_size=0.1):
        super().__init__()
        self.func = func
        self.solver = solver
        self.step_size = step_size
        
    def forward(self, x, t_span):
        # Simple Euler solver for demonstration
        t0, t1 = t_span
        num_steps = int((t1 - t0) / self.step_size)
        
        h = self.step_size
        for _ in range(num_steps):
            x = x + h * self.func(t0, x)
            t0 += h
            
        return x

# Test Neural ODE
dim = 64
ode_func = ODEFunc(dim)
neural_ode = NeuralODE(ode_func)

x = torch.randn(32, dim)
t_span = (0.0, 1.0)
output = neural_ode(x, t_span)
print(f"Neural ODE output shape: {output.shape}")
print()

# Example 6: Capsule Networks
print("Example 6: Capsule Networks")
print("-" * 30)

def squash(x, dim=-1):
    """Squashing function for capsules"""
    norm = torch.norm(x, dim=dim, keepdim=True)
    scale = norm**2 / (1 + norm**2)
    return scale * x / norm

class PrimaryCapsule(nn.Module):
    """Primary Capsule Layer"""
    def __init__(self, in_channels, out_channels, num_capsules, capsule_dim, kernel_size=9, stride=2):
        super().__init__()
        self.num_capsules = num_capsules
        self.capsule_dim = capsule_dim
        
        self.conv = nn.Conv2d(in_channels, out_channels * num_capsules, 
                             kernel_size, stride=stride, padding=0)
        
    def forward(self, x):
        outputs = self.conv(x)
        outputs = outputs.view(outputs.size(0), self.num_capsules, -1)
        outputs = outputs.transpose(-1, -2)
        return squash(outputs)

class DigitCapsule(nn.Module):
    """Digit Capsule Layer with Dynamic Routing"""
    def __init__(self, num_capsules, num_routes, in_channels, out_channels, num_iterations=3):
        super().__init__()
        self.num_capsules = num_capsules
        self.num_routes = num_routes
        self.num_iterations = num_iterations
        
        self.W = nn.Parameter(torch.randn(1, num_routes, num_capsules, out_channels, in_channels))
        
    def forward(self, x):
        batch_size = x.size(0)
        x = x[:, None, :, :, None]
        W = self.W.repeat(batch_size, 1, 1, 1, 1)
        
        # Transform inputs
        u_hat = torch.matmul(W, x).squeeze(-1)
        
        # Dynamic routing
        b = torch.zeros(batch_size, self.num_routes, self.num_capsules, 1).to(x.device)
        
        for iteration in range(self.num_iterations):
            c = F.softmax(b, dim=1)
            outputs = squash((c * u_hat).sum(dim=1, keepdim=True))
            
            if iteration < self.num_iterations - 1:
                delta_b = (u_hat * outputs).sum(dim=-1, keepdim=True)
                b = b + delta_b
                
        return outputs.squeeze(1)

# Test Capsule Network components
primary_caps = PrimaryCapsule(256, 32, num_capsules=8, capsule_dim=1152)
digit_caps = DigitCapsule(num_capsules=10, num_routes=1152, in_channels=8, out_channels=16)

# Simulate primary capsule output
primary_output = torch.randn(2, 1152, 8)
digit_output = digit_caps(primary_output)
print(f"Capsule network output shape: {digit_output.shape}")
print()

# Example 7: Self-Supervised Vision Transformer (MAE-style)
print("Example 7: Masked Autoencoder (MAE)")
print("-" * 30)

class MaskedAutoencoder(nn.Module):
    """Masked Autoencoder for self-supervised learning"""
    def __init__(self, img_size=224, patch_size=16, in_channels=3, 
                 embed_dim=768, depth=12, num_heads=12, decoder_embed_dim=512, 
                 decoder_depth=8, decoder_num_heads=16, mask_ratio=0.75):
        super().__init__()
        
        # Encoder
        self.patch_embed = PatchEmbedding(img_size, patch_size, in_channels, embed_dim)
        num_patches = self.patch_embed.num_patches
        
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        
        self.blocks = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads) for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(embed_dim)
        
        # Decoder
        self.decoder_embed = nn.Linear(embed_dim, decoder_embed_dim, bias=True)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))
        self.decoder_pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, decoder_embed_dim))
        
        self.decoder_blocks = nn.ModuleList([
            TransformerBlock(decoder_embed_dim, decoder_num_heads) for _ in range(decoder_depth)
        ])
        self.decoder_norm = nn.LayerNorm(decoder_embed_dim)
        self.decoder_pred = nn.Linear(decoder_embed_dim, patch_size**2 * in_channels, bias=True)
        
        self.mask_ratio = mask_ratio
        
    def random_masking(self, x, mask_ratio):
        """Random masking of patches"""
        N, L, D = x.shape
        len_keep = int(L * (1 - mask_ratio))
        
        noise = torch.rand(N, L, device=x.device)
        ids_shuffle = torch.argsort(noise, dim=1)
        ids_restore = torch.argsort(ids_shuffle, dim=1)
        
        # Keep the first subset
        ids_keep = ids_shuffle[:, :len_keep]
        x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))
        
        # Generate mask: 0 is keep, 1 is remove
        mask = torch.ones([N, L], device=x.device)
        mask[:, :len_keep] = 0
        mask = torch.gather(mask, dim=1, index=ids_restore)
        
        return x_masked, mask, ids_restore
    
    def forward_encoder(self, x, mask_ratio):
        # Embed patches
        x = self.patch_embed(x)
        x = x + self.pos_embed[:, 1:, :]
        
        # Masking
        x, mask, ids_restore = self.random_masking(x, mask_ratio)
        
        # Append cls token
        cls_token = self.cls_token + self.pos_embed[:, :1, :]
        cls_tokens = cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        
        # Apply Transformer blocks
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)
        
        return x, mask, ids_restore
    
    def forward_decoder(self, x, ids_restore):
        # Embed tokens
        x = self.decoder_embed(x)
        
        # Append mask tokens to sequence
        mask_tokens = self.mask_token.repeat(x.shape[0], ids_restore.shape[1] + 1 - x.shape[1], 1)
        x_ = torch.cat([x[:, 1:, :], mask_tokens], dim=1)
        x_ = torch.gather(x_, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, x.shape[2]))
        x = torch.cat([x[:, :1, :], x_], dim=1)
        
        # Add pos embed
        x = x + self.decoder_pos_embed
        
        # Apply Transformer blocks
        for blk in self.decoder_blocks:
            x = blk(x)
        x = self.decoder_norm(x)
        
        # Predictor projection
        x = self.decoder_pred(x)
        x = x[:, 1:, :]
        
        return x
    
    def forward(self, imgs, mask_ratio=None):
        if mask_ratio is None:
            mask_ratio = self.mask_ratio
            
        latent, mask, ids_restore = self.forward_encoder(imgs, mask_ratio)
        pred = self.forward_decoder(latent, ids_restore)
        return pred, mask

# Create MAE model
mae = MaskedAutoencoder(
    img_size=32, 
    patch_size=4, 
    embed_dim=192, 
    depth=6, 
    decoder_embed_dim=128, 
    decoder_depth=4
)

# Test MAE
imgs = torch.randn(4, 3, 32, 32)
reconstruction, mask = mae(imgs)
print(f"MAE reconstruction shape: {reconstruction.shape}")
print(f"Mask shape: {mask.shape}")
print()

# Example 8: Multimodal Architecture (Vision + Language)
print("Example 8: Multimodal Architecture")
print("-" * 30)

class MultimodalTransformer(nn.Module):
    """Simple multimodal transformer for vision and language"""
    def __init__(self, vocab_size, max_seq_len, img_size=224, patch_size=16,
                 embed_dim=512, depth=6, num_heads=8, num_classes=1000):
        super().__init__()
        
        # Vision encoder
        self.vision_embed = PatchEmbedding(img_size, patch_size, 3, embed_dim)
        num_patches = self.vision_embed.num_patches
        
        # Language encoder
        self.text_embed = nn.Embedding(vocab_size, embed_dim)
        self.text_pos_embed = nn.Parameter(torch.zeros(1, max_seq_len, embed_dim))
        
        # Modality embeddings
        self.vision_type_embed = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.text_type_embed = nn.Parameter(torch.zeros(1, 1, embed_dim))
        
        # Shared transformer
        self.blocks = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads) for _ in range(depth)
        ])
        
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)
        
    def forward(self, images=None, text_ids=None):
        embeddings = []
        
        # Process vision input
        if images is not None:
            vision_embeds = self.vision_embed(images)
            vision_embeds = vision_embeds + self.vision_type_embed
            embeddings.append(vision_embeds)
        
        # Process text input
        if text_ids is not None:
            text_embeds = self.text_embed(text_ids)
            seq_len = text_embeds.size(1)
            text_embeds = text_embeds + self.text_pos_embed[:, :seq_len]
            text_embeds = text_embeds + self.text_type_embed
            embeddings.append(text_embeds)
        
        # Concatenate modalities
        x = torch.cat(embeddings, dim=1)
        
        # Apply transformer
        for block in self.blocks:
            x = block(x)
        
        x = self.norm(x)
        x = x.mean(dim=1)  # Global average pooling
        x = self.head(x)
        
        return x

# Test multimodal model
multimodal = MultimodalTransformer(
    vocab_size=10000, 
    max_seq_len=128, 
    img_size=32, 
    patch_size=4,
    embed_dim=256, 
    depth=4, 
    num_classes=100
)

# Test with both modalities
images = torch.randn(2, 3, 32, 32)
text_ids = torch.randint(0, 10000, (2, 20))
output = multimodal(images=images, text_ids=text_ids)
print(f"Multimodal output shape: {output.shape}")
print()

# Performance comparison
print("Model Architecture Comparison")
print("=" * 50)

architectures = {
    "GCN": sum(p.numel() for p in gcn.parameters()),
    "GAT": sum(p.numel() for p in gat.parameters()),
    "Vision Transformer": sum(p.numel() for p in vit.parameters()),
    "EfficientNet": sum(p.numel() for p in efficientnet.parameters()),
    "Neural ODE": sum(p.numel() for p in neural_ode.parameters()),
    "MAE": sum(p.numel() for p in mae.parameters()),
    "Multimodal": sum(p.numel() for p in multimodal.parameters())
}

for name, params in architectures.items():
    print(f"{name}: {params/1e6:.2f}M parameters")

print("\nKey Insights:")
print("- GNNs are efficient for graph-structured data")
print("- ViT scales well but requires large datasets")
print("- EfficientNet provides good accuracy/efficiency trade-off")
print("- Neural ODEs offer continuous-depth models")
print("- Self-supervised methods (MAE) learn without labels")
print("- Multimodal models can process multiple input types")