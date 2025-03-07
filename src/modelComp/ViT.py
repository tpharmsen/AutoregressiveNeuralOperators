import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class PatchEmbedding(nn.Module):
    def __init__(self, img_size=32, patch_size=4, in_channels=3, embed_dim=128):
        super().__init__()
        self.patch_size = patch_size
        self.img_size = img_size
        #print(self.patch_size, self.img_size)
        self.num_patches_h = img_size[0] // patch_size[0]
        self.num_patches_w = img_size[1] // patch_size[1]
        self.num_patches = self.num_patches_h * self.num_patches_w

        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size[0], stride=patch_size[0])
        self.flatten = nn.Flatten(2)
        self.positional_encoding = PositionalEncoding2D(embed_dim, self.num_patches_h, self.num_patches_w)

    def forward(self, x):
        #(self.patch_size, self.img_size)
        x = self.proj(x) 
        x = self.flatten(x).transpose(1, 2)  # (B, C, H*W) -> (B, N_patches, C)
        x = self.positional_encoding(x) 
        return x
"""
class PositionalEncoding2D(nn.Module):
    def __init__(self, d_model, num_patches_h, num_patches_w):
        
        super().__init__()
        self.d_model = d_model
        self.num_patches_h = num_patches_h
        self.num_patches_w = num_patches_w

        pe = torch.zeros(num_patches_h, num_patches_w, d_model)

        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        
        pos_h = torch.arange(num_patches_h).float().unsqueeze(1) * div_term
        pos_w = torch.arange(num_patches_w).float().unsqueeze(1) * div_term

        pe[:, :, 0::2] = torch.sin(pos_h).unsqueeze(1)  
        pe[:, :, 1::2] = torch.cos(pos_w).unsqueeze(0) 

        self.register_buffer('pe', pe.view(num_patches_h * num_patches_w, d_model).unsqueeze(0))

    def forward(self, x):
        #print(x.shape, self.pe.shape)
        return x + self.pe

"""
class PositionalEncoding2D(nn.Module):
    def __init__(self, d_model, num_patches_h, num_patches_w):
        super().__init__()
        self.d_model = d_model
        self.num_patches_h = num_patches_h
        self.num_patches_w = num_patches_w

        # Initialize positional encoding matrix
        pe = torch.zeros(num_patches_h, num_patches_w, d_model)

        # Generate absolute positions for height and width
        pos_h = torch.arange(num_patches_h).float().unsqueeze(1).unsqueeze(2)
        pos_w = torch.arange(num_patches_w).float().unsqueeze(0).unsqueeze(2)

        # Apply sine and cosine functions to the positions
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        
        pe[:, :, 0::2] = torch.sin(pos_h * div_term)
        pe[:, :, 1::2] = torch.cos(pos_w * div_term)

        # Register the positional encoding as a buffer
        self.register_buffer('pe', pe.view(num_patches_h * num_patches_w, d_model).unsqueeze(0))

    def forward(self, x):
        # Add positional encoding to the input
        return x + self.pe

class AttentionHead(nn.Module):
  def __init__(self, d_model, head_size):
    super().__init__()
    self.head_size = head_size

    self.query = nn.Linear(d_model, head_size)
    self.key = nn.Linear(d_model, head_size)
    self.value = nn.Linear(d_model, head_size)

  def forward(self, x):

    Q = self.query(x)
    K = self.key(x)
    V = self.value(x)

    attention = Q @ K.transpose(-2,-1)

    attention = attention / (self.head_size ** 0.5)

    attention = torch.softmax(attention, dim=-1)

    attention = attention @ V

    return attention
  
class MultiHeadAttention(nn.Module):
  def __init__(self, d_model, n_heads):
    super().__init__()
    self.head_size = d_model // n_heads

    self.W_o = nn.Linear(d_model, d_model)

    self.heads = nn.ModuleList([AttentionHead(d_model, self.head_size) for _ in range(n_heads)])

  def forward(self, x):

    out = torch.cat([head(x) for head in self.heads], dim=-1)

    out = self.W_o(out)

    return out
  
class TransformerEncoder(nn.Module):
  def __init__(self, d_model, n_heads, r_mlp=4):
    super().__init__()
    self.d_model = d_model
    self.n_heads = n_heads

    self.ln1 = nn.LayerNorm(d_model)

    self.mha = MultiHeadAttention(d_model, n_heads)

    self.ln2 = nn.LayerNorm(d_model)

    self.mlp = nn.Sequential(
        nn.Linear(d_model, d_model*r_mlp),
        nn.GELU(),
        nn.Linear(d_model*r_mlp, d_model)
    )

  def forward(self, x):

    out = x + self.mha(self.ln1(x))

    out = out + self.mlp(self.ln2(out))

    return out
  
  
class decodeToImage(nn.Module):
  def __init__(self, d_model, patch_size, num_patches_h, num_patches_w, out_channels):
    super().__init__()
    self.d_model = d_model
    self.patch_size = patch_size
    self.num_patches_h = num_patches_h
    self.num_patches_w = num_patches_w
    self.out_channels = out_channels

    self.pixelConv = nn.Conv2d(self.d_model, self.out_channels * self.patch_size[0]**2, kernel_size=1)
    self.pixelShuffle = nn.PixelShuffle(self.patch_size[0])


  def forward(self, x):
    B= x.shape[0]
    x = x.transpose(1,2)
    #print(x.shape)
    x = x.view(B, self.d_model, self.num_patches_h, self.num_patches_w)#.contiguous()
    #print(x.shape)

    x = self.pixelConv(x)
    #print(x.shape)
    x = self.pixelShuffle(x)
    return x

  
  
class VisionTransformer(nn.Module):
  def __init__(self, d_model, img_size, patch_size, in_channels, n_heads, n_layers, out_channels):
    super().__init__()
    #print(img_size, patch_size)

    assert img_size[0] % patch_size[0] == 0 and img_size[1] % patch_size[1] == 0, "img_size dimensions must be divisible by patch_size dimensions"
    assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
    assert patch_size[0] == patch_size[1], "Only square patches are supported for now"

    self.d_model = d_model
    self.img_size = img_size 
    self.patch_size = patch_size 
    self.in_channels = in_channels 
    self.n_heads = n_heads 
    self.out_channels = out_channels

    self.num_patches_h = img_size[0] // patch_size[0]
    self.num_patches_w = img_size[1] // patch_size[1]
    self.num_patches = self.num_patches_h * self.num_patches_w

    self.patch_embedding = PatchEmbedding(self.img_size, self.patch_size, self.in_channels, self.d_model)
    self.transformer_encoder = nn.Sequential(*[TransformerEncoder(self.d_model, self.n_heads) for _ in range(n_layers)])

    self.reconstruct = decodeToImage(self.d_model, self.patch_size, self.num_patches_h, self.num_patches_w, self.out_channels)

  def forward(self, images):
    x = self.patch_embedding(images)

    #x = self.positional_encoding(x)

    x = self.transformer_encoder(x)
    #print(x.shape)
    x = self.reconstruct(x)

    return x