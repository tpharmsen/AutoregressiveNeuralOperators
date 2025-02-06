import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

class PatchEmbedding(nn.Module):
    def __init__(self, in_channels, embed_dim, patch_size):
        super().__init__()
        self.patch_size = patch_size
        self.projection = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)
    
    def forward(self, x):
        print(x.shape)
        x = self.projection(x)  # (B, embed_dim, H/P, W/P)
        print(x.shape)
        x = rearrange(x, "b (h w) (p1 p2 c) -> b c (h p1) (w p2)", h=4, w=4, p1=12, p2=12, c=20)  # Flatten to patch tokens
        return x

class TransformerBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, mlp_ratio=4.0, dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, int(embed_dim * mlp_ratio)),
            nn.GELU(),
            nn.Linear(int(embed_dim * mlp_ratio), embed_dim)
        )
    
    def forward(self, x):
        x = x + self.attn(self.norm1(x), self.norm1(x), self.norm1(x))[0]
        x = x + self.mlp(self.norm2(x))
        return x

class ViT(nn.Module):
    def __init__(self, in_channels, out_channels, img_size=128, patch_size=16, embed_dim=256, depth=6, num_heads=8):
        super().__init__()
        self.patch_embed = PatchEmbedding(in_channels, embed_dim, patch_size)
        num_patches = (img_size // patch_size) ** 2
        self.pos_embed = nn.Parameter(torch.randn(1, num_patches, embed_dim))
        self.encoder = nn.Sequential(*[TransformerBlock(embed_dim, num_heads) for _ in range(depth)])
        self.decoder = nn.Linear(embed_dim, patch_size * patch_size * out_channels)
        self.patch_size = patch_size
        self.out_channels = out_channels
        
    def forward(self, x):
        x = self.patch_embed(x)  # Convert to patch tokens
        x += self.pos_embed  # Add position encoding
        x = self.encoder(x)  # Apply transformer blocks
        x = self.decoder(x)  # Project back to image patches
        x = rearrange(x, 'b (h w) (p1 p2 c) -> b c (h p1) (w p2)', p1=self.patch_size, p2=self.patch_size, c=self.out_channels)
        return x
