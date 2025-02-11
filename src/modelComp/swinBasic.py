import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

class PatchEmbedding(nn.Module):
    def __init__(self, in_channels, embed_dim, patch_size):
        super().__init__()
        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)
    
    def forward(self, x):
        return self.proj(x)
    
class WindowAttention(nn.Module):
    def __init__(self, dim, num_heads, window_size):
        super().__init__()
        self.num_heads = num_heads
        self.scale = (dim // num_heads) ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias = False)
        self.proj = nn.Linear(dim, dim)

        self.window_size = window_size
        self.relative_position_bias = nn.Parameter(torch.zeros(
            (2*window_size-1)*(2*window_size-1), num_heads))
    
    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn += self.relative_position_bias.unsqueeze(0)
        attn = attn.softmax(dim=-1)
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        return self.proj(x)
    

class SwinBlock(nn.Module):
    def __init__(self, dim, num_heads, window_size=7):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = WindowAttention(dim, num_heads, window_size)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Linear(dim * 4, dim)
        )

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class PatchMerging(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.reduction = nn.Linear(4 * dim, dim)
        self.norm = nn.LayerNorm(4 * dim)
    
    def forward(self, x):
        B, H, W, C = x.shape
        x = rearrange(x, 'b (h p1) (w p2) c -> b h w (p1 p2 c)', p1=2, p2=2)
        x = self.norm(x)
        return self.reduction(x)

class SwinTransformer(nn.Module):
    def __init__(self, img_size=48, in_channels=1, embed_dim=96, depths=[2, 2], num_heads=[3, 6], window_size=7):
        super().__init__()
        self.patch_embed = PatchEmbedding(in_channels, embed_dim)
        self.layers = nn.ModuleList()
        dim = embed_dim
        for i in range(len(depths)):
            blocks = [SwinBlock(dim, num_heads[i], window_size) for _ in range(depths[i])]
            self.layers.append(nn.Sequential(*blocks))
            if i < len(depths) - 1:
                self.layers.append(PatchMerging(dim))
                dim *= 2
        self.norm = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, in_channels * img_size * img_size // 16)
    
    def forward(self, x):
        x = self.patch_embed(x)
        B, C, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)
        for layer in self.layers:
            x = layer(x)
        x = self.norm(x)
        x = self.head(x)
        x = x.view(B, 1, H * 4, W * 4)  # Upsampling back to original resolution
        return x

# Example Usage
model = SwinTransformer()
x = torch.randn(1, 1, 48, 48)
out = model(x)
print(out.shape)  # Expected output: (1, 1, 48, 48)
