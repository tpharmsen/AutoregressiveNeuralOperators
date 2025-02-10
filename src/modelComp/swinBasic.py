import torch
import torch.nn as nn
from einops import rearrange
from timm.models.swin_transformer import SwinTransformerBlock

class SwinPDEForecaster(nn.Module):
    def __init__(self, img_size=48, patch_size=4, in_channels=3, out_channels=3, embed_dim=128, depths=[2, 2], num_heads=[4, 8], window_size=4):
        """
        SwinPDEForecaster: A Swin Transformer-based model for autoregressive prediction of 2D PDE systems.

        Args:
            img_size (int): Size of the input image (assumed to be square).
            patch_size (int): Size of the patches for patch partitioning.
            in_channels (int): Number of input channels (e.g., temperature, velocity, phase).
            out_channels (int): Number of output channels (same as input for autoregressive prediction).
            embed_dim (int): Embedding dimension for the transformer.
            depths (list): Number of Swin Transformer blocks in each stage.
            num_heads (list): Number of attention heads in each stage.
            window_size (int): Size of the local window for window-based attention.
        """
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.embed_dim = embed_dim
        self.num_stages = len(depths)

        # Patch partition and linear embedding
        self.patch_embed = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.pos_drop = nn.Dropout(p=0.1)

        # Swin Transformer stages
        self.stages = nn.ModuleList()
        for i in range(self.num_stages):
            stage = nn.Sequential(
                *[SwinTransformerBlock(
                    dim=embed_dim * (2 ** i),
                    num_heads=num_heads[i],
                    window_size=window_size,
                    shift_size=0 if (i % 2 == 0) else window_size // 2,
                ) for _ in range(depths[i])]
            )
            self.stages.append(stage)

            # Patch merging (except for the last stage)
            if i < self.num_stages - 1:
                self.stages.append(PatchMerging(embed_dim * (2 ** i)))

        # Decoder to reconstruct the output
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(embed_dim * (2 ** (self.num_stages - 1)), out_channels, kernel_size=patch_size, stride=patch_size),
            nn.Tanh()  # Normalize output to [-1, 1]
        )

    def forward(self, x):
        """
        Forward pass for the SwinPDEForecaster.

        Args:
            x (torch.Tensor): Input tensor of shape (B, C, H, W).

        Returns:
            torch.Tensor: Output tensor of shape (B, C, H, W).
        """
        # Patch embedding
        x = self.patch_embed(x)  # (B, embed_dim, H/patch_size, W/patch_size)
        x = self.pos_drop(x)

        # Swin Transformer stages
        for stage in self.stages:
            x = stage(x)

        # Decoder to reconstruct the output
        x = self.decoder(x)  # (B, out_channels, H, W)
        return x


class PatchMerging(nn.Module):
    """
    Patch Merging layer for downsampling in Swin Transformer.
    """
    def __init__(self, dim):
        super().__init__()
        self.norm = nn.LayerNorm(4 * dim)
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)

    def forward(self, x):
        """
        Forward pass for PatchMerging.

        Args:
            x (torch.Tensor): Input tensor of shape (B, C, H, W).

        Returns:
            torch.Tensor: Output tensor of shape (B, 2*C, H/2, W/2).
        """
        B, C, H, W = x.shape
        x = rearrange(x, 'b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=2, p2=2)  # (B, H/2 * W/2, 4*C)
        x = self.norm(x)
        x = self.reduction(x)  # (B, H/2 * W/2, 2*C)
        x = rearrange(x, 'b (h w) c -> b c h w', h=H//2, w=W//2)  # (B, 2*C, H/2, W/2)
        return x