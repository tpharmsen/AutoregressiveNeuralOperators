import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class SwiGLU(nn.Module):
    def forward(self, x):
        x1, x2 = x.chunk(2, dim=-1) 
        return x1 * F.silu(x2)
    
class MLP(nn.Module): #might change name to FFN
    def __init__(self, in_features, hidden_features=None, out_features=None, act=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features * 2 if act == SwiGLU else hidden_features)
        self.act = act()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x
"""
class ConvNeXtBlock(nn.Module):
    #Taken from: https://github.com/facebookresearch/ConvNeXt/blob/main/models/convnext.py
    #ConvNeXt Block. There are two equivalent implementations:
    #(1) DwConv -> LayerNorm (channels_first) -> 1x1 Conv -> GELU -> 1x1 Conv; all in (N, C, H, W)
    #(2) DwConv -> Permute to (N, H, W, C); LayerNorm (channels_last) -> Linear -> GELU -> Linear; Permute back
    #We use (2) as we find it slightly faster in PyTorch
    #
    #Args:
    #    dim (int): Number of input channels.
    #    drop_path (float): Stochastic depth rate. Default: 0.0
    #    layer_scale_init_value (float): Init value for Layer Scale. Default: 1e-6.
    

    def __init__(self, emb_dim, layer_scale_init_value=1e-6, layer_norm_eps=1e-5):
        super().__init__()
        self.dwconv = nn.Conv2d(
            emb_dim, emb_dim, kernel_size=7, padding=3, groups=emb_dim
        )  # depthwise conv
        
        self.norm = nn.LayerNorm(emb_dim, eps=layer_norm_eps)
        self.pwconv1 = nn.Linear(
            emb_dim, 4 * emb_dim
        )  # pointwise/1x1 convs, implemented with linear layers
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(4 * emb_dim, emb_dim)
        self.weight = (
            nn.Parameter(layer_scale_init_value * torch.ones((emb_dim)), requires_grad=True)
            if layer_scale_init_value > 0
            else None
        )  # was gamma before
        

    def forward(self, x):
        batch_size, sequence_length, hidden_size = x.shape
        #! assumes square images
        input_dim = math.floor(sequence_length**0.5)

        input = x
        x = x.reshape(batch_size, input_dim, input_dim, hidden_size)
        #print(x.shape)
        x = x.permute(0, 3, 1, 2)
        x = self.dwconv(x)
        x = x.permute(0, 2, 3, 1)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        if self.weight is not None:
            x = self.weight * x
        x = x.reshape(batch_size, sequence_length, hidden_size)

        x = input + x
        return x
"""

class ConvNeXtBlock(nn.Module):
    def __init__(self, emb_dim, layer_scale_init_value=1e-6, layer_norm_eps=1e-5):
        super().__init__()
        self.dwconv = nn.Conv2d(
            emb_dim, emb_dim, kernel_size=7, padding=3, groups=emb_dim
        )  # depthwise conv

        self.norm = nn.LayerNorm(emb_dim, eps=layer_norm_eps)
        self.pwconv1 = nn.Linear(emb_dim, 4 * emb_dim)
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(4 * emb_dim, emb_dim)

        self.weight = (
            nn.Parameter(layer_scale_init_value * torch.ones((emb_dim)), requires_grad=True)
            if layer_scale_init_value > 0
            else None
        )

    def forward(self, x):
        # x: (B, T, N, E) where N = H * W, E = embedding dim
        print(x.shape)
        B, T, N, E = x.shape
        H = W = int(math.sqrt(N))
        assert H * W == N, f"N={N} is not a perfect square."

        residual = x

        # Reshape to (B*T, E, H, W)
        x = x.view(B * T, H, W, E).permute(0, 3, 1, 2)  # (B*T, E, H, W)
        print(x.shape)
        x = self.dwconv(x)

        # Back to (B*T, H*W, E)
        x = x.permute(0, 2, 3, 1).contiguous().view(B * T, N, E)

        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)

        if self.weight is not None:
            x = self.weight * x

        # Reshape back to (B, T, N, E)
        x = x.view(B, T, N, E)
        x = residual + x
        return x

class ResNetBlock(nn.Module):
    # taken from poseidon code
    def __init__(self, dim):
        super().__init__()
        kernel_size = 3
        pad = (kernel_size - 1) // 2
        self.conv1 = nn.Conv2d(dim, dim, kernel_size=kernel_size, stride=1, padding=pad)
        self.conv2 = nn.Conv2d(dim, dim, kernel_size=kernel_size, stride=1, padding=pad)
        self.bn1 = nn.BatchNorm2d(dim)
        self.bn2 = nn.BatchNorm2d(dim)

    def forward(self, x):
        batch_size, sequence_length, hidden_size = x.shape
        input_dim = math.floor(sequence_length**0.5)

        input = x
        x = x.reshape(batch_size, input_dim, input_dim, hidden_size)
        x = x.permute(0, 3, 1, 2)
        x = self.conv1(x)
        x = self.bn1(x)
        x = nn.functional.leaky_relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = x.permute(0, 2, 3, 1)
        x = x.reshape(batch_size, sequence_length, hidden_size)
        x = x + input
        return x


ACT_MAPPER = {
    "relu": nn.ReLU,
    "gelu": nn.GELU,
    "swiglu": SwiGLU,
    "leaky_relu": nn.LeakyReLU,
}

SKIPBLOCK_MAPPER = {
    "convnext": ConvNeXtBlock,
    "resblock": ResNetBlock,
    "none": None,
}