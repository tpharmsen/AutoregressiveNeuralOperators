import torch
import torch.nn as nn
import torch.nn.functional as F

class SelfAttention(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.query = nn.Conv2d(in_channels, in_channels // 8, kernel_size=1)
        self.key = nn.Conv2d(in_channels, in_channels // 8, kernel_size=1)
        self.value = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))  # Scaling parameter

    def forward(self, x):
        batch, channels, height, width = x.shape
        query = self.query(x).view(batch, -1, height * width).permute(0, 2, 1)  # (B, H*W, C//8)
        key = self.key(x).view(batch, -1, height * width)  # (B, C//8, H*W)
        attn_map = torch.softmax(torch.bmm(query, key), dim=-1)  # (B, H*W, H*W)

        value = self.value(x).view(batch, -1, height * width)  # (B, C, H*W)
        attn_output = torch.bmm(value, attn_map.permute(0, 2, 1))  # (B, C, H*W)
        attn_output = attn_output.view(batch, channels, height, width)  # Reshape

        return self.gamma * attn_output + x  # Skip connection with learned scaling


class UNet2D(nn.Module):
    def __init__(self, in_channels, out_channels, depth=4, base_filters=64, activation='relu'):
        super().__init__()

        # Define activation function
        if activation == 'relu':
            self.activation = nn.ReLU(inplace=True)
        elif activation == 'leaky_relu':
            self.activation = nn.LeakyReLU(inplace=True)
        elif activation == 'gelu':
            self.activation = nn.GELU()
        else:
            raise ValueError('Activation function not supported')

        self.depth = depth
        self.encoders = nn.ModuleList()
        self.pools = nn.ModuleList()
        self.decoders = nn.ModuleList()
        self.upconvs = nn.ModuleList()
        self.attention_blocks = nn.ModuleList()  # Self-attention layers for skip connections

        # Encoder
        current_filters = base_filters
        for i in range(depth):
            self.encoders.append(
                nn.Sequential(
                    nn.Conv2d(
                        in_channels if i == 0 else current_filters // 2,
                        current_filters,
                        kernel_size=3,
                        padding=1
                    ),
                    nn.GroupNorm(num_groups=8, num_channels=current_filters),
                    self.activation,
                    nn.Conv2d(current_filters, current_filters, kernel_size=3, padding=1),
                    nn.GroupNorm(num_groups=8, num_channels=current_filters),
                    self.activation
                )
            )
            if i < depth - 1:
                self.pools.append(nn.MaxPool2d(kernel_size=2, stride=2))
                current_filters *= 2

        # Decoder
        for i in range(depth - 1):
            current_filters //= 2
            self.upconvs.append(
                nn.ConvTranspose2d(current_filters * 2, current_filters, kernel_size=2, stride=2)
            )
            self.decoders.append(
                nn.Sequential(
                    nn.Conv2d(current_filters * 2, current_filters, kernel_size=3, padding=1),
                    nn.GroupNorm(num_groups=8, num_channels=current_filters),
                    self.activation,
                    nn.Conv2d(current_filters, current_filters, kernel_size=3, padding=1),
                    nn.GroupNorm(num_groups=8, num_channels=current_filters),
                    self.activation
                )
            )
            self.attention_blocks.append(SelfAttention(current_filters))  # Add attention to skip connection

        # Output layer
        self.outconv = nn.Conv2d(base_filters, out_channels, kernel_size=1)

    def forward(self, x):
        encoder_outputs = []
        
        # Encoder forward
        for i, encoder in enumerate(self.encoders):
            x = encoder(x)
            encoder_outputs.append(x)  # Store skip connections
            if i < self.depth - 1:
                x = self.pools[i](x)

        # Decoder forward
        for i in range(self.depth - 1):
            x = self.upconvs[i](x)
            
            # Apply self-attention to the skip connection
            skip_connection = encoder_outputs[-(i + 2)]
            skip_connection = self.attention_blocks[i](skip_connection)  # Apply attention

            x = torch.cat([x, skip_connection], dim=1)  # Concatenate skip connections
            x = self.decoders[i](x)

        return self.outconv(x)
