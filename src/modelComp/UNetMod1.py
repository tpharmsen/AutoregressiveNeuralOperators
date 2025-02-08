import torch
import torch.nn as nn
import torch.nn.functional as F

class EfficientSelfAttention(nn.Module):
    def __init__(self, in_channels, reduction_ratio=4):
        super().__init__()
        reduced_channels = max(in_channels // reduction_ratio, 8)  # Ensure minimum feature map size
        
        self.query = nn.Conv2d(in_channels, reduced_channels, kernel_size=1)
        self.key = nn.Conv2d(in_channels, reduced_channels, kernel_size=1)
        self.value = nn.Conv2d(in_channels, in_channels, kernel_size=1)

        self.spatial_pool = nn.AvgPool2d(kernel_size=2, stride=2)  # Downsampling for efficiency
        self.gamma = nn.Parameter(torch.zeros(1))  # Scaling factor

    def forward(self, x):
        batch, channels, height, width = x.shape
        
        # Downsample key and value
        key = self.spatial_pool(self.key(x))  # (B, C//reduction, H/2, W/2)
        value = self.spatial_pool(self.value(x))  # (B, C, H/2, W/2)

        # Compute query on full resolution
        query = self.query(x)  # (B, C//reduction, H, W)
        
        # Flatten spatial dimensions
        query = query.view(batch, -1, height * width).permute(0, 2, 1)  # (B, H*W, C//reduction)
        key = key.view(batch, -1, (height // 2) * (width // 2))  # (B, C//reduction, (H/2)*(W/2))
        value = value.view(batch, -1, (height // 2) * (width // 2))  # (B, C, (H/2)*(W/2))

        # Compute attention
        attn_map = torch.bmm(query, key) / key.shape[1]  # Normalize
        attn_map = torch.softmax(attn_map, dim=-1)  # (B, H*W, (H/2)*(W/2))

        # Weighted sum
        attn_output = torch.bmm(value, attn_map.permute(0, 2, 1))  # (B, C, H*W)
        attn_output = attn_output.view(batch, channels, height, width)  # Reshape to original

        return self.gamma * attn_output + x  # Skip connection

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
            self.attention_blocks.append(EfficientSelfAttention(current_filters))  # Add attention to skip connection

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
