import torch
import torch.nn as nn

class UNet2D(nn.Module):
    def __init__(self, in_channels, out_channels, depth=4, base_filters=64, activation='relu', multiplier_list=None, num_heads=4):
        super().__init__()

        activations = {
            'relu': nn.ReLU(inplace=True),
            'leaky_relu': nn.LeakyReLU(inplace=True),
            'gelu': nn.GELU()
        }
        if activation not in activations:
            raise ValueError(f"Unsupported activation: {activation}")

        self.activation = activations[activation]

        if multiplier_list is None:
            multiplier_list = [2**i for i in range(depth)]

        assert len(multiplier_list) == depth, "Multiplier list must match the depth."

        self.depth = depth
        self.encoders = nn.ModuleList()
        self.pools = nn.ModuleList()
        self.decoders = nn.ModuleList()
        self.upconvs = nn.ModuleList()
        self.attention_layers = nn.ModuleList()

        # Encoder
        current_filters = base_filters
        for i in range(depth):
            filters = base_filters * multiplier_list[i]
            self.encoders.append(self.conv_block(in_channels if i == 0 else prev_filters, filters))
            if i < depth - 1:
                self.pools.append(nn.MaxPool2d(kernel_size=2, stride=2))
            prev_filters = filters

        # Bottleneck
        bottleneck_filters = prev_filters * 2
        self.bottleneck = self.conv_block(prev_filters, bottleneck_filters)

        # Decoder
        current_filters = bottleneck_filters
        for i in range(depth - 1):
            next_filters = base_filters * multiplier_list[depth - 2 - i]
            self.upconvs.append(nn.ConvTranspose2d(current_filters, next_filters, kernel_size=2, stride=2))
            self.decoders.append(self.conv_block(next_filters + next_filters, next_filters))

            # Replace Attention Block with MultiheadAttention
            self.attention_layers.append(nn.MultiheadAttention(embed_dim=next_filters, num_heads=num_heads, batch_first=True))

            current_filters = next_filters

        # Output layer
        self.outconv = nn.Conv2d(base_filters, out_channels, kernel_size=1)

    def conv_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            self.activation,
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            self.activation
        )

    def apply_attention(self, x, skip, attention_layer):
        """Applies multihead attention on 2D feature maps."""
        B, C, H, W = skip.shape
        x = x.view(B, C, H * W).permute(0, 2, 1)  # (B, HW, C)
        skip = skip.view(B, C, H * W).permute(0, 2, 1)  # (B, HW, C)

        attn_output, _ = attention_layer(skip, x, x)  # Apply attention
        attn_output = attn_output.permute(0, 2, 1).view(B, C, H, W)  # Reshape back

        return attn_output

    def forward(self, x):
        encoder_outputs = []

        for i, encoder in enumerate(self.encoders):
            x = encoder(x)
            encoder_outputs.append(x)
            if i < self.depth - 1:
                x = self.pools[i](x)

        x = self.bottleneck(x)

        for i in range(self.depth - 1):
            x = self.upconvs[i](x)
            skip_connection = encoder_outputs[-(i + 2)]
            skip_connection = self.apply_attention(x, skip_connection, self.attention_layers[i])
            x = torch.cat([x, skip_connection], dim=1)  
            x = self.decoders[i](x)

        return self.outconv(x)
