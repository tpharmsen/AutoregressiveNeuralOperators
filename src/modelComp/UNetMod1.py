import torch
import torch.nn as nn
class ConvNeXtBlock(nn.Module):
    def __init__(self, in_channels, out_channels, activation='gelu'):
        super().__init__()

        # Activation function mapping
        activations = {
            'relu': nn.ReLU(inplace=True),
            'leaky_relu': nn.LeakyReLU(inplace=True),
            'gelu': nn.GELU()
        }
        if activation not in activations:
            raise ValueError(f"Unsupported activation: {activation}")

        self.activation = activations[activation]

        # Depthwise convolution (only if in_channels == out_channels)
        if in_channels == out_channels:
            self.depthwise_conv = nn.Conv2d(
                in_channels, out_channels, kernel_size=7, padding=3, groups=in_channels
            )
        else:
            self.depthwise_conv = nn.Conv2d(
                in_channels, out_channels, kernel_size=7, padding=3
            )

        # Pointwise convolution and normalization
        self.pointwise_conv1 = nn.Conv2d(out_channels, out_channels, kernel_size=1)
        self.norm1 = nn.LayerNorm(out_channels)  # Normalize over the channel dimension
        self.pointwise_conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=1)
        self.norm2 = nn.LayerNorm(out_channels)  # Normalize over the channel dimension

    def forward(self, x):
        # Depthwise convolution
        x = self.depthwise_conv(x)
        # Pointwise convolution 1
        x = self.pointwise_conv1(x)
        # Permute dimensions for LayerNorm: [batch, channels, height, width] -> [batch, height, width, channels]
        x = x.permute(0, 2, 3, 1)
        x = self.norm1(x)
        x = x.permute(0, 3, 1, 2)  # Permute back to [batch, channels, height, width]
        x = self.activation(x)
        # Pointwise convolution 2
        x = self.pointwise_conv2(x)
        # Permute dimensions for LayerNorm: [batch, channels, height, width] -> [batch, height, width, channels]
        x = x.permute(0, 2, 3, 1)
        x = self.norm2(x)
        x = x.permute(0, 3, 1, 2)  # Permute back to [batch, channels, height, width]
        x = self.activation(x)
        return x

class UNet2D(nn.Module):
    def __init__(self, in_channels, out_channels, depth=4, base_filters=64, activation='gelu', multiplier_list=None):
        super().__init__()

        # Activation function mapping
        activations = {
            'relu': nn.ReLU(inplace=True),
            'leaky_relu': nn.LeakyReLU(inplace=True),
            'gelu': nn.GELU()
        }
        if activation not in activations:
            raise ValueError(f"Unsupported activation: {activation}")

        self.activation = activations[activation]

        # Default multiplier list (if not provided)
        if multiplier_list is None:
            multiplier_list = [2**i for i in range(depth)]  # Default: [1, 2, 4, 8, ...]

        assert len(multiplier_list) == depth, "Multiplier list must match the depth."

        self.depth = depth
        self.encoders = nn.ModuleList()
        self.pools = nn.ModuleList()
        self.decoders = nn.ModuleList()
        self.upconvs = nn.ModuleList()

        # Encoder
        current_filters = base_filters
        for i in range(depth):
            filters = base_filters * multiplier_list[i]
            self.encoders.append(ConvNeXtBlock(in_channels if i == 0 else prev_filters, filters, activation))
            if i < depth - 1:
                self.pools.append(nn.MaxPool2d(kernel_size=2, stride=2))
            prev_filters = filters

        # Bottleneck
        bottleneck_filters = prev_filters * 2
        self.bottleneck = ConvNeXtBlock(prev_filters, bottleneck_filters, activation)

        # Decoder
        current_filters = bottleneck_filters
        for i in range(depth - 1):
            next_filters = base_filters * multiplier_list[depth - 2 - i]
            self.upconvs.append(nn.ConvTranspose2d(current_filters, next_filters, kernel_size=2, stride=2))
            self.decoders.append(ConvNeXtBlock(next_filters + next_filters, next_filters, activation))  # Fix mismatch
            current_filters = next_filters

        # Output layer
        self.outconv = nn.Conv2d(base_filters, out_channels, kernel_size=1)

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
            x = torch.cat([x, skip_connection], dim=1)
            x = self.decoders[i](x)

        return self.outconv(x)