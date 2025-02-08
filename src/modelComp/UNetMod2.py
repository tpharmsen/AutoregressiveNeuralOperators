import torch
import torch.nn as nn

class AttentionBlock(nn.Module):
    def __init__(self, f_g, f_l, f_int):
        super().__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(f_g, f_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(f_int)
        )
        self.W_x = nn.Sequential(
            nn.Conv2d(f_l, f_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(f_int)
        )
        self.psi = nn.Sequential(
            nn.Conv2d(f_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.Sigmoid()
        )
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, g, x):
        psi = self.relu(self.W_g(g) + self.W_x(x))
        psi = self.psi(psi)
        return x * psi

class UNet2D(nn.Module):
    def __init__(self, in_channels, out_channels, depth=4, base_filters=64, activation='relu', multiplier_list=None):
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
        self.attention_blocks = nn.ModuleList()  # ✅ Fixed: Add attention blocks

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

            # ✅ Add Attention Blocks
            self.attention_blocks.append(AttentionBlock(f_g=next_filters, f_l=next_filters, f_int=next_filters // 2))

            current_filters = next_filters

        # Output layer
        self.outconv = nn.Conv2d(base_filters, out_channels, kernel_size=1)

    def conv_block(self, in_channels, out_channels):
        """Creates a convolutional block with BatchNorm and activation."""
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            self.activation,
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            self.activation
        )

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
            skip_connection = self.attention_blocks[i](x, skip_connection)
            x = torch.cat([x, skip_connection], dim=1)  
            x = self.decoders[i](x)

        return self.outconv(x)
