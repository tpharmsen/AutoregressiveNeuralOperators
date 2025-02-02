import torch
import torch.nn as nn
import torch.nn.functional as F

class UNetMod2(nn.Module):
    def __init__(self, in_channels, out_channels, features, activation='relu'):
        super().__init__()

        if activation == 'relu':
            self.activation = nn.ReLU(inplace=True)
        elif activation == 'leaky_relu':
            self.activation = nn.LeakyReLU(inplace=True)
        elif activation == 'gelu':
            self.activation = nn.GELU()
        else:
            raise ValueError('Activation function not supported')

        self.features = features
        self.depth = len(features)
        
        self.encoders = nn.ModuleList()
        self.pools = nn.ModuleList()
        self.decoders = nn.ModuleList()
        self.upconvs = nn.ModuleList()

        # Encoder
        current_filters = features[0]
        for i in range(self.depth):
            self.encoders.append(
                nn.Sequential(
                    nn.Conv2d(
                        in_channels if i == 0 else features[i-1],  # Input channels based on previous layer's output
                        current_filters,
                        kernel_size=3,
                        padding=1
                    ),
                    nn.BatchNorm2d(current_filters),
                    self.activation,
                    nn.Conv2d(
                        current_filters,
                        current_filters,
                        kernel_size=3,
                        padding=1
                    ),
                    nn.BatchNorm2d(current_filters),
                    self.activation
                )
            )
            if i < self.depth - 1:
                self.pools.append(nn.MaxPool2d(kernel_size=2, stride=2))
                current_filters = features[i + 1]

        # Decoder
        for i in range(self.depth - 1):
            current_filters = features[self.depth - i - 2]  # Reverse the filter size for upconvs
            self.upconvs.append(
                nn.ConvTranspose2d(features[self.depth - i - 1] * 2, current_filters, kernel_size=2, stride=2)
            )
            self.decoders.append(
                nn.Sequential(
                    nn.Conv2d(
                        current_filters * 2,
                        current_filters,
                        kernel_size=3,
                        padding=1
                    ),
                    nn.BatchNorm2d(current_filters),
                    self.activation,
                    nn.Conv2d(
                        current_filters,
                        current_filters,
                        kernel_size=3,
                        padding=1
                    ),
                    nn.BatchNorm2d(current_filters),
                    self.activation
                )
            )

        # Output layer
        self.outconv = nn.Conv2d(features[0], out_channels, kernel_size=1)

    def forward(self, x):
        encoder_outputs = []

        # Encoder forward
        for i, encoder in enumerate(self.encoders):
            x = encoder(x)
            encoder_outputs.append(x)
            if i < self.depth - 1:
                x = self.pools[i](x)

        # Decoder forward
        for i in range(self.depth - 1):
            x = self.upconvs[i](x)
            skip_connection = encoder_outputs[-(i + 2)]
            x = torch.cat([x, skip_connection], dim=1)
            x = self.decoders[i](x)

        # Output layer
        x = self.outconv(x)
        return x