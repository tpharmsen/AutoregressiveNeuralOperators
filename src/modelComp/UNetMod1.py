import torch
import torch.nn as nn
import torch.nn.functional as F

class UNetMod1(nn.Module):
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
'''
class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ConvBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.gelu = nn.GELU()

    def forward(self, x):
        x = self.gelu(self.conv1(x))
        x = self.gelu(self.conv2(x))
        return x

class UNetPlusPlus(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UNetPlusPlus, self).__init__()
        self.encoder1 = ConvBlock(in_channels, 64)
        self.encoder2 = ConvBlock(64, 128)
        self.encoder3 = ConvBlock(128, 256)

        self.middle = ConvBlock(256, 512)

        self.decoder3 = ConvBlock(512 + 256, 256)
        self.decoder2 = ConvBlock(256 + 128, 128)
        self.decoder1 = ConvBlock(128 + 64, 64)

        self.final_conv = nn.Conv2d(64, out_channels, kernel_size=1)

        # Intermediate convolutions for skip connections
        self.intermediate_conv2 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.intermediate_conv1 = nn.Conv2d(64, 64, kernel_size=3, padding=1)

    def forward(self, x):
        # Encoder path
        enc1 = self.encoder1(x)
        enc2 = self.encoder2(F.max_pool2d(enc1, 2))
        enc3 = self.encoder3(F.max_pool2d(enc2, 2))

        # Middle
        middle = self.middle(F.max_pool2d(enc3, 2))

        # Decoder path with intermediate convolutions in skip connections
        dec3 = self.decoder3(torch.cat([F.interpolate(middle, scale_factor=2, mode='bilinear', align_corners=True), enc3], dim=1))
        dec2 = self.decoder2(torch.cat([F.interpolate(dec3, scale_factor=2, mode='bilinear', align_corners=True), self.intermediate_conv2(enc2)], dim=1))
        dec1 = self.decoder1(torch.cat([F.interpolate(dec2, scale_factor=2, mode='bilinear', align_corners=True), self.intermediate_conv1(enc1)], dim=1))

        # Final output
        out = self.final_conv(dec1)
        return out
'''