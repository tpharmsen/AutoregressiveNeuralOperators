import torch
import torch.nn as nn
import torch.nn.functional as F

class UNet3D(nn.Module):
    def __init__(self, in_channels, out_channels, features=(64, 128, 256, 512), time_steps=1):
        super(UNet2D, self).__init__()
        self.time_steps = time_steps
        self.encoder = nn.ModuleList([
            self._conv_block(in_channels * time_steps if i == 0 else features[i - 1], feature)
            for i, feature in enumerate(features)
        ])

        self.bottleneck = self._conv_block(features[-1], features[-1] * 2)

        self.decoder = nn.ModuleList([
            nn.Sequential(
                nn.ConvTranspose2d(feature * 2, feature, kernel_size=2, stride=2),
                self._conv_block(feature * 2, feature)
            )
            for feature in reversed(features)
        ])

        self.final_layer = nn.Conv2d(features[0], out_channels * time_steps, kernel_size=1)

    def forward(self, x):
        # Reshape: Combine time steps into channels
        batch_size, time_steps, channels, height, width = x.shape
        x = x.view(batch_size, time_steps * channels, height, width)

        skip_connections = []
        for down in self.encoder:
            x = down(x)
            skip_connections.append(x)
            x = F.max_pool2d(x, kernel_size=2, stride=2)

        x = self.bottleneck(x)
        skip_connections.reverse()

        for idx, up in enumerate(self.decoder):
            x = up[0](x)
            skip = skip_connections[idx]
            x = torch.cat((skip, x), dim=1)
            x = up[1](x)

        x = self.final_layer(x)
        # Reshape: Split time steps from channels
        x = x.view(batch_size, self.time_steps, -1, height, width)
        return x

    @staticmethod
    def _conv_block(in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.GELU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.GELU(),
        )

class UNet2D(nn.Module):
    def __init__(self, in_channels, out_channels, features=(64, 128, 256, 512)):
        super(UNet2D, self).__init__()

        self.encoder = nn.ModuleList([
            self._conv_block(in_channels if i == 0 else features[i - 1], feature)
            for i, feature in enumerate(features)
        ])

        self.bottleneck = self._conv_block(features[-1], features[-1] * 2)

        self.decoder = nn.ModuleList([
            nn.Sequential(
                nn.ConvTranspose2d(feature * 2, feature, kernel_size=2, stride=2),
                self._conv_block(feature * 2, feature)
            )
            for feature in reversed(features)
        ])

        self.final_layer = nn.Conv2d(features[0], out_channels, kernel_size=1)

    def forward(self, x):
        skip_connections = []
        for down in self.encoder:
            x = down(x)
            skip_connections.append(x)
            x = F.max_pool2d(x, kernel_size=2, stride=2)

        x = self.bottleneck(x)
        skip_connections.reverse()

        for idx, up in enumerate(self.decoder):
            x = up[0](x)
            skip = skip_connections[idx]
            #if x.shape != skip.shape:
            #    x = F.interpolate(x, size=skip.shape[2:])
            x = torch.cat((skip, x), dim=1)
            x = up[1](x)

        return self.final_layer(x)

    @staticmethod
    def _conv_block(in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.GELU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.GELU(),
        )
