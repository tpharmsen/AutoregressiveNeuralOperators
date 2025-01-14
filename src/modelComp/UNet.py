import torch
import torch.nn as nn
import torch.nn.functional as F


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
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
        )

class UNet2DTest(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        # Encoder
        # Input: 572x572x3
        self.e11 = nn.Conv2d(in_channels, 64, kernel_size=3, padding=1)  # Output: 570x570x64
        self.e12 = nn.Conv2d(64, 64, kernel_size=3, padding=1)       # Output: 568x568x64
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)           # Output: 284x284x64

        # Input: 284x284x64
        self.e21 = nn.Conv2d(64, 128, kernel_size=3, padding=1)      # Output: 282x282x128
        self.e22 = nn.Conv2d(128, 128, kernel_size=3, padding=1)     # Output: 280x280x128
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)           # Output: 140x140x128

        # Input: 140x140x128
        self.e31 = nn.Conv2d(128, 256, kernel_size=3, padding=1)     # Output: 138x138x256
        self.e32 = nn.Conv2d(256, 256, kernel_size=3, padding=1)     # Output: 136x136x256

        # Decoder
        self.upconv1 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.d11 = nn.Conv2d(256, 128, kernel_size=3, padding=1)
        self.d12 = nn.Conv2d(128, 128, kernel_size=3, padding=1)

        self.upconv2 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.d21 = nn.Conv2d(128, 64, kernel_size=3, padding=1)
        self.d22 = nn.Conv2d(64, 64, kernel_size=3, padding=1)

        # Output layer
        self.outconv = nn.Conv2d(64, out_channels, kernel_size=1)

    def forward(self, x):
        # Encoder
        xe11 = nn.ReLU()(self.e11(x))
        xe12 = nn.ReLU()(self.e12(xe11))
        xp1 = self.pool1(xe12)

        xe21 = nn.ReLU()(self.e21(xp1))
        xe22 = nn.ReLU()(self.e22(xe21))
        xp2 = self.pool2(xe22)

        xe31 = nn.ReLU()(self.e31(xp2))
        xe32 = nn.ReLU()(self.e32(xe31))

        # Decoder
        xu1 = self.upconv1(xe32)
        xu11 = torch.cat([xu1, xe22], dim=1)
        xd11 = nn.ReLU()(self.d11(xu11))
        xd12 = nn.ReLU()(self.d12(xd11))

        xu2 = self.upconv2(xd12)
        xu22 = torch.cat([xu2, xe12], dim=1)
        xd21 = nn.ReLU()(self.d21(xu22))
        xd22 = nn.ReLU()(self.d22(xd21))

        # Output layer
        out = self.outconv(xd22)

        return out
