import torch
import torch.nn as nn
import torch.nn.functional as F

from collections import OrderedDict

class UNet3D(nn.Module):
    def __init__(self, in_channels, out_channels, features=(64, 128, 256, 512), time_steps=1):
        super(UNet3D, self).__init__()
        self.time_steps = time_steps
        self.encoder = nn.ModuleList([
            self._conv_block(in_channels * time_steps if i == 0 else features[i - 1], feature)
            for i, feature in enumerate(features)
        ])

        self.bottleneck = self._conv_block(features[-1], features[-1] * 2)

        self.decoder = nn.ModuleList([
            nn.Sequential(
                nn.ConvTranspose3d(feature * 2, feature, kernel_size=2, stride=2),
                self._conv_block(feature * 2, feature)
            )
            for feature in reversed(features)
        ])

        self.final_layer = nn.Conv3d(features[0], out_channels * time_steps, kernel_size=1)

    def forward(self, x):
        # Reshape: Combine time steps into channels
        batch_size, time_steps, channels, height, width = x.shape
        x = x.view(batch_size, time_steps * channels, height, width)

        skip_connections = []
        for down in self.encoder:
            x = down(x)
            skip_connections.append(x)
            x = F.max_pool3d(x, kernel_size=2, stride=2)

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
            nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.GELU(),
            nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_channels),
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


class UNetTest(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UNetTest, self).__init__()
        
        # Contracting path
        self.enc1 = self.conv_block(in_channels, 64)
        self.enc2 = self.conv_block(64, 128)
        self.enc3 = self.conv_block(128, 256)

        # Bottleneck
        self.bottleneck = self.conv_block(256, 512)

        # Expanding path
        self.upconv3 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.dec3 = self.conv_block(512, 256)
        self.upconv2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.dec2 = self.conv_block(256, 128)
        self.upconv1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec1 = self.conv_block(128, 64)

        # Final output layer
        self.final = nn.Conv2d(64, out_channels, kernel_size=1)

    def conv_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        # Encoder
        enc1 = self.enc1(x)
        enc2 = self.enc2(F.max_pool2d(enc1, kernel_size=2))
        enc3 = self.enc3(F.max_pool2d(enc2, kernel_size=2))

        # Bottleneck
        bottleneck = self.bottleneck(F.max_pool2d(enc3, kernel_size=2))

        # Decoder
        dec3 = self.upconv3(bottleneck)
        dec3 = torch.cat((enc3, dec3), dim=1)
        dec3 = self.dec3(dec3)
        dec2 = self.upconv2(dec3)
        dec2 = torch.cat((enc2, dec2), dim=1)
        dec2 = self.dec2(dec2)
        dec1 = self.upconv1(dec2)
        dec1 = torch.cat((enc1, dec1), dim=1)
        dec1 = self.dec1(dec1)

        # Final output
        return self.final(dec1)


class UNetBubbleML(nn.Module):
    def __init__(self, in_channels, out_channels, init_features=32):
        super(UNetBubbleML, self).__init__()
        features = init_features
        self.encoder1 = UNetBubbleML._block(in_channels, features, name="enc1")
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder2 = UNetBubbleML._block(features, features * 2, name="enc2")
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder3 = UNetBubbleML._block(features * 2, features * 4, name="enc3")
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.bottleneck = UNetBubbleML._block(features * 4, features * 8, name="bottleneck")

        self.upconv3 = nn.ConvTranspose2d(
            features * 8, features * 4, kernel_size=2, stride=2
        )
        self.decoder3 = UNetBubbleML._block((features * 4) * 2, features * 4, name="dec3")
        self.upconv2 = nn.ConvTranspose2d(
            features * 4, features * 2, kernel_size=2, stride=2
        )
        self.decoder2 = UNetBubbleML._block((features * 2) * 2, features * 2, name="dec2")
        self.upconv1 = nn.ConvTranspose2d(
            features * 2, features, kernel_size=2, stride=2
        )
        self.decoder1 = UNetBubbleML._block(features * 2, features, name="dec1")

        self.conv = nn.Conv2d(
            in_channels=features, out_channels=out_channels, kernel_size=1
        )

    def forward(self, x):
        enc1 = self.encoder1(x)
        enc2 = self.encoder2(self.pool1(enc1))
        enc3 = self.encoder3(self.pool2(enc2))

        bottleneck = self.bottleneck(self.pool4(enc3))

        dec3 = self.upconv3(bottleneck)
        dec3 = torch.cat((dec3, enc3), dim=1)
        dec3 = self.decoder3(dec3)
        dec2 = self.upconv2(dec3)
        dec2 = torch.cat((dec2, enc2), dim=1)
        dec2 = self.decoder2(dec2)
        dec1 = self.upconv1(dec2)
        dec1 = torch.cat((dec1, enc1), dim=1)
        dec1 = self.decoder1(dec1)
        return self.conv(dec1)

    @staticmethod
    def _block(in_channels, features, name):
        return nn.Sequential(
            OrderedDict(
                [
                    (
                        name + "conv1",
                        nn.Conv2d(
                            in_channels=in_channels,
                            out_channels=features,
                            kernel_size=3,
                            padding=1,
                            bias=False,
                        ),
                    ),
                    (name + "norm1", nn.BatchNorm2d(num_features=features)),
                    (name + "tanh1", nn.GELU()),
                    (
                        name + "conv2",
                        nn.Conv2d(
                            in_channels=features,
                            out_channels=features,
                            kernel_size=3,
                            padding=1,
                            bias=False,
                        ),
                    ),
                    (name + "norm2", nn.BatchNorm2d(num_features=features)),
                    (name + "tanh2", nn.GELU()),
                ]
            )
        )
    