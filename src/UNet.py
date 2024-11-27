import torch
import torch.nn as nn
import torch.nn.functional as F

class UNet3D(nn.Module):
    def __init__(self, in_neurons, out_neurons, features=(64, 128, 256)):
        super(UNet3D, self).__init__()
        self.loss_function = F.mse_loss

        # Encoder: Down-sampling path
        self.encoder = nn.ModuleList()
        for feature in features:
            self.encoder.append(self._conv_block(in_neurons, feature))
            in_neurons = feature

        # Bottleneck
        self.bottleneck = self._conv_block(features[-1], features[-1] * 2)

        # Decoder: Up-sampling path
        self.decoder = nn.ModuleList()
        for feature in reversed(features):
            self.decoder.append(nn.ConvTranspose3d(feature * 2, feature, kernel_size=2, stride=2))
            self.decoder.append(self._conv_block(feature * 2, feature))

        # Final output layer
        self.final_layer = nn.Conv3d(features[0], out_neurons, kernel_size=1)

    def forward(self, x):
        # Permute input to [Batchsize, Channels, X, Y, T]
        #x = x.permute(0, 1, 2, 3)
        #x = x.permute(0, 2, 3, 1)

        # Save skip connections
        skip_connections = []
        for down in self.encoder:
            x = down(x)
            skip_connections.append(x)
            x = F.max_pool3d(x, kernel_size=2, stride=2)

        # Bottleneck
        x = self.bottleneck(x)

        # Decode with skip connections
        skip_connections = skip_connections[::-1]  # Reverse for decoding
        for idx in range(0, len(self.decoder), 2):
            x = self.decoder[idx](x)  # Transpose convolution
            skip = skip_connections[idx // 2]
            if x.shape != skip.shape:
                x = F.interpolate(x, size=skip.shape[2:])  # Match spatial dimensions
            x = torch.cat((skip, x), dim=1)  # Concatenate along channels
            x = self.decoder[idx + 1](x)  # Conv block

        # Final layer
        x = self.final_layer(x)

        # Permute output back to [Batchsize, X, Y, T, Channels]
        #return x.permute(0, 2, 3, 4, 1)
        #return x.permute(0, 2, 3, 1)
        x.squeeze_(dim=1)
        return x

    @staticmethod
    def _conv_block(in_channels, out_channels):
        return nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
        )