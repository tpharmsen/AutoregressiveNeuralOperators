import torch
import torch.nn as nn
import torch.nn.functional as F

class UNet2D(nn.Module):
    def __init__(self, in_channels, out_channels, depth=4, base_filters=64, activation='relu'):
        super().__init__()
        if activation == 'relu':
            self.activation = nn.ReLU(inplace=True)
        elif activation == 'leaky_relu':
            self.activation = nn.LeakyReLU(inplace=True)
        elif activation == 'gelu':
            self.activation = nn.GELU()
        else:
            raise ValueError('Activation function not supported')
        print(self.activation)

        self.depth = depth
        self.encoders = nn.ModuleList()
        self.pools = nn.ModuleList()
        self.decoders = nn.ModuleList()
        self.upconvs = nn.ModuleList()

        # pre-Encoder
        '''
        self.pre_encoder = nn.Sequential(
            nn.Conv2d(in_channels, base_filters, kernel_size=1),
            nn.BatchNorm2d(base_filters),
            self.activation,
            nn.Conv2d(base_filters, base_filters, kernel_size=1),
            nn.BatchNorm2d(base_filters),
            self.activation
        )
        '''

        # Encoder
        current_filters = base_filters
        for i in range(depth):
            self.encoders.append(
                nn.Sequential(
                    nn.Conv2d(
                        in_channels if i == 0 else current_filters // 2,#base_filters if i == 0 else current_filters // 2,#in_channels if i == 0 else current_filters,
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
        self.outconv = nn.Conv2d(base_filters, out_channels, kernel_size=1)
        #self.outconv2 = nn.Conv2d(base_filters, out_channels, kernel_size=1)

    def forward(self, x):
        encoder_outputs = []
        # pre-Encoder forward
        #x = self.pre_encoder(x)
        #print('\npre-encoder\n', x.shape)
        # Encoder forward
        for i, encoder in enumerate(self.encoders):
            #print(f'encoder {i}', x.shape)
            x = encoder(x)
            #print(f'encoder {i} done', x.shape)
            encoder_outputs.append(x)
            if i < self.depth - 1:
                x = self.pools[i](x)
                #print(f'pool {i} done', x.shape)
        #print('\nhalfway\n')
        # Decoder forward
        for i in range(self.depth - 1):
            #print(f'decoder {i}', x.shape)
            x = self.upconvs[i](x)
            #print(f'upconv {i} done', x.shape)
            skip_connection = encoder_outputs[-(i + 2)]
            x = torch.cat([x, skip_connection], dim=1)
            #print(f'concat {i} done', x.shape)
            x = self.decoders[i](x)
            #print(f'decoder {i} done', x.shape)
        #print('\nend\n')
        # Output layer
        
        x = self.outconv(x)
        return x



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
