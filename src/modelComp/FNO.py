import torch
import torch.nn as nn
import torch.fft

class FNO2d(nn.Module):
    def __init__(self, in_channels, out_channels, modes1, modes2, width):
        super(FNO2d, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1
        self.modes2 = modes2
        self.width = width

        self.fc0 = nn.Linear(self.in_channels, self.width)
        self.fourier_layers = nn.ModuleList([FourierLayer(self.width, self.modes1, self.modes2) for _ in range(4)])
        self.fc1 = nn.Linear(self.width, 128)
        self.fc2 = nn.Linear(128, self.out_channels)

    def forward(self, x):
        batchsize = x.shape[0]
        size_x = x.shape[-2]
        size_y = x.shape[-1]

        x = x.view(batchsize, self.in_channels, -1).permute(0, 2, 1)
        x = self.fc0(x)
        x = x.permute(0, 2, 1).view(batchsize, self.width, size_x, size_y)

        for layer in self.fourier_layers:
            x = layer(x)

        x = x.view(batchsize, self.width, -1).permute(0, 2, 1)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        x = x.permute(0, 2, 1).view(batchsize, self.out_channels, size_x, size_y)
        return x

class FourierLayer(nn.Module):
    def __init__(self, width, modes1, modes2):
        super(FourierLayer, self).__init__()
        self.width = width
        self.modes1 = modes1
        self.modes2 = modes2
        self.scale = 1 / (width * width)
        self.weights1 = nn.Parameter(self.scale * torch.rand(width, modes1, modes2, 2))
        self.weights2 = nn.Parameter(self.scale * torch.rand(width, modes1, modes2, 2))

    def forward(self, x):
        x_ft = torch.fft.rfftn(x, dim=(-2, -1))
        out_ft = torch.zeros_like(x_ft, dtype=torch.cfloat)

        out_ft[:, :, :self.modes1, :self.modes2] = compl_mul2d(x_ft[:, :, :self.modes1, :self.modes2], self.weights1)
        out_ft[:, :, -self.modes1:, :self.modes2] = compl_mul2d(x_ft[:, :, -self.modes1:, :self.modes2], self.weights2)

        x = torch.fft.irfftn(out_ft, s=x.shape[-2:])
        return x

def compl_mul2d(a, b):
    return torch.einsum("bixy,ioxy->boxy", a, torch.view_as_complex(b))
