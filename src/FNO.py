import torch
import torch.nn as nn
import torch.nn.functional as F

class FourierLayer(nn.Module):
    def __init__(self, in_neurons, out_neurons, modesSpace, modesTime, scaling=True):
        super().__init__()
        
        self.in_neurons = in_neurons
        self.out_neurons = out_neurons
        self.modesSpace = modesSpace
        self.modesTime = modesTime
        
        if scaling:
            self.scale = 1 / (self.in_neurons * self.out_neurons)
        else:
            self.scale = 1
            
        self.weights  = nn.Parameter(self.scale * torch.rand(in_neurons, out_neurons, self.modesSpace * 2, self.modesSpace * 2, self.modesTime, dtype=torch.cfloat))


    def compl_mul3d(self, input, weights, einsumBool=True): 
    # (batch, in_channel, x,y,t), (in_channel, out_channel, x,y,t) -> (batch, out_channel, x,y,t)
        if einsumBool: # time for 1 forward t = 0.0082
            return torch.einsum("bixyt,ioxyt->boxyt", input, weights)
        else: # time for 1 forward t = 0.058
            batch_size = input.shape[0]
            # out_neurons = self.weights.shape[1]
            x_size = input.shape[2]
            y_size = input.shape[3]
            t_size = input.shape[4]

            out = torch.zeros(batch_size, self.out_neurons, x_size, y_size, t_size)
            for i in range(t_size):
                for j in range(y_size):
                    for k in range(x_size):
                        out[..., k, j, i] = torch.matmul(input[..., k, j, i], self.weights[..., k, j, i])
            return out
        

    def forward(self, x):
        batchsize = x.shape[0]
        x_ft = torch.fft.rfftn(x, dim=[-3, -2, -1])
        xShapeLast = x.shape[-1]
        del x
        x_ft = torch.fft.fftshift(x_ft, dim=(-3, -2))

        out_ft = torch.zeros(batchsize, self.out_neurons, x_ft.size(-3), x_ft.size(-2), x_ft.size(-1), dtype=torch.cfloat, device=torch.device('cuda')) # device=x.device
        midX, midY =  x_ft.size(-3) // 2, x_ft.size(-2) // 2
        
        out_ft[..., midX - self.modesSpace:midX + self.modesSpace, midY - self.modesSpace:midY + self.modesSpace, :self.modesTime] = \
            self.compl_mul3d(x_ft[..., midX - self.modesSpace:midX + self.modesSpace, midY - self.modesSpace:midY + self.modesSpace, :self.modesTime], self.weights)
        
        del x_ft
        out_ft = torch.fft.fftshift(out_ft, dim=(-3, -2))
        out_ft = torch.fft.irfftn(out_ft, s=(out_ft.size(-3), out_ft.size(-2), xShapeLast))
        return out_ft
    
class MLP(nn.Module):
    def __init__(self, in_neurons, hidden_neurons, out_neurons, kernel_size):
        super().__init__()
        self.mlp1 = nn.Conv3d(in_neurons, hidden_neurons, kernel_size)
        self.mlp2 = nn.Conv3d(hidden_neurons, out_neurons, kernel_size)

    def forward(self, x):
        x = self.mlp1(x)
        x = F.gelu(x)
        x = self.mlp2(x)
        return x
    

class FNOModel(nn.Module):
    def __init__(self, in_neurons, hidden_neurons, out_neurons, modesSpace, modesTime, time_padding, input_size, learning_rate, restart_at_epoch_n, train_loader, loss_function):
        super().__init__()
        #self.save_hyperparameters()
        self.learning_rate = learning_rate
        self.restart_at_epoch_n = restart_at_epoch_n
        self.padding = time_padding # set padding here based on input_size
        self.n_batches = len(train_loader)
        self.n_training_samples = len(train_loader.dataset)
        self.loss_name = loss_function
        #train_batch, _ = next(iter(train_loader))
        #x_shape = train_batch.size()
        #self.register_buffer("meshgrid", get_meshgrid(x_shape))
        
        # Network architechture
        self.p = nn.Linear(input_size, out_neurons)
        
        self.fourier1 = FourierLayer(in_neurons, out_neurons, modesSpace, modesTime)
        self.mlp1 = MLP(in_neurons, hidden_neurons, out_neurons, kernel_size=1)
        self.w1 = nn.Conv3d(in_neurons, out_neurons, kernel_size=1)
        
        self.fourier2 = FourierLayer(in_neurons, out_neurons, modesSpace, modesTime)
        self.mlp2 = MLP(in_neurons, hidden_neurons, out_neurons, kernel_size=1)
        self.w2 = nn.Conv3d(in_neurons, out_neurons, kernel_size=1)
        
        self.fourier3 = FourierLayer(in_neurons, out_neurons, modesSpace, modesTime)
        self.mlp3 = MLP(in_neurons, hidden_neurons, out_neurons, kernel_size=1)
        self.w3 = nn.Conv3d(in_neurons, out_neurons, kernel_size=1)
        
        self.fourier4 = FourierLayer(in_neurons, out_neurons, modesSpace, modesTime)
        self.mlp4 = MLP(in_neurons, hidden_neurons, out_neurons, kernel_size=1)
        self.w4 = nn.Conv3d(in_neurons, out_neurons, kernel_size=1)
        
        self.q = MLP(in_neurons, 4 * hidden_neurons, 1, kernel_size=1) # Single output predicts T timesteps
        
        if loss_function == 'MSE':
            self.loss_function = F.mse_loss
        elif loss_function == 'MAE':
            self.loss_function = F.l1_loss
    
            
    def forward(self, x): # input dim: [B, X, Y, T, T_in]
        #meshgrid = get_meshgrid(x.shape).to(torch.device('cuda'))
        #x = torch.concat((x, meshgrid), dim=-1) # [B, X, Y, T, 3 + T_in]
        #del meshgrid
        x = self.p(x) # [B, X, Y, T, H]
        x = x.permute(0, 4, 1, 2, 3) # [B, H, X, Y, T]
        x = F.pad(x, [0, self.padding]) # Zero-pad
        x1 = self.fourier1(x)
        x1 = self.mlp1(x1)
        x2 = self.w1(x)
        x = x1 + x2
        del x1
        del x2
        x = F.gelu(x)
        
        x1 = self.fourier2(x)
        x1 = self.mlp2(x1)
        x2 = self.w2(x)
        x = x1 + x2
        del x1
        del x2
        x = F.gelu(x)
        
        x1 = self.fourier3(x)
        x1 = self.mlp3(x1)
        x2 = self.w3(x)
        x = x1 + x2
        del x1
        del x2
        x = F.gelu(x)
        
        x1 = self.fourier4(x)
        x1 = self.mlp4(x1)
        x2 = self.w4(x)
        x = x1 + x2
        del x1
        del x2
        
        x = x[..., :-self.padding] # Unpad zeros
        x = self.q(x) # [B, 1, X, Y, T]
        x = x.permute(0, 2, 3, 4, 1)  # [B, X, Y, T, 1]
        x = x.squeeze_(dim=-1)
        return x
    