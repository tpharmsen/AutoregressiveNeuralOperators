import torch
import torch.nn.functional as F
from torch import nn
from einops import rearrange
from einops.layers.torch import Rearrange
import numpy as np


class SwinEmbedding(nn.Module):
    def __init__(self, patch_size=4, emb_size=96):
        super().__init__()
        self.linear_embedding = nn.Conv2d(14, emb_size, kernel_size = patch_size, stride = patch_size)
        self.rearrange = Rearrange('b c h w -> b (h w) c')
        
    def forward(self, x):
        x = self.linear_embedding(x)
        x = self.rearrange(x)
        return x
    

class PatchMerging(nn.Module):
    def __init__(self, emb_size):
        super().__init__()
        self.linear = nn.Linear(4*emb_size, 2*emb_size)

    def forward(self, x):
        B, L, C = x.shape
        H = W = int(np.sqrt(L)/2)
        x = rearrange(x, 'b (h s1 w s2) c -> b (h w) (s1 s2 c)', s1=2, s2=2, h=H, w=W)
        x = self.linear(x)
        return x
    
class ShiftedWindowMSA(nn.Module):
    def __init__(self, emb_size, num_heads, window_size=7, shifted=True):
        super().__init__()
        self.emb_size = emb_size
        self.num_heads = num_heads
        self.window_size = window_size
        self.shifted = shifted
        self.linear1 = nn.Linear(emb_size, 3*emb_size)
        self.linear2 = nn.Linear(emb_size, emb_size)
        self.pos_embeddings = nn.Parameter(torch.randn(window_size*2 - 1, window_size*2 - 1))
        self.indices = torch.tensor(np.array([[x, y] for x in range(window_size) for y in range(window_size)]))
        self.relative_indices = self.indices[None, :, :] - self.indices[:, None, :]
        self.relative_indices += self.window_size - 1
    def forward(self, x):
        h_dim = self.emb_size / self.num_heads
        height = width = int(np.sqrt(x.shape[1]))
        x = self.linear1(x)
        
        x = rearrange(x, 'b (h w) (c k) -> b h w c k', h=height, w=width, k=3, c=self.emb_size)
        
        if self.shifted:
            x = torch.roll(x, (-self.window_size//2, -self.window_size//2), dims=(1,2))
        
        x = rearrange(x, 'b (Wh w1) (Ww w2) (e H) k -> b H Wh Ww (w1 w2) e k', w1 = self.window_size, w2 = self.window_size, H = self.num_heads)            
        
        Q, K, V = x.chunk(3, dim=6)
        Q, K, V = Q.squeeze(-1), K.squeeze(-1), V.squeeze(-1)
        wei = (Q @ K.transpose(4,5)) / np.sqrt(h_dim)
        
        rel_pos_embedding = self.pos_embeddings[self.relative_indices[:, :, 0], self.relative_indices[:, :, 1]]
        wei += rel_pos_embedding
        
        if self.shifted:
            row_mask = torch.zeros((self.window_size**2, self.window_size**2)).cuda()
            row_mask[-self.window_size * (self.window_size//2):, 0:-self.window_size * (self.window_size//2)] = float('-inf')
            row_mask[0:-self.window_size * (self.window_size//2), -self.window_size * (self.window_size//2):] = float('-inf')
            column_mask = rearrange(row_mask, '(r w1) (c w2) -> (w1 r) (w2 c)', w1=self.window_size, w2=self.window_size)
            wei[:, :, -1, :] += row_mask
            wei[:, :, :, -1] += column_mask
        
        wei = F.softmax(wei, dim=-1) @ V
        
        x = rearrange(wei, 'b H Wh Ww (w1 w2) e -> b (Wh w1) (Ww w2) (H e)', w1 = self.window_size, w2 = self.window_size, H = self.num_heads)
        x = rearrange(x, 'b h w c -> b (h w) c')
        
        return self.linear2(x)
    
class MLP(nn.Module):
    def __init__(self, emb_size):
        super().__init__()
        self.ff = nn.Sequential(
                         nn.Linear(emb_size, 4*emb_size),
                         nn.GELU(),
                         nn.Linear(4*emb_size, emb_size),
                  )
    
    def forward(self, x):
        return self.ff(x)
    
class SwinEncoder(nn.Module):
    def __init__(self, emb_size, num_heads, window_size=7):
        super().__init__()
        self.WMSA = ShiftedWindowMSA(emb_size, num_heads, window_size, shifted=False)
        self.SWMSA = ShiftedWindowMSA(emb_size, num_heads, window_size, shifted=True)
        self.ln = nn.LayerNorm(emb_size)
        self.MLP = MLP(emb_size)
        
    def forward(self, x):
        # Window Attention
        x = x + self.WMSA(self.ln(x))
        x = x + self.MLP(self.ln(x))
        # shifted Window Attention
        x = x + self.SWMSA(self.ln(x))
        x = x + self.MLP(self.ln(x))
        
        return x
    

class decodeToImage(nn.Module):
    def __init__(self, d_model, patch_size, num_patches_h, num_patches_w, out_channels):
        super().__init__()
        self.d_model = d_model
        self.patch_size = patch_size
        self.num_patches_h = num_patches_h
        self.num_patches_w = num_patches_w
        self.out_channels = out_channels

        self.pixelConv = nn.Conv2d(self.d_model, self.out_channels * self.patch_size[0]**2, kernel_size=1)
        self.pixelShuffle = nn.PixelShuffle(self.patch_size[0])


    def forward(self, x):
        B= x.shape[0]
        x = x.transpose(1,2)
        #print(x.shape)
        x = x.view(B, self.d_model, self.num_patches_h, self.num_patches_w)#.contiguous()
        #print(x.shape)

        x = self.pixelConv(x)
        #print(x.shape)
        x = self.pixelShuffle(x)
        return x
    
class Swin(nn.Module):
    def __init__(self):
        super().__init__()
        self.Embedding = SwinEmbedding()
        self.PatchMerging = nn.ModuleList()
        emb_size = 96
        for i in range(3):
            self.PatchMerging.append(PatchMerging(emb_size))
            emb_size *= 2
        
        self.stage1 = SwinEncoder(96, 6, window_size=4)
        self.stage2 = nn.ModuleList([SwinEncoder(192, 12, window_size=4),
                                     SwinEncoder(192, 12, window_size=4),
                                     SwinEncoder(192, 12, window_size=4) 
                                    ])
        #self.stage2 = SwinEncoder(192, 16, window_size=4)
        #self.stage3 = nn.ModuleList([SwinEncoder(384, 12),
        #                             SwinEncoder(384, 12),
        #                             SwinEncoder(384, 12) 
        #                            ])
        #self.stage4 = SwinEncoder(768, 24)
        
        #self.avgpool1d = nn.AdaptiveAvgPool1d(output_size = 1)
        #self.avg_pool_layer = nn.AvgPool1d(kernel_size=49)
        
        #self.layer = nn.Linear(768, num_class)

        self.reconstruct = decodeToImage(192, (8, 8), 12, 12, 12)

    def forward(self, x):
        x = self.Embedding(x)
        #print(x.shape)
        x = self.stage1(x)
        #print(x.shape)
        x = self.PatchMerging[0](x)
        for stage in self.stage2:
            x = stage(x)
        #print(x.shape)
        #x = self.stage2(x)
        #print(x.shape)
        #x = self.PatchMerging[1](x)
        #print(x.shape)
        #for stage in self.stage3:
        #    x = stage(x)
        #print(x.shape)
        #x = self.PatchMerging[2](x)
        #print(x.shape)
        #x = self.stage4(x)
        #print(x.shape)
        #x = self.layer(self.avgpool1d(x.transpose(1, 2)).squeeze(2))
        
        x = self.reconstruct(x)
        #print(x.shape)
        return x
    