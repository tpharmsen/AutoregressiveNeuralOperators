import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from einops.layers.torch import Rearrange
import numpy as np
import math
from .utils import ConvNeXtBlock, ResNetBlock, SwiGLU, MLP


def window_partition(x, window_size):

    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    return windows

def window_reverse(windows, window_size, H, W):

    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x



class LinearEmbedding(nn.Module):

    def __init__(self, emb_dim = 96, data_dim = (1,4,128,128), patch_size = (8,8), hiddenout_dim = 256, act=nn.GELU):
        super().__init__()
        
        self.B, self.C, self.H, self.W = data_dim
        self.emb_dim = emb_dim
        self.pH, self.pW = patch_size
        self.hiddenout_dim = hiddenout_dim
        self.patch_grid_res = (self.H // self.pH, self.W // self.pW)
        

        assert self.H % self.pH == 0 and self.W % self.pW == 0, "spatial input dim must be divisible by patch_size"
        assert self.H == self.W, "must be square"
        

        self.patchify = nn.Unfold(kernel_size=patch_size, stride=patch_size)
        self.unpatchify = nn.Fold(output_size=(self.H, self.W), kernel_size=patch_size, stride=patch_size)
        self.pre_proj = nn.Sequential(
            nn.Linear(self.C * self.pH * self.pW, self.emb_dim * 2 if act == SwiGLU else self.emb_dim),
            act(),
            nn.Linear(self.emb_dim, self.emb_dim),
        )

        self.post_proj = nn.Sequential(
            nn.Linear(self.emb_dim, self.hiddenout_dim* 2 if act == SwiGLU else self.hiddenout_dim),
            act(),
            nn.Linear(self.hiddenout_dim, self.hiddenout_dim* 2 if act == SwiGLU else self.hiddenout_dim),
            act(),
            nn.Linear(self.hiddenout_dim, self.C * self.pH * self.pW),
        )

    #def get_pos_embeddings(self, t_len):
    #    return (self.time_embed[:, :t_len] + self.patch_position_embeddings).view(1, -1, self.emb_dim)  # (1, t*p*p, d)

    def encode(self, x, proj=True):

        B = x.size(0)
        #print(1, x.shape)
        #x = rearrange(x, "b t c h w -> (b t) c h w") #might change to .permute
        #print(2, x.shape)
        x = self.patchify(x)  
        #print(3, x.shape)
        #x = rearrange(x, "(b t) d pp -> b (t pp) d", b=B) # should think about this
        x = rearrange(x, "b d pp -> b pp d")
        #print(4, x.shape)

        # TODO: add Positional Encoding
        if proj:
            #print(5, x.shape)
            return self.pre_proj(x)#.transpose(1, 2)
        else:
            return x#.transpose(1, 2)

    def decode(self, x, proj=True):
        if proj:
            x = self.post_proj(x)  

        B = x.size(0)
        #x = rearrange(x, "b (t pp) d -> (b t) d pp", pp=self.patch_grid_res[0]*self.patch_grid_res[1]) #might change to .permute
        x = rearrange(x, "b pp d -> b d pp")
        x = self.unpatchify(x)  
        #print(x.shape)
        #x = rearrange(x, "(b t) c h w -> b t c h w", b=B)

        return x

class SpatialPositionalEncoding(nn.Module):
    def __init__(self, emb_dim, emb_amount):
        super().__init__()
        self.emb_dim = emb_dim
        self.H = self.W = int(emb_amount**0.5)
        self.row_embed = nn.Parameter(torch.randn(1, self.H, 1, self.emb_dim // 2))  
        self.col_embed = nn.Parameter(torch.randn(1, 1, self.W, self.emb_dim // 2))  

    def forward(self, x):
        pos = torch.cat([
            self.row_embed.expand(-1, -1, self.W, -1),
            self.col_embed.expand(-1, self.H, -1, -1)
        ], dim=-1)  
        pos = pos.view(1, -1, self.emb_dim)
        return x + pos
 
class PatchMerge(nn.Module):
    def __init__(self, emb_dim):
        super().__init__()
        self.linear = nn.Linear(4*emb_dim, 2*emb_dim)

    def forward(self, x):
        B, L, C = x.shape
        H = W = int(np.sqrt(L)/2)
        x = rearrange(x, 'b (h s1 w s2) c -> b (h w) (s1 s2 c)', s1=2, s2=2, h=H, w=W) #might change to .permute
        x = self.linear(x)
        return x

class PatchUnMerge(nn.Module):
    def __init__(self, emb_dim):
        super().__init__()
        self.linear = nn.Linear(emb_dim, 2*emb_dim)
    
    def forward(self, x):
        B, L, C = x.shape
        H = W = int(np.sqrt(L))
        #print(x.shape, H, W)
        x = self.linear(x)
        #print(x.shape)
        x = rearrange(x, 'b (h w) (s1 s2 c) -> b (h s1 w s2) c', s1=2, s2=2, h=H, w=W) #might change to .permute
        return x


class WindowAttention(nn.Module):

    def __init__(self, emb_dim, window_size, num_heads, qkv_bias=True, attn_drop=0., proj_drop=0.,
    use_flex_attn=True, act=nn.ReLU):

        super().__init__()
        self.emb_dim = emb_dim
        self.window_size = window_size
        self.num_heads = num_heads
        self.qkv_bias = qkv_bias
        self.use_flex_attn = use_flex_attn # original from logits scale BCAT?

        assert emb_dim % num_heads == 0, "embedding dimension must be divisible by number of heads"
        

        if self.use_flex_attn: 
            self.flex_attn = nn.Parameter(torch.log(10 * torch.ones((num_heads, 1, 1))), requires_grad=True)
            #self.register_parameter("flex_attn", nn.Parameter(self.flex_attn))

        # mlp to generate continuous relative position bias
        self.cpb_mlp = nn.Sequential(nn.Linear(2, 512, bias=True),
                                     act(),
                                     nn.Linear(512, num_heads, bias=False))

        # get relative_coords_table
        relative_coords_h = torch.arange(-(self.window_size[0] - 1), self.window_size[0], dtype=torch.float32)
        relative_coords_w = torch.arange(-(self.window_size[1] - 1), self.window_size[1], dtype=torch.float32)
        relative_coords_table = torch.stack(
            torch.meshgrid([relative_coords_h,
                            relative_coords_w])).permute(1, 2, 0).contiguous().unsqueeze(0)  
        
        relative_coords_table[:, :, :, 0] /= (self.window_size[0] - 1)
        relative_coords_table[:, :, :, 1] /= (self.window_size[1] - 1)
        relative_coords_table *= 8  # normalize to -8, 8 TODO: understand why
        relative_coords_table = torch.sign(relative_coords_table) * torch.log2(
            torch.abs(relative_coords_table) + 1.0) / np.log2(8)

        self.register_buffer("relative_coords_table", relative_coords_table)

        # get pair-wise relative position index for each token inside the window
        coords_h = torch.arange(self.window_size[0])
        coords_w = torch.arange(self.window_size[1])
        coords = torch.stack(torch.meshgrid([coords_h, coords_w])) 
        coords_flatten = torch.flatten(coords, 1)  
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  
        relative_coords[:, :, 0] += self.window_size[0] - 1  
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
        relative_position_index = relative_coords.sum(-1)  
        self.register_buffer("relative_position_index", relative_position_index)

        self.qkv = nn.Linear(emb_dim, emb_dim * 3, bias=False)
        if qkv_bias:
            self.q_bias = nn.Parameter(torch.zeros(emb_dim))
            self.v_bias = nn.Parameter(torch.zeros(emb_dim))
            # TODO: understand why not key bias
        else:
            self.q_bias = None
            self.v_bias = None
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj_out = nn.Linear(emb_dim, emb_dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, mask=None):

        B, N, C = x.shape
        qkv_bias = None
        if self.q_bias is not None:
            qkv_bias = torch.cat((self.q_bias, torch.zeros_like(self.v_bias, requires_grad=False), self.v_bias))
        #print(x.shape, self.qkv.weight.shape, qkv_bias.shape)
        qkv = F.linear(input=x, weight=self.qkv.weight, bias=qkv_bias)
        #print('test')
        qkv = qkv.reshape(B, N, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)

        # cosine attention
        attn = (F.normalize(q, dim=-1) @ F.normalize(k, dim=-1).transpose(-2, -1))
        if self.use_flex_attn:
            flex_attn = torch.clamp(self.flex_attn, max=torch.log(torch.tensor(1. / 0.01, device=x.device))).exp()
            attn = attn * flex_attn

        relative_position_bias_table = self.cpb_mlp(self.relative_coords_table).view(-1, self.num_heads)
        relative_position_bias = relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.window_size[0] * self.window_size[1], self.window_size[0] * self.window_size[1], -1) 
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  
        relative_position_bias = 16 * torch.sigmoid(relative_position_bias)
        attn = attn + relative_position_bias.unsqueeze(0)

        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj_out(x)
        x = self.proj_drop(x)
        return x

    
class SwinV2Block(nn.Module): #change name to something else

    def __init__(self, emb_dim, patch_grid_res, num_heads, window_size=4, shift_size=0,
                 mlp_ratio=4., qkv_bias=True, drop=0., attn_drop=0., use_flex_attn=True, 
                 act=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.emb_dim = emb_dim
        self.patch_grid_res = patch_grid_res
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        #self.use_proj_in = use_proj_in
        
        assert 0 <= self.shift_size < self.window_size, "shift_size must in 0-window_size"

        self.norm1 = norm_layer(emb_dim)
        #print('test')
        self.attn = WindowAttention(
            emb_dim, window_size=(self.window_size, self.window_size), num_heads=num_heads,
            qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop, use_flex_attn=use_flex_attn,
            act=nn.ReLU) # act is for relative position bias mlp so maybe not use swiglu/gelu
        #print('test')
        self.norm2 = norm_layer(emb_dim)
        mlp_hidden_dim = int(emb_dim * mlp_ratio)
        self.mlp = MLP(in_features=emb_dim, hidden_features=mlp_hidden_dim, act=act, drop=drop)

        if self.shift_size > 0:
            # calculate attention mask for SW-MSA (from original swin paper source code)
            H, W = self.patch_grid_res
            img_mask = torch.zeros((1, H, W, 1))  # 1 H W 1
            h_slices = (slice(0, -self.window_size),
                        slice(-self.window_size, -self.shift_size),
                        slice(-self.shift_size, None))
            w_slices = (slice(0, -self.window_size),
                        slice(-self.window_size, -self.shift_size),
                        slice(-self.shift_size, None))
            cnt = 0
            for h in h_slices:
                for w in w_slices:
                    img_mask[:, h, w, :] = cnt
                    cnt += 1
            #print('test')
            mask_windows = window_partition(img_mask, self.window_size)  # nW, window_size, window_size, 1
            mask_windows = mask_windows.view(-1, self.window_size * self.window_size)
            attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
            attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))
        else:
            attn_mask = None

        self.register_buffer("attn_mask", attn_mask)

    def forward(self, x):
        H, W = self.patch_grid_res
        B, L, C = x.shape
        #print(L, H, W)
        #print('t', x.shape)
        #if self.use_proj_in:
        #    x = x.transpose(1,2)
        #    x = self.proj_in(x)
        #    x = x.transpose(1,2)
        #else:
        assert L == H * W, "input feature has wrong size for window partitioning (use proj_in=True for projection)"
        #print('u', x.shape)
        shortcut = x
        x = x.view(B, H, W, C)

        if self.shift_size > 0:
            shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
        else:
            shifted_x = x

        x_windows = window_partition(shifted_x, self.window_size)  # nW*B, window_size, window_size, C
        x_windows = x_windows.view(-1, self.window_size * self.window_size, C)  # nW*B, window_size*window_size, C

        # W-MSA/SW-MSA
        attn_windows = self.attn(x_windows, mask=self.attn_mask)  # nW*B, window_size*window_size, C
        #TODO: save attention mask somewhere?
        
        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, C)
        shifted_x = window_reverse(attn_windows, self.window_size, H, W)  # B H' W' C

        if self.shift_size > 0:
            x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        else:
            x = shifted_x
        x = x.view(B, H * W, C)
        x = shortcut + self.norm1(x)

        x = x + self.norm2(self.mlp(x))

        return x
    
class SwinStage(nn.Module): # change name since stage also includes patch merge formally
    
    def __init__(self, emb_dim, patch_grid_res, stage_depth, num_heads, window_size,
                 mlp_ratio=4., qkv_bias=True, drop=0., attn_drop=0., use_flex_attn=True,
                 act=nn.GELU, norm_layer=nn.LayerNorm):

        super().__init__()
        self.emb_dim = emb_dim
        self.patch_grid_res = patch_grid_res
        self.stage_depth = stage_depth

        assert stage_depth % 2 == 0, "stage depth must be divisible by 2"

        # build blocks
        self.blocks = nn.ModuleList([
            SwinV2Block(emb_dim=emb_dim, patch_grid_res=patch_grid_res,
                                 num_heads=num_heads, window_size=window_size,
                                 shift_size=0 if (i % 2 == 0) else window_size // 2,
                                 mlp_ratio=mlp_ratio,
                                 qkv_bias=qkv_bias, 
                                 drop=drop,
                                 attn_drop=attn_drop,
                                 use_flex_attn=use_flex_attn,
                                 act=act,
                                 norm_layer=norm_layer
                                 )
            for i in range(stage_depth)])

    def forward(self, x):
        for blk in self.blocks:
            x = blk(x)
        return x

class SwinUnet(nn.Module):
    def __init__(self, emb_dim, data_dim, patch_size, hiddenout_dim, depth, 
                 stage_depths, num_heads, window_size=8, mlp_ratio=4., 
                 qkv_bias=True, drop=0., attn_drop=0., use_flex_attn=True, norm_layer=nn.LayerNorm,
                 act=nn.GELU, skip_connect=ConvNeXtBlock, gradient_flowthrough=[True, False, False]):
        super().__init__()

        self.embedding = LinearEmbedding(emb_dim, data_dim, patch_size, hiddenout_dim, act)
        self.pos_encoding = SpatialPositionalEncoding(emb_dim, data_dim[2] * data_dim[3] // (patch_size[0] * patch_size[1]))
        
        self.blockDown = nn.ModuleList()
        self.blockMiddle = nn.ModuleList()
        self.blockUp = nn.ModuleList()
        self.patchMerges = nn.ModuleList()
        self.patchUnmerges = nn.ModuleList()
        self.skip_connects = nn.ModuleList()
        # TODO: implement act 

        self.depth = depth
        self.middleblocklen = stage_depths[depth]
        self.gradient_flowthrough = gradient_flowthrough

        for i in range(depth):
            patch_grid_res = (data_dim[2] // (patch_size[0] * 2**i), data_dim[3] // (patch_size[1] * 2**i))
            
            #print(emb_dim * 2**i, patch_grid_res, stage_depths[i], num_heads[i], window_size)
            self.blockDown.append(
                SwinStage(
                    emb_dim * 2**i, 
                    patch_grid_res=patch_grid_res,
                    stage_depth=stage_depths[i], 
                    num_heads=num_heads[i], 
                    window_size=window_size, 
                    mlp_ratio = mlp_ratio, 
                    qkv_bias = qkv_bias, 
                    drop=drop,
                    attn_drop = attn_drop,
                    use_flex_attn = use_flex_attn, 
                    act=act, 
                    norm_layer=norm_layer
                )
            )
            self.patchMerges.append(PatchMerge(emb_dim * 2**i))


        patch_grid_res = (data_dim[2] // (patch_size[0] * 2**depth), data_dim[3] // (patch_size[1] * 2**depth))
        full_window_size = data_dim[2] // (patch_size[0] * 2**depth)
        for i in range(self.middleblocklen):
            self.blockMiddle.append(
                SwinV2Block(
                    emb_dim * 2**depth,
                    patch_grid_res=patch_grid_res,
                    num_heads=num_heads[depth],
                    window_size = full_window_size,
                    shift_size=0,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    drop=0.,
                    attn_drop=0.,
                    use_flex_attn=use_flex_attn,
                    act=act,
                    norm_layer=norm_layer
                )
            )

        for i in reversed(range(depth)):
            patch_grid_res = (data_dim[2] // (patch_size[0] * 2**i), data_dim[3] // (patch_size[1] * 2**i))
            #print(emb_dim * 2**i, patch_grid_res, stage_depths[2*depth - i], num_heads[2*depth - i], window_size)

            self.blockUp.append(
                SwinStage(
                    emb_dim * 2**i, 
                    patch_grid_res=patch_grid_res,
                    stage_depth=stage_depths[2*depth - i], 
                    num_heads=num_heads[2*depth - i], 
                    window_size=window_size, 
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    drop=drop,
                    attn_drop=attn_drop,
                    use_flex_attn=use_flex_attn,
                    act=act,
                    norm_layer=norm_layer
                )
            )
            self.patchUnmerges.append(PatchUnMerge(emb_dim * 2**(i+1)))
            self.skip_connects.append(skip_connect(emb_dim * 2**i))

    def forward(self, x):
        skips = []

        x = self.embedding.encode(x, proj=True)
        x = self.pos_encoding(x)
        
        # ===== DOWN =====
        for i in range(self.depth):
            if self.gradient_flowthrough[0]:
                residual = x
                x = self.blockDown[i](x)
                skips.append(x)
                x = x + residual
            else:
                x = self.blockDown[i](x)
                skips.append(x)

            x = self.patchMerges[i](x)

        # ===== MIDDLE =====
        for i in range(0, self.middleblocklen, 2):
            residual = x
            x = self.blockMiddle[i](x)
            
            if i + 1 < self.middleblocklen:
                x = self.blockMiddle[i + 1](x)
        
            if self.gradient_flowthrough[1]:
                x = x + residual

        # ===== UP =====
        for i in range(self.depth):
            x = self.patchUnmerges[i](x)
            x = x + self.skip_connects[i](skips[self.depth - i - 1])
            if self.gradient_flowthrough[2]:
                residual = x
                x = self.blockUp[i](x) + residual
            else:
                x = self.blockUp[i](x)


        x = self.embedding.decode(x, proj=True)

        return x

