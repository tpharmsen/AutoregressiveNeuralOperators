import lightning as L
from lightning.pytorch.loggers import WandbLogger
from torch.utils.data.distributed import DistributedSampler
import torch.distributed as dist
from lightning.pytorch.utilities.rank_zero import rank_zero_only
from lightning.pytorch.callbacks import ModelCheckpoint
import torch 
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, ConcatDataset, random_split, Subset
from torch.optim.lr_scheduler import StepLR, ReduceLROnPlateau, CosineAnnealingLR, SequentialLR, ConstantLR
from datetime import datetime
import time
import wandb
import yaml
import random
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import matplotlib.animation as animation
import os
import subprocess
import platform
import json

from dataloaders import *
from dataloaders import PREPROC_MAPPER
from dataloaders.utils import get_dataset, ZeroShotSampler, spatial_resample
#from trainers.utils import make_plot, animate_rollout, magnitude_vel, rollout
from trainers.utils import animate_rollout, magnitude_vel, rollout, compute_energy_enstrophy_spectra
from modelComp.utils import ACT_MAPPER, SKIPBLOCK_MAPPER

from trainers.MTT import MTT as MTTbase
from trainers.MTT import MTTmodel, MTTdata

plt.style.use('dark_background')
plt.rcParams['figure.facecolor'] = '#1F1F1F'
plt.rcParams['axes.facecolor'] = '#1F1F1F'
plt.rcParams['savefig.facecolor'] = '#1F1F1F'


# following is a gpu mig bug fix
if "MIG" in subprocess.check_output(["nvidia-smi", "-L"], text=True):
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    print('MIG GPU detected, using GPU 0')
else:
    print('No MIG GPU detected, using all available GPUs')

torch.set_float32_matmul_precision('medium')

class FlowMatching(MTTbase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


class FMmodel(MTTmodel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    
    def _initialize_model(self):
        if self.cm.model_name == "FluidGPT_FM":
            from modelComp.FluidGPT_FM import FluidGPT_FM
            self.model = FluidGPT_FM(emb_dim=96,
                            data_dim=[self.ct.batch_size, self.cm.temporal_bundling, self.cm.in_channels, self.cd.resample_shape, self.cd.resample_shape],
                            patch_size=(self.cm.patch_size, self.cm.patch_size),
                            hiddenout_dim=self.cm.hiddenout_dim,
                            flowmatching_emb_dim=self.cm.flowmatching_emb_dim,
                            depth=self.cm.depth,
                            stage_depths=self.cm.stage_depths,
                            num_heads=self.cm.num_heads,
                            window_size=self.cm.window_size,
                            use_flex_attn=self.cm.use_flex_attn,
                            act=ACT_MAPPER[self.cm.act],
                            skip_connect=SKIPBLOCK_MAPPER[self.cm.skipblock],
                            gradient_flowthrough=self.cm.gradient_flowthrough,
                            )
        else:
            raise ValueError('MODEL NOT RECOGNIZED') 
    def random_fft_perturb(x, perturbation_strength=1.0):
        x_fft = torch.fft.fftshift(torch.fft.fft2(x, dim=(-2, -1)))
        x_fft = x_fft * torch.exp(1j * torch.randn_like(x_fft) * perturbation_strength)
        x_recon = torch.fft.ifft2(torch.fft.ifftshift(x_fft)).real
        return x_recon

    def training_step(self, batch, batch_idx):
        front, label = batch

        pred = self(front)
        train_loss = F.mse_loss(pred, label)
        self.train_losses.append(train_loss.item())
        #return train_loss
        xnoise = random_fft_perturb(front, perturbation_strength=ratio)
        target = y - xnoise
        #t = torch.zeros(y.size(0), device=y.device)
        t = torch.rand(y.size(0), device=y.device)
        xt = (1 - t[:, None, None, None, None]) * xnoise + t[:, None, None, None, None] * y
        pred = model(xt, t)

        #loss = F.mse_loss(pred, target)
        loss = ((pred - target)**2).mean()
        #loss = (F.mse_loss(pred, target, reduction='none')).mean()
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        losses.append(loss.item())
        epoch_losses.append(loss.item())
        return 

    def validation_step(self, batch, batch_idx, dataloader_idx):

        front, label = batch
        
        if dataloader_idx == 0:
            pred = self(front)
            val_loss = F.mse_loss(pred, label)
            self.val_SS_losses.append(val_loss.item())
            self.log("val_SS_loss", val_loss, on_epoch=True, prog_bar=False, sync_dist=True)
        elif dataloader_idx == 1:
            # a few forward steps for the forward step loss
            pred = front
            for _ in range(self.ct.forward_steps_loss):
                pred = self.model(pred)
            val_loss = F.mse_loss(pred, label)
            self.val_FS_losses.append(val_loss.item())
            #self.log("val_FS_loss", val_loss, on_epoch=True, prog_bar=False, sync_dist=True)
            
        return val_loss

    
class FMdata(MTTdata):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)