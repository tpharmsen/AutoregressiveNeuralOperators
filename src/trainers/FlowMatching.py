import lightning as pl
from lightning.pytorch.loggers import WandbLogger
from torch.utils.data.distributed import DistributedSampler
import torch.distributed as dist
from lightning.pytorch.utilities.rank_zero import rank_zero_only
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

from .MTT import MTT, MTTmodel, MTTdata

from dataloaders import *
from dataloaders import PREPROC_MAPPER
from dataloaders.utils import get_dataset, ZeroShotSampler, spatial_resample
#from trainers.utils import make_plot, animate_rollout, magnitude_vel, rollout
from trainers.utils import animate_rollout, magnitude_vel, rollout, compute_energy_enstrophy_spectra
from modelComp.utils import ACT_MAPPER, SKIPBLOCK_MAPPER

plt.style.use('dark_background')
plt.rcParams['figure.facecolor'] = '#1F1F1F'
plt.rcParams['axes.facecolor'] = '#1F1F1F'
plt.rcParams['savefig.facecolor'] = '#1F1F1F'



class FlowMatching(MTT):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    
    def train(self):
        model = FMmodel(self.cb, self.cd, self.cm, self.ct)
        datamodule = FMdata(self.cb, self.cd, self.cm, self.ct)

        num_gpus = torch.cuda.device_count()
        print(f"Number of GPUs available: {num_gpus}")
        print(torch.cuda.get_device_name(0))
        wandb_logger = WandbLogger(project="FluidGPT", config = self.build_wandb_config(), name=self.cb.wandb_name, save_dir=self.cb.save_path + self.cb.folder_out)
        trainer = pl.Trainer(
            precision="bf16-mixed",
            accelerator="gpu",
            devices= 'auto',
            logger=wandb_logger,
            strategy=self.ct.strategy,
            max_epochs=self.ct.epochs,
            num_sanity_val_steps=0
        )
        
        trainer.fit(model, datamodule)

    def build_wandb_config(self):
        wandb_config = {}
        configs = {
        'cb': self.cb,
        'cd': self.cd,
        'cm': self.cm,
        'ct': self.ct
        }
        def flatten_dict(d, parent_key=''):
            items = []
            for k, v in d.items():
                new_key = f"{parent_key}.{k}" if parent_key else k
                if isinstance(v, dict):
                    items.extend(flatten_dict(v, new_key).items())
                else:
                    items.append((new_key, v))
            return dict(items)

        for prefix, config in configs.items():
            flat_config = flatten_dict(config)
            for key, value in flat_config.items():
                wandb_config[f"{prefix}.{key}"] = value

        return wandb_config



class FMmodel(MTTmodel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def training_step(self, batch, batch_idx):
        #print('just a test\n')
        front, label = batch
        # add noise to the batch
        #noise = torch.randn_like(front) * 0.05
        fmsteps = 100
        for i in range(fmsteps):
            x1 = label
            x0 = torch.randn_like(front)
            target = x1 - x0
            t = torch.rand(x1.size(0))
            xt = (1 - t[:, None]) * x0 + t[:, None] * x1
            
        front = torch.randn_like(front) * 1
        pred = self(front)
        train_loss = F.mse_loss(pred, label)
        self.train_losses.append(train_loss.item())
        return train_loss
    
class FMdata(MTTdata):
    def __call__(self, *args, **kwds):
        return super().__call__(*args, **kwds)
