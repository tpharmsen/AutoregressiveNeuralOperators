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
from trainers.utils import animate_rollout, magnitude_vel, compute_energy_enstrophy_spectra
from modelComp.utils import ACT_MAPPER, SKIPBLOCK_MAPPER

from trainers.MTT import MTTtrainer, MTTmodel, MTTdata
from trainers.utils import rollout_prb

class FMTtrainer(MTTtrainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        print('Initializing FlowMatching Trainer')
    
    def init_modules(self):
        print('Initializing FlowMatching Modules')
        self.modelmodule = FMTmodel(self.cb, self.cd, self.cm, self.ct)
        self.datamodule = FMTdata(self.cb, self.cd, self.cm, self.ct)

class FMTmodel(MTTmodel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.automatic_optimization = False
    
    def _initialize_model(self):
        print()
        print(self.cm.model_name)
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
        
    def forward(self, x, t):
        return self.model(x, t)
        
    def random_fft_perturb(self, x, perturbation_strength):
        x_fft = torch.fft.fftshift(torch.fft.fft2(x, dim=(-2, -1)))
        x_fft = x_fft * torch.exp(1j * torch.randn_like(x_fft) * perturbation_strength)
        x_recon = torch.fft.ifft2(torch.fft.ifftshift(x_fft)).real
        return x_recon

    def training_step(self, batch, batch_idx):
        opt = self.optimizers()
        

        front, label = batch
        total_loss = 0.0

        for _ in range(self.ct.train_steps_per_batch):
            opt.zero_grad()
            xnoise = self.random_fft_perturb(front, self.ct.perturbation_strength)
            target = label - xnoise
            t = torch.rand(label.size(0), device=label.device)
            xt = (1 - t[:, None, None, None, None]) * xnoise + t[:, None, None, None, None] * label
            pred = self(xt, t)

            train_loss = F.mse_loss(pred, target, reduction='mean')
            self.train_losses.append(train_loss.item())
            # not sure if correct loss is returned
            self.manual_backward(train_loss)
            opt.step()
            total_loss += train_loss

        avg_loss = total_loss / self.ct.train_steps_per_batch
        return avg_loss
        #return train_loss

    def validation_step(self, batch, batch_idx, dataloader_idx):

        front, label = batch

        if dataloader_idx == 0:
            xprior = self.random_fft_perturb(front, self.ct.perturbation_strength)
            t = torch.rand(label.size(0), device=label.device)
            xt = (1 - t[:, None, None, None, None]) * xprior + t[:, None, None, None, None] * label
            pred = self(xt, t)
            val_loss = F.mse_loss(pred, label, reduction='mean')
        #elif dataloader_idx == 1: # no forward step loss in flowmatching training
            
        """
        steps = self.ct.int_steps
        if dataloader_idx == 0:
            xt = self.random_fft_perturb(front, self.ct.perturbation_strength)
            for i, t in enumerate(torch.linspace(0, 1, steps), start=1):
                pred = self(xt, t.to(label.device).expand(xt.size(0)))
                xt = xt + (1 / steps) * pred
            val_loss = F.mse_loss(pred, label, reduction='mean')
            self.val_SS_losses.append(val_loss.item())
            self.log("val_SS_loss", val_loss, on_epoch=True, prog_bar=False, sync_dist=True)
        elif dataloader_idx == 1:
            # a few forward steps for the forward step loss
            xt = self.random_fft_perturb(front, self.ct.perturbation_strength)
            for _ in range(self.ct.forward_steps_loss):
                for i, t in enumerate(torch.linspace(0, 1, steps), start=1):
                    pred = self(xt, t.to(label.device).expand(xt.size(0)))
                    xt = xt + (1 / steps) * pred
            val_loss = F.mse_loss(xt, label, reduction='mean')
            self.val_FS_losses.append(val_loss.item())
            #self.log("val_FS_loss", val_loss, on_epoch=True, prog_bar=False, sync_dist=True)
        """
        return val_loss


    def random_rollout(self, device='cuda'):
        self.model.eval()
        with torch.no_grad():
            dataset_idx = torch.randint(0, len(self.trainer.datamodule.val_datasets), (1,)).item()
            traj_idx = self.trainer.datamodule.val_samplers[dataset_idx].random_val_traj()
            val_traj = self.trainer.datamodule.val_datasets[dataset_idx].dataset.get_single_traj(traj_idx)
            #print('val_traj:', val_traj.shape)
            if self.ct.strategy == "deepspeed":
                front = val_traj[:self.cm.temporal_bundling].unsqueeze(0).to(device).to(torch.bfloat16)
            else:
                front = val_traj[:self.cm.temporal_bundling].unsqueeze(0).float().to(device)#.to(torch.bfloat16)
            #print('len:', len(val_traj) // self.cm.temporal_bundling)
            stacked_pred = rollout_prb(front, self.model, len(val_traj) // self.cm.temporal_bundling, 
                                       self.random_fft_perturb, self.ct.perturbation_strength,
                                       self.ct.int_steps)
            stacked_pred = stacked_pred.float() #.to(torch.bfloat16) 
            #print('stacked_pred:', stacked_pred.shape)
            stacked_true = val_traj.unsqueeze(0).float()
            #print('stacked_true:', stacked_true.shape)
            dataset_name = str(self.trainer.datamodule.val_datasets[dataset_idx].dataset.name)
            #print('dataset_name:', dataset_name)
            stacked_pred, stacked_true = stacked_pred * self.global_std + self.global_mean, stacked_true * self.global_std + self.global_mean
            #print('stacked_pred:', stacked_pred.shape)
            #print('stacked_true:', stacked_true.shape)
            return stacked_pred, stacked_true, dataset_name
        
    def make_plot(self, output_path, mode='val', device='cuda'):
        self.model.eval()

        if mode == 'val':
            dataset_idx = torch.randint(0, len(self.trainer.datamodule.val_datasets), (1,)).item()
            indices = list(self.trainer.datamodule.val_samplers[dataset_idx].indices)
            sample = random.choice(indices)
            front, label = self.trainer.datamodule.val_datasets[dataset_idx].dataset.__getitem__(sample)
            dataset_name = self.trainer.datamodule.val_datasets[dataset_idx].dataset.name
        elif mode == 'train':
            dataset_idx = torch.randint(0, len(self.trainer.datamodule.train_datasets), (1,)).item()
            indices = list(self.trainer.datamodule.train_samplers[dataset_idx].indices)
            sample = random.choice(indices)
            front, label = self.trainer.datamodule.train_datasets[dataset_idx].dataset.__getitem__(sample)
            dataset_name = self.trainer.datamodule.train_datasets[dataset_idx].dataset.name
        elif mode == 'val_forward':
            dataset_idx = torch.randint(0, len(self.trainer.datamodule.val_forward_datasets), (1,)).item()
            indices = list(self.trainer.datamodule.val_forward_samplers[dataset_idx].indices)
            sample = random.choice(indices)
            front, label = self.trainer.datamodule.val_forward_datasets[dataset_idx].dataset.__getitem__(sample)
            dataset_name = self.trainer.datamodule.val_forward_datasets[dataset_idx].dataset.name
        else:
            raise ValueError('PLOTMODE NOT RECOGNIZED')

        front, label = front.to(device).unsqueeze(0), label.to(device).unsqueeze(0)
        if self.ct.strategy == "deepspeed":
            front, label = front[0].unsqueeze(0).to(torch.bfloat16), label[0].unsqueeze(0).to(torch.bfloat16)
        else:
            front, label = front[0].unsqueeze(0).float(), label[0].unsqueeze(0).float()  # .to(torch.bfloat16)
        #front, label = front[0].unsqueeze(0).to(torch.bfloat16), label[0].unsqueeze(0).to(torch.bfloat16)
        with torch.no_grad():
            steps = self.ct.int_steps
            xt = self.random_fft_perturb(front, perturbation_strength=self.ct.perturbation_strength)
            if mode == 'val_forward':
                for _ in range(self.ct.forward_steps_loss):
                    for i, t in enumerate(torch.linspace(0, 1, steps), start=1):
                        pred = self(xt, t.to(label.device).expand(xt.size(0)))
                        xt = xt + (1 / steps) * pred
            else:
                for i, t in enumerate(torch.linspace(0, 1, steps), start=1):
                    pred = self(xt, t.to(label.device).expand(xt.size(0)))
                    xt = xt + (1 / steps) * pred

        
        front = front.float() * self.global_std + self.global_mean #.to(torch.bfloat16)
        pred = pred.float() * self.global_std + self.global_mean #.to(torch.bfloat16)
        label = label.float() * self.global_std + self.global_mean #.to(torch.bfloat16)

        front_x, front_y = front[0, :, 0].cpu(), front[0, :, 1].cpu()
        pred_x, pred_y = pred[0, :, 0].cpu(), pred[0, :, 1].cpu()
        label_x, label_y = label[0, :, 0].cpu(), label[0, :, 1].cpu()
        diff_x, diff_y = (label_x - pred_x).abs(), (label_y - pred_y).abs()

        tb = self.cm.temporal_bundling
        cols_per_side = 4
        spacer = 1
        total_cols = cols_per_side * 2 + spacer  # 4 + 1 + 4 = 9

        fig = plt.figure(figsize=(3 * total_cols, 4 * tb))
        fig.suptitle(f"Epoch {self.trainer.current_epoch} on dataset {dataset_name}", fontsize=20)
        gs = gridspec.GridSpec(tb, total_cols, wspace=0.1, hspace=0.1)

        titles = ["Front", "Pred", "True", "Diff"]

        for t in range(tb):
            for i, (img_x, img_y) in enumerate(zip(
                [front_x[t], pred_x[t], label_x[t], diff_x[t]],
                [front_y[t], pred_y[t], label_y[t], diff_y[t]]
            )):
                ax_x = fig.add_subplot(gs[t, i])
                ax_y = fig.add_subplot(gs[t, i + cols_per_side + spacer])

                ax_x.imshow(img_x, cmap='viridis' if i < 3 else 'magma')
                ax_y.imshow(img_y, cmap='viridis' if i < 3 else 'magma')

                ax_x.set_xticks([]); ax_x.set_yticks([])
                ax_y.set_xticks([]); ax_y.set_yticks([])
                for ax in (ax_x, ax_y):
                    for spine in ax.spines.values():
                        spine.set_visible(False)

                if t == 0:
                    ax_x.set_title(titles[i], fontsize=14)
                    ax_y.set_title(titles[i], fontsize=14)
        fig.text(0.25, 0.92, r"$v_x$", fontsize=20, ha='center')
        fig.text(0.80, 0.92, r"$v_y$", fontsize=20, ha='center')
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        plt.savefig(output_path, bbox_inches='tight')
        plt.close()

    
class FMTdata(MTTdata):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

