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
from trainers.utils import animate_rollout, magnitude_vel, rollout_det, compute_energy_enstrophy_spectra
from modelComp.utils import ACT_MAPPER, SKIPBLOCK_MAPPER

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

class MTTtrainer(L.LightningModule):
    def __init__(self, cb, cd, cm, ct):
        super().__init__()
        self.cb = cb
        self.cd = cd
        self.cm = cm
        self.ct = ct

    def init_modules(self):
        self.modelmodule = MTTmodel(self.cb, self.cd, self.cm, self.ct)
        self.datamodule = MTTdata(self.cb, self.cd, self.cm, self.ct)

    def train(self):
        self.init_modules()
        self.modelmodule._initialize_model()
        
        num_gpus = torch.cuda.device_count()
        print(f"Number of GPUs: {num_gpus} of type {torch.cuda.get_device_name(0)}")
        wandb_logger = WandbLogger(project="FluidGPT", 
                                   config = self.build_wandb_config(), 
                                   name=self.cb.wandb_name, 
                                   save_dir=self.cb.save_path + self.cb.folder_out
                                   )
        
        callbacks = []
        if self.cb.save_on:
            
            self.checkpoint_path = self.cb.save_path + self.cb.folder_out + self.cm.model_name + '/' + str(wandb_logger.experiment.id)
            if not os.path.exists(self.checkpoint_path):
                os.makedirs(self.checkpoint_path)
            if not os.path.exists(self.cb.save_path + self.cb.folder_out):
                os.makedirs(self.cb.save_path + self.cb.folder_out)
            manualCheckpoint = ModelCheckpoint(
                dirpath= self.checkpoint_path,
                filename = "{epoch:04d}-{val_SS_loss_checkpoint:.6f}",
                #filename=r"{epoch:04d}-val_SS_loss_dataloader_idx_0={val_SS_loss/dataloader_idx_0:.4f}",
                monitor="val_SS_loss_checkpoint",#/dataloader_idx_0",
                mode="min",  
                save_top_k=5,           
                every_n_epochs=1,       
                save_weights_only=False,
            )       
            callbacks.append(manualCheckpoint)
        else:
            manualCheckpoint = None

        trainer = L.Trainer(
            precision="bf16-mixed" if platform.system() != "Windows" else "16-mixed",
            accelerator="gpu",
            devices= 'auto',
            logger=wandb_logger,
            strategy = self.ct.strategy if platform.system() != "Windows" else "auto",
            max_epochs=self.ct.epochs,
            num_sanity_val_steps=0, 
            callbacks=callbacks
        )
        if self.cb.save_on:
            print(f"Manual checkpointing and zero-shot split saving at {self.checkpoint_path}")
        else:
            print("No manual checkpointing or zero-shot split saving enabled")
        trainer.fit(self.modelmodule, self.datamodule)

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
    
class MTTmodel(L.LightningModule):
    def __init__(self, cb, cd, cm, ct):
        super().__init__()

        self.cb = cb
        self.cd = cd
        self.cm = cm
        self.ct = ct

        self.base_path = self.cb.save_path + self.cb.folder_out
        self.out_0 = self.base_path + self.ct.plottrain_out
        self.out_1 = self.base_path + self.ct.plotval_out
        self.out_2 = self.base_path + self.ct.plotvalf_out
        self.out_3 = self.base_path + self.ct.anim_out
        self.out_4 = self.base_path + self.ct.spectra_out

        self.train_losses = []
        self.val_SS_losses = []
        self.val_FS_losses = []
        self.epoch_time = None
        self.log_time = None
        
        #self._initialize_model()   
        self.counter = 0

    def _initialize_model(self):
        if self.cm.model_name == "FluidGPT":
            from modelComp.FluidGPT_B import FluidGPT_B
            self.model = FluidGPT_B(emb_dim=self.cm.emb_dim,
                            data_dim=[self.ct.batch_size, self.cm.temporal_bundling, self.cm.in_channels, self.cd.resample_shape, self.cd.resample_shape],
                            patch_size=(self.cm.patch_size, self.cm.patch_size),
                            hiddenout_dim=self.cm.hiddenout_dim,
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

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        front, label = batch

        pred = self(front)
        train_loss = F.mse_loss(pred, label)
        self.train_losses.append(train_loss.item())
        return train_loss

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

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.ct.init_lr,
            weight_decay=self.ct.weight_decay
        )
        
        scheduler = {
            "scheduler": ReduceLROnPlateau(
                optimizer,
                mode='min', 
                factor=0.1,
                patience=self.ct.patience,
                min_lr=1e-7
            ),
            "monitor": "val_SS_loss/dataloader_idx_0",
            "interval": "epoch",
            "frequency": 1
        }
        return {"optimizer": optimizer, "lr_scheduler": scheduler}

    def on_train_epoch_start(self):
        if not self.trainer.sanity_checking:#print(self.device)
            dataloader = self.trainer.datamodule.train_dataloader()
            if isinstance(dataloader.sampler, DistributedSampler):
                dataloader.sampler.set_epoch(self.current_epoch)
            
                self.epoch_time = time.time()

    
    def on_validation_epoch_end(self):
        val_SS_loss = np.mean(self.val_SS_losses)
        self.log("val_SS_loss_checkpoint", val_SS_loss)

        if not self.trainer.sanity_checking and rank_zero_only.rank == 0:
            epoch = self.trainer.current_epoch
            self.epoch_time = time.time() - self.epoch_time
            self.log_time = time.time()

            train_loss = np.mean(self.train_losses)
            #val_SS_loss = np.mean(self.val_SS_losses)
            val_FS_loss = np.mean(self.val_FS_losses)
            #self.log("val_SS_loss_checkpoint", val_SS_loss)

            self.global_mean = self.trainer.datamodule.global_mean
            self.global_std = self.trainer.datamodule.global_std

            '''
            self.log_dict({
                "val_SS_loss": val_SS_loss,
            }, prog_bar=False)
            '''
            
            visuals = self.cb.viz and epoch % self.cb.viz_freq == 0
            if visuals:
                device = 'cuda' if torch.cuda.is_available() else 'cpu'
                self.make_plot(self.out_1, mode='val', device=device)
                self.make_plot(self.out_0, mode='train', device=device)
                self.make_plot(self.out_2, mode='val_forward', device=device)
                stacked_pred, stacked_true, dataset_name = self.random_rollout(device=device)
                self.make_anim(stacked_pred, stacked_true, dataset_name, self.out_3)
                self.spectra_plot(stacked_pred, stacked_true, dataset_name, self.out_4)
            
            self.log_time = time.time() - self.log_time
            self.logger.experiment.log({
                "epoch": epoch,
                "train_loss": train_loss,
                "val_SS_loss": val_SS_loss,
                "val_FS_loss": val_FS_loss,
                "Learning Rate": self.trainer.optimizers[0].param_groups[0]['lr'],
                "Epoch Time": self.epoch_time,
                "train_plot": wandb.Image(self.out_0) if visuals else None,
                "val_plot": wandb.Image(self.out_1) if visuals else None,
                "val_forward_plot": wandb.Image(self.out_2) if visuals else None,
                "val_anim": wandb.Video(self.out_3, format="gif") if visuals else None,
                "val_spectra": wandb.Image(self.out_4) if visuals else None,
                "Log Time": self.log_time
            })

        self.train_losses = []
        self.val_SS_losses = []
        self.val_FS_losses = []

    def spectra_plot(self, stacked_pred, stacked_true, dataset_name, output_path):
        #clock = time.time()
        stacked_pred, stacked_true = stacked_pred.squeeze(0), stacked_true.squeeze(0)
        tmax = min(stacked_pred.shape[0], stacked_true.shape[0])
        t1 = tmax // 4
        tmax = tmax - 1
        #print('maxt:', tmax)
        #print('stacked_pred:', stacked_pred.shape)
        #print('stacked_true:', stacked_true.shape)
        #print()
        #print('\nstarting spectra calculation ', time.time() - clock, '\n')
        kinit, Einit, Zinit = compute_energy_enstrophy_spectra(stacked_true[0,0], stacked_true[0,1], dataset_name, Lx=1.0, Ly=1.0)
        ktrue0, Etrue0, Ztrue0 = compute_energy_enstrophy_spectra(stacked_true[t1,0], stacked_true[t1,1], dataset_name, Lx=1.0, Ly=1.0)
        kpred0, Epred0, Zpred0 = compute_energy_enstrophy_spectra(stacked_pred[t1,0], stacked_pred[t1,1], dataset_name, Lx=1.0, Ly=1.0)
        ktrue1, Etrue1, Ztrue1 = compute_energy_enstrophy_spectra(stacked_true[tmax,0], stacked_true[tmax,1], dataset_name, Lx=1.0, Ly=1.0)
        kpred1, Epred1, Zpred1 = compute_energy_enstrophy_spectra(stacked_pred[tmax,0], stacked_pred[tmax,1], dataset_name, Lx=1.0, Ly=1.0)
        #print('\ncalculated spectra', time.time() - clock, '\n')

        kinit, Einit, Zinit = kinit.cpu().numpy(), Einit.cpu().numpy(), Zinit.cpu().numpy()
        ktrue0, Etrue0, Ztrue0 = ktrue0.cpu().numpy(), Etrue0.cpu().numpy(), Ztrue0.cpu().numpy()
        kpred0, Epred0, Zpred0 = kpred0.cpu().numpy(), Epred0.cpu().numpy(), Zpred0.cpu().numpy()
        ktrue1, Etrue1, Ztrue1 = ktrue1.cpu().numpy(), Etrue1.cpu().numpy(), Ztrue1.cpu().numpy()
        kpred1, Epred1, Zpred1 = kpred1.cpu().numpy(), Epred1.cpu().numpy(), Zpred1.cpu().numpy()

        fig, axs = plt.subplots(1, 2, figsize=(14, 5)) 
        colors = ['#AA0140', '#D1205A', '#08457E', '#2D6FBF']
    
        ref_k = np.array([7,70])
        ref_E_53 = (ref_k / ref_k[0])**(-5/3) * Etrue1[5]
        ref_Z_3 = (ref_k / ref_k[0])**(-3) * Ztrue1[5]
        
        # energy spectrum with ref k^{-5/3}
        axs[0].loglog(kinit, Einit, label='Init', color='gray')
        axs[0].loglog(ktrue0, Etrue0, label=f'True (t={t1})', color=colors[0])
        axs[0].loglog(kpred0, Epred0, label=f'Pred (t={t1})', color=colors[2])
        axs[0].loglog(ktrue1, Etrue1, label=f'True (t={tmax})', color=colors[1])
        axs[0].loglog(kpred1, Epred1, label=f'Pred (t={tmax})', color=colors[3])
        
        axs[0].loglog(ref_k, ref_E_53, 'k--', label=r'$k^{-5/3}$', color='white')
        
        axs[0].set_xlabel(r"Wavenumber $k$ [1/m]")
        axs[0].set_ylabel(r"Energy [$\mathrm{m}^2/\mathrm{s}^2$]")
        axs[0].set_title(r"Energy Spectrum ")
        axs[0].legend()
        axs[0].grid(True, color='gray')
        
        # now same for enstrophy
        axs[1].loglog(kinit, Zinit, label='Init', color='gray')
        axs[1].loglog(ktrue0, Ztrue0, label=f'True (t={t1})', color=colors[0])
        axs[1].loglog(kpred0, Zpred0, label=f'Pred (t={t1})', color=colors[2])
        axs[1].loglog(ktrue1, Ztrue1, label=f'True (t={tmax})', color=colors[1])
        axs[1].loglog(kpred1, Zpred1, label=f'Pred (t={tmax})', color=colors[3])

        axs[1].loglog(ref_k, ref_Z_3, 'k--', label=r'$k^{-3}$', color='white')
        
        axs[1].set_xlabel(r"Wavenumber $k$ [1/m]")
        axs[1].set_ylabel(r"Enstrophy [$\mathrm{s}^{-2}/\mathrm{m}$]")
        axs[1].set_title(r"Enstrophy Spectrum")
        axs[1].legend()
        axs[1].grid(True, color='gray')
        
        plt.suptitle("Isotropic Energy & Enstrophy Spectra with Theoretical Slopes, dataset: " + dataset_name, fontsize=20)
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        #print('\nsaving spectra to:', output_path, '- ', time.time() - clock, '\n')
        plt.savefig(output_path, bbox_inches='tight')
        #print('\nsaved spectra to:', output_path, '- ', time.time() - clock, '\n')
        plt.close()
        

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
            stacked_pred = rollout_det(front, self.model, len(val_traj) // self.cm.temporal_bundling)
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
        
    def make_anim(self, stacked_pred, stacked_true, dataset_name, output_path):
        stacked_pred = magnitude_vel(stacked_pred)
        stacked_true = magnitude_vel(stacked_true)
        #print('stacked_pred:', stacked_pred.shape)
        #print('stacked_true:', stacked_true.shape)
        animate_rollout(stacked_pred.squeeze(0), stacked_true.squeeze(0), dataset_name, output_path)

    def make_plot(self, output_path, mode='val', device='cuda'):
        self.model.eval()

        if mode == 'val':
            dataset_idx = torch.randint(0, len(self.trainer.datamodule.val_datasets), (1,)).item()
            indices = list(self.trainer.datamodule.val_samplers[dataset_idx].indices)
            sample = random.choice(indices)
            front, label = self.trainer.datamodule.val_datasets[dataset_idx].dataset.__getitem__(sample)
            dataset_name = self.trainer.datamodule.val_datasets[dataset_idx].dataset.name
            #loader = iter(self.trainer.datamodule.val_dataloader()[0])
            #for front, label in self.trainer.datamodule.val_dataloader()[0]:
                #front, label = front.to(device), label.to(device)
                #break
        elif mode == 'train':
            dataset_idx = torch.randint(0, len(self.trainer.datamodule.train_datasets), (1,)).item()
            indices = list(self.trainer.datamodule.train_samplers[dataset_idx].indices)
            sample = random.choice(indices)
            front, label = self.trainer.datamodule.train_datasets[dataset_idx].dataset.__getitem__(sample)
            dataset_name = self.trainer.datamodule.train_datasets[dataset_idx].dataset.name
            #for front, label in self.trainer.datamodule.train_dataloader():
                #front, label = front.to(device), label.to(device)
                #break
            #loader = iter(self.trainer.datamodule.train_dataloader())
        elif mode == 'val_forward':
            dataset_idx = torch.randint(0, len(self.trainer.datamodule.val_forward_datasets), (1,)).item()
            indices = list(self.trainer.datamodule.val_forward_samplers[dataset_idx].indices)
            sample = random.choice(indices)
            front, label = self.trainer.datamodule.val_forward_datasets[dataset_idx].dataset.__getitem__(sample)
            dataset_name = self.trainer.datamodule.val_forward_datasets[dataset_idx].dataset.name
            #for front, label in self.trainer.datamodule.val_dataloader()[1]:
                #front, label = front.to(device), label.to(device)
                #break
            #loader = iter(self.trainer.datamodule.val_dataloader()[1])
        else:
            raise ValueError('PLOTMODE NOT RECOGNIZED')

        #front, label = next(loader)
        front, label = front.to(device).unsqueeze(0), label.to(device).unsqueeze(0)
        if self.ct.strategy == "deepspeed":
            front, label = front[0].unsqueeze(0).to(torch.bfloat16), label[0].unsqueeze(0).to(torch.bfloat16)
        else:
            front, label = front[0].unsqueeze(0).float(), label[0].unsqueeze(0).float()  # .to(torch.bfloat16)
        #front, label = front[0].unsqueeze(0).to(torch.bfloat16), label[0].unsqueeze(0).to(torch.bfloat16)
        with torch.no_grad():
            pred = self(front)
            if mode == 'val_forward':
                pred = front
                for _ in range(self.ct.forward_steps_loss):
                    pred = self.model(pred)
            else:
                pred = self(front)
        
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
            #fig.add_subplot(gs[t, 0]).set_ylabel(r"$v_x$", rotation=0, labelpad=20, fontsize=12)
            #fig.add_subplot(gs[t, cols_per_side + spacer]).set_ylabel(r"$v_y$", rotation=0, labelpad=20, fontsize=12)
        fig.text(0.25, 0.92, r"$v_x$", fontsize=20, ha='center')
        fig.text(0.80, 0.92, r"$v_y$", fontsize=20, ha='center')
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        plt.savefig(output_path, bbox_inches='tight')
        plt.close()


class MTTdata(L.LightningDataModule):
    def __init__(self, cb, cd, cm, ct):
        super().__init__()
        self.cb = cb
        self.cd = cd 
        self.cm = cm  
        self.ct = ct 

    def prepare_data(self): 

        for item in self.cd.datasets:
            
            preproc_savepath = str(self.cb.data_base + 'preproc_' + item["name"])

            if not os.path.exists(preproc_savepath):
                print("preprocessing dataset", item["name"] , "at path", preproc_savepath, "...")
                get_dataset(
                    dataset_obj=PREPROC_MAPPER[item['ppclass']],
                    preproc_savepath=preproc_savepath,
                    folderPath=str(self.cb.data_base + item["path"]),
                    file_ext=item["file_ext"],
                    resample_shape=self.cd.resample_shape,
                    resample_mode=self.cd.resample_mode,
                    timesample=item["timesample"],
                    dataset_name=item["name"]
                )
                
            else:
                print("folder", item["name"], "already exists, skipping preproccessing")

        if hasattr(self.cd, 'preproc_only'):
            if self.cd.preproc_only:
                raise ValueError("Preprocessing only mode is enabled, stopping after preprocessing.")
        
        
    def setup(self, stage=None):

        self.train_datasets = []
        self.val_datasets = []
        self.val_forward_datasets = []
        self.train_samplers = []
        self.val_samplers = []
        self.val_forward_samplers = []
        
        means, stds, sizes = [], [], []
                
        for item in self.cd.datasets:
            preproc_savepath = str(self.cb.data_base + 'preproc_' + item["name"])
            dataset_SS = DiskDatasetDiv(preproc_savepath, temporal_bundling=self.cm.temporal_bundling, forward_steps=1)
            dataset_FS = DiskDatasetDiv(preproc_savepath, temporal_bundling=self.cm.temporal_bundling, forward_steps=self.ct.forward_steps_loss)

            # generate random seed
            random_seed = random.randint(0, 10000)
            train_sampler = ZeroShotSampler(dataset_SS, train_ratio=self.ct.train_ratio, split="train", seed=random_seed, forward_steps=1)
            val_sampler = ZeroShotSampler(dataset_SS, train_ratio=self.ct.train_ratio, split="val", seed=random_seed, forward_steps=1)
            val_forward_sampler = ZeroShotSampler(dataset_FS, train_ratio=self.ct.train_ratio, split="val", seed=random_seed, forward_steps=self.ct.forward_steps_loss)

            self.train_datasets.append(Subset(dataset_SS, train_sampler.indices))
            self.val_datasets.append(Subset(dataset_SS, val_sampler.indices))
            self.val_forward_datasets.append(Subset(dataset_FS, val_forward_sampler.indices))
            self.train_samplers.append(train_sampler)
            self.val_samplers.append(val_sampler)
            self.val_forward_samplers.append(val_forward_sampler)
            
            
            split = {
                "name": item["name"],
                "seed": random_seed,
                "train_trajs": train_sampler.train_trajs,
                "val_trajs": val_sampler.val_trajs,
                "val_forward_trajs": val_forward_sampler.val_trajs,
                "train_idxs": train_sampler.indices,
                "val_idxs": val_sampler.indices,
                "val_forward_idxs": val_forward_sampler.indices,
            }
            if self.cb.save_on:
                for callback in self.trainer.callbacks:
                    if isinstance(callback, ModelCheckpoint):
                        save_split_path = os.path.join(callback.dirpath, "traj_split_" + item["name"] + ".json")
                if save_split_path is None:
                    raise ValueError("ModelCheckpoint callback not found, unable to save trajectory split.")
                with open(save_split_path, "w") as f:
                    json.dump(split, f, indent=0)
                if wandb.run is not None:
                    wandb.save(save_split_path)
            
            mean_i = dataset_SS.avg
            std_i = dataset_SS.std
            N_i = np.prod(dataset_SS.datashape)

            means.append(mean_i)
            stds.append(std_i)
            sizes.append(N_i)
            print("dataset", item["name"], "loaded,", np.prod(dataset_SS.datashape), "elements")

        means = np.array(means)
        stds = np.array(stds)
        sizes = np.array(sizes)

        self.global_mean = np.sum(sizes * means) / np.sum(sizes)
        self.global_std = np.sqrt(np.sum(sizes * (stds**2 + (means - self.global_mean)**2)) / np.sum(sizes))
        if self.ct.normalize:
            for dataset_list in [self.train_datasets, self.val_datasets, self.val_forward_datasets]:
                for subset in dataset_list:
                    subset.dataset.avgnorm = self.global_mean
                    subset.dataset.stdnorm = self.global_std
                
        self.train_dataset = ConcatDataset(self.train_datasets)
        self.val_dataset = ConcatDataset(self.val_datasets)
        self.val_forward_dataset = ConcatDataset(self.val_forward_datasets)
        print("datasets ready")

    def create_sampler(self, dataset, shuffle):
        if dist.is_available() and dist.is_initialized():
            return DistributedSampler(
                dataset,
                shuffle=shuffle,
                drop_last=False,
                rank=dist.get_rank(),
                num_replicas=dist.get_world_size()
            )
        return None

    def train_dataloader(self):
        train_sampler = self.create_sampler(self.train_dataset, shuffle=True)
        
        return DataLoader(
            self.train_dataset,
            batch_size=self.ct.batch_size,
            shuffle=(train_sampler is None),
            sampler=train_sampler,
            pin_memory=self.ct.pin_memory, 
            num_workers=self.ct.num_workers,
            persistent_workers=self.ct.persistent_workers if self.ct.num_workers > 0 else False,
            prefetch_factor=self.ct.prefetch_factor if self.ct.num_workers > 0 else None
        )

    def val_dataloader(self):
        val_SS_sampler = self.create_sampler(self.val_dataset, shuffle=True)
        val_FS_sampler = self.create_sampler(self.val_forward_dataset, shuffle=True)

        val_SS_loader = DataLoader(
            self.val_dataset,
            batch_size=self.ct.batch_size,
            shuffle=(val_SS_sampler is None),
            sampler=val_SS_sampler, 
            pin_memory=self.ct.pin_memory, 
            num_workers=self.ct.num_workers,
            persistent_workers=self.ct.persistent_workers if self.ct.num_workers > 0 else False,
            prefetch_factor=self.ct.prefetch_factor if self.ct.num_workers > 0 else None
        )
        val_FS_loader = DataLoader(
            self.val_forward_dataset,
            batch_size=self.ct.batch_size,
            shuffle=(val_FS_sampler is None),
            sampler=val_FS_sampler,
            pin_memory=self.ct.pin_memory, 
            num_workers=self.ct.num_workers,
            persistent_workers=self.ct.persistent_workers if self.ct.num_workers > 0 else False,
            prefetch_factor=self.ct.prefetch_factor if self.ct.num_workers > 0 else None
        )
        return [val_SS_loader, val_FS_loader]


