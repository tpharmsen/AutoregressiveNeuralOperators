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
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.animation as animation
import os

from dataloaders import *
from dataloaders import DATASET_MAPPER
from dataloaders.utils import get_dataset, ZeroShotSampler, spatial_resample
#from trainers.utils import make_plot, animate_rollout, magnitude_vel, rollout
from trainers.utils import animate_rollout, magnitude_vel, rollout

plt.style.use('dark_background')
plt.rcParams['figure.facecolor'] = '#1F1F1F'
plt.rcParams['axes.facecolor'] = '#1F1F1F'
plt.rcParams['savefig.facecolor'] = '#1F1F1F'

class STT:
    def __init__(self, cb, cd, cm, ct):
        self.cb = cb
        self.cd = cd
        self.cm = cm
        self.ct = ct
        self.device = torch.device(self.cb.device)
        self.out_0 = self.cb.save_path + self.cb.folder_out + self.ct.plottrain_out
        self.out_1 = self.cb.save_path + self.cb.folder_out + self.ct.plotval_out
        self.out_2 = self.cb.save_path + self.cb.folder_out + self.ct.plotvalf_out
        self.out_3 = self.cb.save_path + self.cb.folder_out + self.ct.anim_out
        self.out_4 = self.cb.save_path + self.cb.folder_out + self.ct.checkpoint
        #os.makedirs(self.cb.save_path + self.cb.folder_out, exist_ok=True)            

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

    def _initialize_model(self):
        if self.cm.model_name == "swinUnet":
            from modelComp.swinUnet import SwinUnet, ConvNeXtBlock, ResNetBlock
            self.model = SwinUnet(emb_dim=96,
                            data_dim=[self.ct.batch_size, self.cm.in_channels, self.cd.resample_shape, self.cd.resample_shape],
                            patch_size=(self.cm.patch_size, self.cm.patch_size),
                            hiddenout_dim=self.cm.hiddenout_dim,
                            depth=self.cm.depth,
                            stage_depths=self.cm.stage_depths,
                            num_heads=self.cm.num_heads,
                            window_size=self.cm.window_size,
                            use_flex_attn=self.cm.use_flex_attn,
                            act=nn.GELU,
                            skip_connect=ConvNeXtBlock
                            ).to(self.device)
        else:
            raise ValueError('MODEL NOT RECOGNIZED')
        
        self.cm.nparams = self.nparams(self.model)
        #print(self.ct.init_lr, self.ct.weight_decay)
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.ct.init_lr, weight_decay=self.ct.weight_decay)
        self.criterion = nn.MSELoss()
        self.scheduler = ReduceLROnPlateau(self.optimizer, mode='min', factor=0.1, patience=self.ct.patience, min_lr=1e-7)

    def nparams(self, model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)

    def prepare_dataloader(self):
        train_datasets = []
        self.val_datasets = []
        self.val_samplers = []
        self.val_forward_datasets = []

        for item in self.cd.datasets:
            dataset = get_dataset(
                dataset_obj=DATASET_MAPPER[item['dataset']], 
                folderPath=str(self.cb.data_base + item["path"]), 
                file_ext=item["file_ext"], 
                resample_shape=self.cd.resample_shape, 
                resample_mode=self.cd.resample_mode, 
                timesample=item["timesample"],
                n = 1
            )
            dataset.name = item['name']

            train_sampler = ZeroShotSampler(dataset, train_ratio=self.ct.train_ratio, split="train", n=1)
            val_sampler = ZeroShotSampler(dataset, train_ratio=self.ct.train_ratio, split="val", n=1)
            val_forward_sampler = ZeroShotSampler(dataset, train_ratio=self.ct.train_ratio, split="val", n=self.ct.forward_steps_loss)

            train_datasets.append(Subset(dataset, train_sampler.indices))
            self.val_datasets.append(Subset(dataset, val_sampler.indices))
            self.val_samplers.append(val_sampler)
            self.val_forward_datasets.append(Subset(dataset, val_forward_sampler.indices))
            #print(len(train_sampler), len(val_sampler), len(val_forward_sampler))
        
        train_dataset = ConcatNormDataset(train_datasets)
        self.val_dataset = ConcatNormDataset(self.val_datasets)
        self.val_forward_dataset = ConcatNormDataset(self.val_forward_datasets)

        #unnorm_trainset = train_dataset.normalize_velocity()
        #unnorm_valset = val_dataset.normalize_velocity()
        if self.ct.normalize:
            unnorm_dataset = train_dataset.normalize_velocity()
            self.ct.norm_factor = unnorm_dataset.item()
            #print('datasets norm factor:', unnorm_dataset.item())
        else:
            self.ct.norm_factor = None

        train_loader = DataLoader(
            train_dataset,
            batch_size=self.ct.batch_size,
            shuffle=True,
            pin_memory=self.ct.pin_memory)
        val_loader = DataLoader(
            self.val_dataset,
            batch_size=self.ct.batch_size,
            shuffle=True, # simply set to true to provide random sampling for image plotting function
            pin_memory=self.ct.pin_memory) 
        val_forward_loader = DataLoader(
            self.val_forward_dataset,
            batch_size=self.ct.batch_size,
            shuffle=True,
            pin_memory=self.ct.pin_memory
        )
        return train_loader, val_loader, val_forward_loader
            
    def train_one_epoch(self):
        losses = []
        self.model.train()

        for idx, (front, label) in enumerate(self.train_loader):
            #print('shapes:', front.shape, torch.max(front), torch.min(front))
            #print(f"{idx/len(self.train_loader):2f}", end='\r')
            front, label = front.to(self.device), label.to(self.device)
            self.optimizer.zero_grad()
            pred = self.model(front)
            loss = self.criterion(pred, label)
            loss.backward()
            self.optimizer.step()
            losses.append(loss.detach().item())

        return np.mean(losses)

    def validate(self):
        loss_ts = self._validate_timestep()
        loss_fs = self._validate_forward()
        return loss_ts, loss_fs

    def _validate_timestep(self):
        losses = []
        self.model.eval()

        with torch.no_grad():
            for idx, (front, label) in enumerate(self.val_loader):
                front, label = front.to(self.device), label.to(self.device)
                pred = self.model(front)
                loss = self.criterion(pred, label)
                losses.append(loss.detach().item())
        return np.mean(losses)
    
    def _validate_forward(self):
        losses = []
        self.model.eval()

        for subset in self.val_forward_dataset.datasets:
            subset.dataset.fs = self.ct.forward_steps_loss

        with torch.no_grad():
            for idx, (front, label) in enumerate(self.val_forward_loader):
                pred = front
                pred, label = pred.to(self.device), label.to(self.device)
                for i in range(self.ct.forward_steps_loss):
                    pred = self.model(pred)
                loss = self.criterion(pred, label)
                losses.append(loss.detach().item())

        for subset in self.val_forward_dataset.datasets:
            subset.dataset.fs = 1
        return np.mean(losses)

    
    def train(self):
        print("initialising model...", end='\r')
        self._initialize_model()
        print("setting up dataflow...", end='\r')
        self.train_loader, self.val_loader, self.val_forward_loader = self.prepare_dataloader()
        
        if self.cb.wandb:
            print("booting up wandb...", end='\r')
            wandb_config = self.build_wandb_config()
            wandb.init(project="FluidGPT", name=self.cb.wandb_name, config=wandb_config)    
            #wandb.config.update(self.config)

        best_loss = np.inf#float('inf')
        print('front/label trainpairs:', self.ct.batch_size * len(self.train_loader), "- nparams model:", self.cm.nparams, "- norm factor:", self.ct.norm_factor)
        for self.epoch in range(self.ct.epochs):
            start_time = time.time()
            train_loss = self.train_one_epoch()
            val_loss_SS, val_loss_FS = self.validate()
            epoch_time = time.time() - start_time
            makeviz = self.epoch % self.cb.viz_freq == 0
            if makeviz and self.cb.viz:
                plot_time = time.time()
                self.make_plot(self.out_0, 'train')
                self.make_plot(self.out_1, 'val')
                self.make_plot(self.out_2, 'val_forward')
                self.make_anim(self.out_3)
                plot_time = time.time() - plot_time
            else:
                plot_time = 0

            print(f"Epoch {self.epoch}:  "
                    f"Train Loss = {train_loss:.8f} - "
                    f"Val Loss SS = {val_loss_SS:.8f} - "
                    f"Val Loss FS = {val_loss_FS:.8f} - "
                    f"LR: {self.optimizer.param_groups[0]['lr']:.1e} - "
                    f"ET: {epoch_time:.2f} s - "
                    f"PT: {plot_time:.4f} s")

            if self.cb.wandb:
                wandb.log({
                    "epoch": self.epoch,
                    "Train Loss": train_loss,
                    "Val Loss SS": val_loss_SS,
                    "Val Loss FS": val_loss_FS,
                    "Learning Rate": self.optimizer.param_groups[0]['lr'],
                    "Epoch time": epoch_time,
                    "Plot time": plot_time,
                    "rollout_val_gif": wandb.Video(self.out_3, format="gif") if self.cb.viz and makeviz else None,
                    "val_forward_plot": wandb.Image(self.out_2) if self.cb.viz and makeviz else None,
                    "val_plot": wandb.Image(self.out_1) if self.cb.viz and makeviz else None,
                    "train_plot": wandb.Image(self.out_0) if self.cb.viz and makeviz else None,
                })

            self.scheduler.step(val_loss_SS)

            if self.cb.save_on:
                if val_loss_SS < best_loss:
                    torch.save({
                        'epoch': self.epoch,
                        'model_state_dict': self.model.state_dict(),
                        'optimizer_state_dict': self.optimizer.state_dict(),
                        'train_loss': train_loss,
                        'val_loss_ss': val_loss_SS,
                        'val_loss_fs': val_loss_FS
                    }, self.out_4)
                    best_loss = val_loss_SS

        if self.cb.wandb:
            wandb.finish()
        print('finished')

    def make_anim(self, output_path):
        self.model.eval()
        #print(output_path)
        #print(self.val_datasets)
        dataset_idx = torch.randint(0, len(self.val_datasets), (1,)).item()
        traj_idx = self.val_samplers[dataset_idx].random_val_traj()
        #print(traj_idx)
        #print(np.sort(self.val_samplers[dataset_idx].val_trajs))
        val_traj = self.val_datasets[dataset_idx].dataset.get_single_traj(traj_idx)
        #print(val_traj.shape)
        front = val_traj[0].unsqueeze(0)	
        front = front.to(self.device)
        stacked_pred = rollout(front, self.model, len(val_traj))
        stacked_pred, stacked_true = magnitude_vel(stacked_pred), magnitude_vel(val_traj)
        dataset_name = self.val_datasets[dataset_idx].dataset.name
        animate_rollout(stacked_pred, stacked_true, dataset_name, output_path)
        
    
    def make_plot(self, output_path="output/out.png", mode='val'):
        self.model.eval()

        if mode == 'val':
            for front, label in self.val_loader:
                break           
        elif mode == 'train':
            for front, label in self.train_loader:
                break  
        elif mode == 'val_forward':
            for subset in self.val_forward_dataset.datasets:
                subset.dataset.fs = self.ct.forward_steps_loss
            for front, label in self.val_forward_loader:
                break
            for subset in self.val_forward_dataset.datasets:
                subset.dataset.fs = 1
        else:
            raise ValueError('PLOTMODE NOT RECOGNIZED')
        front, label = front.to(self.device), label.to(self.device)
        front, label = front[0].unsqueeze(0), label[0].unsqueeze(0)

        #print(mode, self.val_dataset.datasets[0].dataset.fs, self.val_forward_dataset.datasets[0].dataset.fs)
        with torch.no_grad():
            pred = self.model(front)
        
        front_x, front_y = front[0, 0].cpu(), front[0, 1].cpu()
        pred_x, pred_y = pred[0, 0].cpu(), pred[0, 1].cpu()
        label_x, label_y = label[0, 0].cpu(), label[0, 1].cpu()
        diff_x, diff_y = (label_x - pred_x).abs(), (label_y - pred_y).abs()

        fig, axes = plt.subplots(2, 4, figsize=(16, 8))
        fig.suptitle(f"Epoch {self.epoch}")

        titles = ["Input", "Prediction", "Target", "Difference"]

        axes[0, 0].imshow(front_x, cmap='viridis')
        axes[0, 1].imshow(pred_x, cmap='viridis')
        axes[0, 2].imshow(label_x, cmap='viridis')
        axes[0, 3].imshow(diff_x, cmap='magma')

        axes[1, 0].imshow(front_y, cmap='viridis')
        axes[1, 1].imshow(pred_y, cmap='viridis')
        axes[1, 2].imshow(label_y, cmap='viridis')
        axes[1, 3].imshow(diff_y, cmap='magma')

        for i in range(4):
            axes[0, i].set_title(titles[i])
            axes[1, i].set_title(titles[i])
        
        axes[0, 0].set_ylabel(r"$v_x$")
        axes[1, 0].set_ylabel(r"$v_y$")
            
        for ax in axes.flat:
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_visible(False)
        
        plt.tight_layout()
        plt.savefig(output_path)
        plt.close()
    
    