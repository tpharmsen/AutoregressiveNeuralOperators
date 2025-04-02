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

from dataloaders import *
from dataloaders import DATASET_MAPPER
from dataloaders.utils import get_dataset, ZeroShotSampler, spatial_resample
from trainers.utils import make_plot, animate_rollout, magnitude_vel, rollout

class STT:
    def __init__(self, cb, cd, cm, ct, cv):
        self.cb = cb
        self.cd = cd
        self.cm = cm
        self.ct = ct
        self.cv = cv
        self.device = torch.device(self.cb.device)

    def _initialize_model(self):
        if self.cm.model_name == "swinUnet":
            from modelComp.swinUnet import SwinUnet, ConvNeXtBlock, ResNetBlock
            self.model = SwinUnet(emb_dim=96,
                            data_dim=[self.ct.batch_size, self.cm.in_channels, 256, 256],
                            patch_size=(8,8),
                            hiddenout_dim=256,
                            depth=2,
                            stage_depths=[2, 2, 6, 2, 2],
                            num_heads=[3, 6, 12, 6, 3],
                            window_size=4,
                            use_flex_attn=True, # fix device
                            act=nn.GELU,
                            skip_connect=ConvNeXtBlock).to(self.device)
        else:
            raise ValueError('MODEL NOT RECOGNIZED')
        
        print('Amount of parameters in model:', self.nparams(self.model))
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

        for item in self.cd.datasets:
            dataset = get_dataset(
                dataset_obj=DATASET_MAPPER[item['dataset']], 
                folderPath=str(self.cb.data_base + item["path"]), 
                file_ext=item["file_ext"], 
                resample_shape=item["resample_shape"], 
                resample_mode=item["resample_mode"], 
                timesample=item["timesample"]
            )
            dataset.name = item['name']

            train_sampler = ZeroShotSampler(dataset, train_ratio=self.ct.train_ratio, split="train")
            val_sampler = ZeroShotSampler(dataset, train_ratio=self.ct.train_ratio, split="val")

            #print(len(train_sampler.indices))
            train_datasets.append(Subset(dataset, train_sampler.indices))
            self.val_datasets.append(Subset(dataset, val_sampler.indices))
            self.val_samplers.append(val_sampler)
        
        train_loader = DataLoader(
            ConcatDataset(train_datasets),
            batch_size=self.ct.batch_size,
            shuffle=True)
        val_loader = DataLoader(
            ConcatDataset(self.val_datasets),
            batch_size=self.ct.batch_size,
            shuffle=True) # simply set to true to provide random sampling for image plotting function

        return train_loader, val_loader
            
    def train_one_epoch(self):
        losses = []
        self.model.train()

        for idx, (front, label) in enumerate(self.train_loader):
            front, label = front.to(self.device), label.to(self.device)
            self.optimizer.zero_grad()
            pred = self.model(front)
            loss = self.criterion(pred, label)
            loss.backward()
            self.optimizer.step()
            losses.append(loss.detach().item())

        return np.mean(losses)

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

    def train(self):
        #self.model = self._initialize_model()
        self._initialize_model()
        self.train_loader, self.val_loader = self.prepare_dataloader()
        print('test')
        self.make_anim()
        print('test done')
        for self.epoch in range(self.ct.epochs):
            start_time = time.time()
            #print('epoch:', self.epoch)
            train_loss = self.train_one_epoch()
            val_loss = self._validate_timestep()
            epoch_time = time.time() - start_time
            make_plot(output_path='output/test_train.png', on_val=False)
            make_plot(output_path='output/test_val.png', on_val=True)
            
            plot_time = time.time() - epoch_time
            print(f"Epoch {self.epoch}:  "
                    f"Train Loss = {train_loss:.8f} - "
                    f"Val Loss = {val_loss:.8f} - "
                    f"LR: {self.optimizer.param_groups[0]['lr']:.1e} - "
                    f"ET: {epoch_time:.2f} s - "
                    f"PT: {plot_time:.2f} s")
            self.scheduler.step(val_loss)
        print('finished')

    def make_anim(self):
        output_path = self.cv.anim_out
        print(self.val_datasets)
        dataset_idx = torch.randint(0, len(self.val_datasets), (1,)).item()
        traj_idx = self.val_samplers[dataset_idx].random_val_traj()
        val_traj = self.val_datasets[dataset_idx].get_full_traj(traj_idx)
        print(val_traj.shape)
        front = val_traj[0].unsqueeze(0)	
        stacked_pred = rollout(front)
        stacked_pred, stacked_true = magnitude_vel(stacked_pred), magnitude_vel(val_traj)
        anim
