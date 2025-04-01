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

from dataloaders import *
from dataloaders import DATASET_MAPPER
from dataloaders.utils import get_dataset, ZeroShotSampler, spatial_resample

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
        val_datasets = []

        for item in self.cd.datasets:
            dataset = get_dataset(
                dataset_obj=DATASET_MAPPER[item['dataset']], 
                folderPath=str(self.cb.data_base + item["path"]), 
                file_ext=item["file_ext"], 
                resample_shape=item["resample_shape"], 
                resample_mode=item["resample_mode"], 
                timesample=item["timesample"]
            )

            train_sampler = ZeroShotSampler(dataset, train_ratio=self.ct.train_ratio, split="train")
            val_sampler = ZeroShotSampler(dataset, train_ratio=self.ct.train_ratio, split="val")

            #train_datasets.append((dataset, train_sampler))
            #val_datasets.append((dataset, val_sampler))

            train_sampler = ZeroShotSampler(dataset, train_ratio=self.ct.train_ratio, split="train")
            val_sampler = ZeroShotSampler(dataset, train_ratio=self.ct.train_ratio, split="val")
            #print(len(train_sampler.indices))
            train_datasets.append(Subset(dataset, train_sampler.indices))
            val_datasets.append(Subset(dataset, val_sampler.indices))
        
        train_loader = DataLoader(
            ConcatDataset(train_datasets),
            batch_size=self.ct.batch_size,
            shuffle=True)
        val_loader = DataLoader(
            ConcatDataset(val_datasets),
            batch_size=self.ct.batch_size,
            shuffle=True) # simply set to true to provide random sampling for image plotting function

        #train_loader = DataLoader(
        #    ConcatDataset([dataset for dataset, _ in train_datasets]),
        #    batch_size=self.ct.batch_size,
        #    sampler=ConcatDataset([sampler for _, sampler in train_datasets])
        #)

        #val_loader = DataLoader(
        #    ConcatDataset([d for d, _ in val_datasets]),
        #    batch_size=self.ct.batch_size,
        #    sampler=ConcatDataset([s for _, s in val_datasets])
        #)

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
        for self.epoch in range(self.ct.epochs):
            start_time = time.time()
            #print('epoch:', self.epoch)
            train_loss = self.train_one_epoch()
            val_loss = self._validate_timestep()
            epoch_time = time.time() - start_time
            self.make_plot(output_path='output/test_train.png', on_val=False)
            self.make_plot(output_path='output/test_val.png', on_val=True)
            plot_time = time.time() - epoch_time
            print(f"Epoch {self.epoch}:  "
                    f"Train Loss = {train_loss:.8f} - "
                    f"Val Loss = {val_loss:.8f} - "
                    f"LR: {self.optimizer.param_groups[0]['lr']:.1e} - "
                    f"ET: {epoch_time:.2f} s - "
                    f"PT: {plot_time:.2f} s")
            self.scheduler.step(val_loss)
        print('finished')

    def make_plot(self, output_path, on_val=True):
        self.model.eval()

        if on_val:
            for front, label in self.val_loader:
                break           
        else:
            for front, label in self.train_loader:
                break  
        front, label = front.to(self.device), label.to(self.device)
        front, label = front[0].unsqueeze(0), label[0].unsqueeze(0)

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
        
        axes[0, 0].set_ylabel("X")
        axes[1, 0].set_ylabel("Y")
            
        for ax in axes.flat:
            ax.set_xticks([])
            ax.set_yticks([])

        plt.tight_layout()
        plt.savefig(output_path)
        plt.close()
