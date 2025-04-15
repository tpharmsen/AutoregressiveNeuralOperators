import pytorch_lightning as pl
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
from dataloaders import READER_MAPPER, DATASET_MAPPER
from dataloaders.utils import get_dataset, ZeroShotSampler, spatial_resample
#from trainers.utils import make_plot, animate_rollout, magnitude_vel, rollout
from trainers.utils import animate_rollout, magnitude_vel, rollout

plt.style.use('dark_background')
plt.rcParams['figure.facecolor'] = '#1F1F1F'
plt.rcParams['axes.facecolor'] = '#1F1F1F'
plt.rcParams['savefig.facecolor'] = '#1F1F1F'

#torch.set_float32_matmul_precision('medium')

class MTT:
    def __init__(self, cb, cd, cm, ct):
        self.cb = cb
        self.cd = cd
        self.cm = cm
        self.ct = ct

        #print('init\n')
    def train(self):
        model = MTTmodel(self.cb, self.cd, self.cm, self.ct)
        datamodule = MTTdata(self.cb, self.cd, self.cm, self.ct)
        trainer = pl.Trainer(
            precision="bf16-mixed",
            accelerator="gpu",
            #devices=2,
            strategy="ddp"
        )
        trainer.fit(model, datamodule)


class MTTmodel(pl.LightningModule):
    def __init__(self, cb, cd, cm, ct):
        super().__init__()

        self.cb = cb
        self.cd = cd
        self.cm = cm
        self.ct = ct
        self.out_0 = self.cb.save_path + self.cb.folder_out + self.ct.plottrain_out
        self.out_1 = self.cb.save_path + self.cb.folder_out + self.ct.plotval_out
        self.out_2 = self.cb.save_path + self.cb.folder_out + self.ct.plotvalf_out
        self.out_3 = self.cb.save_path + self.cb.folder_out + self.ct.anim_out
        self.out_4 = self.cb.save_path + self.cb.folder_out + self.ct.checkpoint

        self._initialize_model()   

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
                            )
        else:
            raise ValueError('MODEL NOT RECOGNIZED')        

    def forward(self, x):
        # Inference logic
        return self.model(x)

    def training_step(self, batch, batch_idx):
        # Training logic
        front, label = batch
        pred = self(front)
        train_loss = F.mse_loss(pred, label)
        self.log("train_loss", train_loss, on_epoch=True)
        return train_loss

    def validation_step(self, batch, batch_idx, dataloader_idx):
        # Validation logic
        front, label = batch
        pred = self(front)
        val_loss = F.mse_loss(pred, label)
        if dataloader_idx == 0:
            self.log("val_SS_loss", val_loss, on_epoch=True)
        elif dataloader_idx == 1:
            self.log("val_FS_loss", val_loss, on_epoch=True)
        return val_loss

    def test_step(self, batch, batch_idx):
        # Testing logic
        x, y = batch
        y_hat = self(x)
        test_loss = F.mse_loss(y_hat, y)
        self.log("test_loss", test_loss)
        return test_loss

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

class MTTdata(pl.LightningDataModule):
    def __init__(self, cb, cd, cm, ct):
        super().__init__()
        self.cb = cb
        self.cd = cd 
        self.cm = cm  
        self.ct = ct 

    def prepare_data(self):
        pass

    def setup(self, stage=None):
        self.train_datasets = []
        self.val_datasets = []
        self.val_samplers = []
        self.val_forward_datasets = []

        for item in self.cd.datasets:
            reader = get_dataset(
                dataset_obj=READER_MAPPER[item['dataset']], 
                folderPath=str(self.cb.data_base + item["path"]), 
                file_ext=item["file_ext"], 
                resample_shape=self.cd.resample_shape, 
                resample_mode=self.cd.resample_mode, 
                timesample=item["timesample"]
            )
            reader.name = item['name']

            dataset_SS = DATASET_MAPPER[item['dataset']](reader, forward_steps = 1)
            dataset_FS = DATASET_MAPPER[item['dataset']](reader, forward_steps = self.ct.forward_steps_loss)

            train_sampler = ZeroShotSampler(dataset_SS, train_ratio=self.ct.train_ratio, split="train", n=1)
            val_sampler = ZeroShotSampler(dataset_SS, train_ratio=self.ct.train_ratio, split="val", n=1)
            val_forward_sampler = ZeroShotSampler(dataset_SS, train_ratio=self.ct.train_ratio, split="val", n=self.ct.forward_steps_loss)

            self.train_datasets.append(Subset(dataset_SS, train_sampler.indices))
            self.val_datasets.append(Subset(dataset_SS, val_sampler.indices))
            self.val_samplers.append(val_sampler)
            self.val_forward_datasets.append(Subset(dataset_FS, val_forward_sampler.indices))

        self.train_dataset = ConcatNormDataset(self.train_datasets)
        self.val_dataset = ConcatNormDataset(self.val_datasets)
        self.val_forward_dataset = ConcatNormDataset(self.val_forward_datasets)

        if self.ct.normalize:
            unnorm_dataset = self.train_dataset.normalize_velocity()
            self.ct.norm_factor = unnorm_dataset.item()
        else:
            self.ct.norm_factor = None

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.ct.batch_size,
            shuffle=True,
            pin_memory=self.ct.pin_memory
        )

    def val_dataloader(self):
        val_SS_loader = DataLoader(
            self.val_dataset,
            batch_size=self.ct.batch_size,
            shuffle=True, 
            pin_memory=self.ct.pin_memory
        )
        val_FS_loader = DataLoader(
            self.val_forward_dataset,
            batch_size=self.ct.batch_size,
            shuffle=True,
            pin_memory=self.ct.pin_memory
        )
        return [val_SS_loader, val_FS_loader]

    def val_forward_dataloader(self):
        return DataLoader(
            self.val_forward_dataset,
            batch_size=self.ct.batch_size,
            shuffle=True,
            pin_memory=self.ct.pin_memory
        )