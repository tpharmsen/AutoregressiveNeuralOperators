import torch 
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, ConcatDataset, random_split
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
from dataloaders.utils import get_dataset

class STT:
    def __init__(self, cb, cd, cm, ct, cv):
        self.cb = cb
        self.cd = cd
        self.cm = cm
        self.ct = ct
        self.cv = cv
        self.device = torch.device(self.cb.device)
        #self._initialize_model()
        #self.prepare_dataloader()

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
        print(self.ct.init_lr, self.ct.weight_decay)
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
            
            train_ratio = self.ct.train_ratio
            train_size = int(len(dataset) * train_ratio)
            val_size = len(dataset) - train_size

            train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
            
            train_datasets.append(train_dataset)
            val_datasets.append(val_dataset)

        train_dataset = ConcatDataset(train_datasets)
        val_dataset = ConcatDataset(val_datasets)

        train_loader = DataLoader(train_dataset, batch_size=self.ct.batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=self.ct.batch_size, shuffle=False)
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
        for epoch in self.ct.epochs:
            print('epoch:', epoch)
            train_loss = self.train_one_epoch()
            val_loss = self._validate_timestep()
            
        print('finished')
