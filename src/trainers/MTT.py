import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
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

os.environ["CUDA_VISIBLE_DEVICES"] = "0" # if gpu mig is used

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
        
        num_gpus = torch.cuda.device_count()
        #print(num_gpus)
        #print()
        wandb_logger = WandbLogger(project="FluidGPT", config = self.build_wandb_config(), name=self.cb.wandb_name, save_dir=self.cb.save_path + self.cb.folder_out)
        trainer = pl.Trainer(
            precision="bf16-mixed",
            accelerator="gpu",
            devices= 'auto',
            logger=wandb_logger,#num_gpus,
            #log_every_n_steps=0,
            callbacks=[LoggerCallback(self.cb, self.cd, self.cm, self.ct)],
            #strategy="ddp"
            max_epochs=self.ct.epochs,
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


class MTTmodel(pl.LightningModule):
    def __init__(self, cb, cd, cm, ct):
        super().__init__()

        self.cb = cb
        self.cd = cd
        self.cm = cm
        self.ct = ct

        self.train_losses = []
        self.val_SS_losses = []
        self.val_FS_losses = []
        self.epoch_start_time = None

        self._initialize_model()   
        self.counter = 0

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
        #self.log("train_loss", train_loss, on_step=False, on_epoch=True, prog_bar=True)
        #return train_loss
        self.train_losses.append(train_loss.item())

    def validation_step(self, batch, batch_idx, dataloader_idx):
        # Validation logic
        front, label = batch
        pred = self(front)
        val_loss = F.mse_loss(pred, label)
        if dataloader_idx == 0:
            self.val_SS_losses.append(val_loss.item())
            #self.log("val_SS_loss", val_loss, on_step=False, on_epoch=True, prog_bar=False)
        elif dataloader_idx == 1:
            self.val_FS_losses.append(val_loss.item())
            #self.log("val_FS_loss", val_loss, on_step=False, on_epoch=True, prog_bar=False)
        #return {'val_loss': val_loss, 'dataloader_idx': dataloader_idx}


    def test_step(self, batch, batch_idx):
        # Testing logic
        x, y = batch
        y_hat = self(x)
        test_loss = F.mse_loss(y_hat, y)
        #self.log("test_loss", test_loss, on_step=False, on_epoch=True, prog_bar=True)
        #return test_loss

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
            #"monitor": "val_SS_loss/dataloader_idx_0", 
            "monitor": "val_SS_loss",
            "interval": "epoch",
            "frequency": 1
        }
        return {"optimizer": optimizer, "lr_scheduler": scheduler}
    
    def on_train_epoch_start(self):
        self.epoch_time = time.time()

    def on_validation_epoch_end(self):
        if not self.trainer.sanity_checking:
            self.epoch_time = time.time() - self.epoch_time

            train_loss = np.mean(self.train_losses)
            val_SS_loss = np.mean(self.val_SS_losses)
            val_FS_loss = np.mean(self.val_FS_losses)
            
            self.logger.experiment.log({
                "epoch": self.current_epoch,
                "train_loss": train_loss,
                "val_SS_loss": val_SS_loss,
                "val_FS_loss": val_FS_loss,
                "Learning Rate": self.trainer.optimizers[0].param_groups[0]['lr'],
                "Epoch Time": self.epoch_time
            })

            self.log_dict({
                "val_SS_loss": val_SS_loss,
            }, prog_bar=False)

        self.train_losses = []
        self.val_SS_losses = []
        self.val_FS_losses = []

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

class LoggerCallback(pl.Callback):
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
        self.callback_start_time = None

    def on_validation_epoch_end(self, trainer, model):
        if trainer.sanity_checking:
            return
        epoch = trainer.current_epoch
        if self.cb.viz and epoch % self.cb.viz_freq == 0:
            self.callback_start_time = time.time()
            device = model.device

            self.make_plot(model, self.out_1, mode='val', device=device)
            self.make_plot(model, self.out_0, mode='train', device=device)
            self.make_plot(model, self.out_2, mode='val_forward', device=device)
            
            #self.make_anim(model, self.out_3, device=device)

            self.callback_start_time = time.time() - self.callback_start_time
            trainer.logger.experiment.log({
                "epoch": epoch,
                "train_plot": wandb.Image(self.out_0),
                "val_plot": wandb.Image(self.out_1),
                "val_forward_plot": wandb.Image(self.out_2),
                #"val_anim": wandb.Video(self.out_3, format="gif"),
                "Callback Time": self.callback_start_time
            })

    def make_anim(self, model, output_path, device):
        model.eval()
        with torch.no_grad():
            dataset_idx = torch.randint(0, len(model.trainer.datamodule.val_datasets), (1,)).item()
            traj_idx = model.trainer.datamodule.val_samplers[dataset_idx].random_val_traj()
            val_traj = model.trainer.datamodule.val_datasets[dataset_idx].dataset.reader.get_single_traj(traj_idx)

            front = val_traj[0].unsqueeze(0).to(device)
            stacked_pred = rollout(front, model, len(val_traj))
            stacked_pred, stacked_true = magnitude_vel(stacked_pred), magnitude_vel(val_traj)
            dataset_name = model.trainer.datamodule.val_datasets[dataset_idx].dataset.reader.name
            animate_rollout(stacked_pred, stacked_true, dataset_name, output_path)

    def make_plot(self, model, output_path, mode='val', device='cuda'):
        model.eval()

        if mode == 'val':
            loader = iter(model.trainer.datamodule.val_dataloader()[0])
        elif mode == 'train':
            loader = iter(model.trainer.datamodule.train_dataloader())
        elif mode == 'val_forward':
            loader = iter(model.trainer.datamodule.val_dataloader()[1])
        else:
            raise ValueError('PLOTMODE NOT RECOGNIZED')

        front, label = next(loader)
        front, label = front.to(device), label.to(device)
        front, label = front[0].unsqueeze(0), label[0].unsqueeze(0)

        with torch.no_grad():
            pred = model(front)

        front_x, front_y = front[0, 0].cpu(), front[0, 1].cpu()
        pred_x, pred_y = pred[0, 0].cpu(), pred[0, 1].cpu()
        label_x, label_y = label[0, 0].cpu(), label[0, 1].cpu()
        diff_x, diff_y = torch.zeros(front_x.shape), torch.zeros(front_y.shape)
        #diff_x, diff_y = (label_x - pred_x).abs(), (label_y - pred_y).abs()

        fig, axes = plt.subplots(2, 4, figsize=(16, 8))
        fig.suptitle(f"Epoch {model.trainer.current_epoch}")

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