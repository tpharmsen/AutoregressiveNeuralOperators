import torch 
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import StepLR, ReduceLROnPlateau
from datetime import datetime
import time
import wandb
import argparse
import yaml
import random
import matplotlib.pyplot as plt
import numpy as np

from dataloaders.PFLoader import PFLoader, HDF5ConcatDataset
from trainers.utils import rollout_temp, create_gif2

class PFTBTrainer:
    def __init__(self, config_path):
        self.config = self.load_config(config_path)
        self.device = torch.device(self.config['device'])
        self.epochs = self.config['training']['epochs']
        self.batch_size = self.config['batch_size']
        self.max_unrolling = self.config['training']['max_unrolling']
        self.tw = self.config['model']['tw']
        self.learning_rate = self.config['training']['learning_rate']
        self.wandb_enabled = self.config['wandb'] in ["True", 1]
        self.model_name = self.config['model']['name']
        self.discard_first = self.config['discard_first']
        self.gif_length = self.config['validation']['gif_length']
        self.tres = None
        self.makegif_val = self.config['validation']['makegif_val'] in ["True", 1]
        self.makeplot_val = self.config['validation']['makeplot_val'] in ["True", 1]
        self.makeplot_train = self.config['validation']['makeplot_train'] in ["True", 1]
        self.use_coords = self.config['model']['use_coords'] in ["True", 1]
        self.makegif_vertical = self.config['validation']['makegif_vertical'] in ["True", 1]
        self.save_on = self.config['save_on'] in ["True", 1]
        #torch.set_printoptions(precision=6, sci_mode=False)
        self.modelprop = self.config['model']['prop']

        self.wandb_config = {}
        prefix = ''
        for key, value in self.config.items():
            full_key = f"{prefix}.{key}" if prefix else key
            if isinstance(value, dict):
                prefix = key
                for key, value in self.config.items():
                    full_key = f"{prefix}.{key}" if prefix else key
                    self.wandb_config[key] = value
            else:
                self.wandb_config[key] = value

    def load_config(self, path):
        with open(path) as file:
            return yaml.safe_load(file)

    def _initialize_model(self):
        if self.model_name:
            from modelComp.UNetplusplus import UNetPlusPlus
            from modelComp.UNet import UNet2DTest, UNet2D
            from modelComp.FNO import FNO2d
            #from neuralop.models import FNO
            #self.model = UNetPlusPlus(in_channels=self.in_channels, out_channels=self.out_channels).to(self.device)
            #self.model = UNet2DTest(in_channels=self.in_channels, out_channels=self.out_channels).to(self.device)
            #self.model = FNO2d(in_channels=self.in_channels, out_channels=self.out_channels, modes1=8, modes2=8, width=6).to(self.device)
            #self.model = FNO(n_modes=(16, 16), hidden_channels=64, in_channels=self.in_channels, out_channels = self.out_channels).to(self.device)
            self.model = UNet2D(in_channels=self.in_channels,
                                out_channels=self.out_channels, 
                                base_filters=self.modelprop[0], 
                                depth=self.modelprop[1],
                                activation=self.modelprop[2]
                                ).to(self.device)
            print('Amount of parameters in model:', self.nparams(self.model))
        else:
            raise ValueError('MODEL NOT RECOGNIZED')
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        self.criterion = nn.MSELoss(reduction='mean')
        #self.scheduler = StepLR(self.optimizer, step_size=20, gamma=0.1)
        self.scheduler = ReduceLROnPlateau(self.optimizer, mode='min', factor=0.1, patience=20, min_lr=1e-6)

    def nparams(self, model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)

    def prepare_dataloader(self):
        train_files = [self.config['data_path'] + file for file in self.config['training']['files']]
        val_files = [self.config['data_path'] + file for file in self.config['validation']['files']]
        #print(len(train_files), len(val_files)) 

        self.train_dataset = HDF5ConcatDataset([PFLoader(file, 
                                                    discard_first=self.discard_first, 
                                                    use_coords=self.use_coords, 
                                                    time_window = self.tw, 
                                                    push_forward_steps=self.max_unrolling) for file in train_files])
        self.train_max_temp = self.train_dataset.normalize_temp_()
        self.train_max_vel = self.train_dataset.normalize_vel_()

        self.val_dataset = HDF5ConcatDataset([PFLoader(file, 
                                                    discard_first=self.discard_first, 
                                                    use_coords=self.use_coords, 
                                                    time_window = self.tw, 
                                                    push_forward_steps=self.max_unrolling) for file in val_files])
        self.val_dataset.normalize_temp_(self.train_max_temp)
        self.val_dataset.normalize_vel_(self.train_max_vel)

        self.in_channels = self.train_dataset.datasets[0].in_channels
        self.out_channels = self.train_dataset.datasets[0].out_channels

        self.train_loader = DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True)
        self.val_loader = DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False)

    def push_forward_prob(self, epoch, max_epochs):

        steps = epoch // 10
        return random.choice(range(1, max(steps + 1, self.max_unrolling + 1)))
        
    def _index_push(self, idx, coords, temp, vel):
        return (coords[:, idx], temp[:, idx], vel[:, idx])


    def _forward_int(self, coords, temp, vel):

        input = torch.cat((temp, vel), dim=1)
        if self.use_coords:
            input = torch.cat((coords, input), dim=1)
        pred = self.model(input)


        temp_pred = pred[:, :self.tw]
        vel_pred = pred[:, self.tw:]

        return temp_pred, vel_pred

    def push_forward_trick(self, coords, temp, vel, push_forward_steps):
        coords_input, temp_input, vel_input = self._index_push(0, coords, temp, vel)
        with torch.no_grad():
            for idx in range(push_forward_steps - 1):
                #print('pf', end='_')
                temp_input, vel_input = self._forward_int(coords_input, temp_input, vel_input)
        temp_pred, vel_pred = self._forward_int(coords_input, temp_input, vel_input)
        return temp_pred, vel_pred

    def train_one_epoch(self):
        losses = []

        for idx, (coords, temp, vel, temp_label, vel_label) in enumerate(self.train_loader):

            #print(f"{idx/len(self.train_loader):2f}, {idx}, {coords.shape[0]}", end='\r')
            coords, temp, vel = coords.to(self.device), temp.to(self.device), vel.to(self.device)
            
            push_forward_steps = self.push_forward_prob(self.epoch, self.epochs)
            #print('push_forward_steps', push_forward_steps)
            temp_pred, vel_pred = self.push_forward_trick(coords, temp, vel, push_forward_steps)

            idx = (push_forward_steps - 1)
            temp_label = temp_label[:, idx].to(self.device)
            idx = (push_forward_steps - 1)
            vel_label = vel_label[:, idx].to(self.device)

            temp_loss = self.criterion(temp_pred, temp_label)
            vel_loss = self.criterion(vel_pred, vel_label)
            loss = (temp_loss + vel_loss) / 2
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            losses.append(loss.detach() / self.batch_size)
            
            del temp, vel, temp_label, vel_label
            
        return torch.stack(losses)
    
    def validate(self):
        val_loss_timestep = self._validate_timestep()
        #val_loss_unrolled = self._validate_unrolled()
        val_loss_unrolled = -1
        
        return val_loss_timestep, val_loss_unrolled

    def _validate_timestep(self):
        losses = []
        for idx, (coords, temp, vel, temp_label, vel_label) in enumerate(self.val_loader):
            coords, temp, vel, = coords.to(self.device), temp.to(self.device), vel.to(self.device)
            if idx == 0:
                with torch.no_grad():
                    temp_pred, vel_pred = self._forward_int(coords[:, 0], temp[:, 0], vel[:, 0])

            # val doesn't apply push-forward
            temp_label = temp_label[:, 0].to(self.device)
            vel_label = vel_label[:, 0].to(self.device)

            with torch.no_grad():
                temp_pred, vel_pred = self._forward_int(coords[:, 0], temp[:, 0], vel[:, 0])
                temp_loss = self.criterion(temp_pred, temp_label)
                vel_loss = self.criterion(vel_pred, vel_label)
                loss = (temp_loss + vel_loss) / 2
            
            losses.append(loss.detach() / self.batch_size)
            del temp, vel, temp_label, vel_label
        return torch.stack(losses)
    
    def make_gif(self, output_path, on_val=True):
        if on_val:
            for coords, temp, vel, temp_label, vel_label in self.val_loader:
                break
            coords, temp, vel = coords[0].unsqueeze(0), temp[0].unsqueeze(0), vel[0].unsqueeze(0)
            coords, temp, vel = coords.to(self.device), temp.to(self.device), vel.to(self.device)
            
            stacked_true = self.val_dataset._get_temp_full(num=0)
        else:
            for coords, temp, vel, temp_label, vel_label in self.train_loader:
                break
            coords, temp, vel = coords[0].unsqueeze(0), temp[0].unsqueeze(0), vel[0].unsqueeze(0)
            coords, temp, vel = coords.to(self.device), temp.to(self.device), vel.to(self.device)
            
            stacked_true = self.train_dataset._get_temp_full(num=6)
        preds = []
        with torch.no_grad():
            coords_input, temp_input, vel_input = self._index_push(0, coords, temp, vel)
            preds.append(temp_input)
            for i in range(1, self.gif_length//self.tw + 1):
                temp_input, vel_input = self._forward_int(coords_input, temp_input, vel_input)
                preds.append(temp_input)
        stacked_pred = torch.cat(preds, dim=1).squeeze(0)
        #print(stacked_true.shape, stacked_pred.shape)

        create_gif2(stacked_true.cpu(), stacked_pred.cpu(), output_path, timesteps=self.gif_length, vertical=self.makegif_vertical)
    
    def make_plot(self, output_path, on_val=True):
        if on_val:
            for coords, temp, vel, temp_label, vel_label in self.val_loader:
                break
            coords, temp, vel = coords[0].unsqueeze(0), temp[0].unsqueeze(0), vel[0].unsqueeze(0)
            coords, temp, vel = coords.to(self.device), temp.to(self.device), vel.to(self.device)
            
        else:
            for coords, temp, vel, temp_label, vel_label in self.train_loader:
                break
            coords, temp, vel = coords[0].unsqueeze(0), temp[0].unsqueeze(0), vel[0].unsqueeze(0)
            coords, temp, vel = coords.to(self.device), temp.to(self.device), vel.to(self.device)
            
        with torch.no_grad():
            coords_input, temp_input, vel_input = self._index_push(0, coords, temp, vel)
            temp_pred, vel_pred = self._forward_int(coords_input, temp_input, vel_input)

        temp_label = temp_label[0, 0].unsqueeze(0).to(self.device)
        

        fig, ax = plt.subplots(3, self.tw, figsize=(3*self.tw, 9))
        fig.suptitle(f"Epoch {self.epoch}")
        ax = ax.flatten()
        for i in range(self.tw):
            ax[i].imshow(temp_input[0, i, :, :].detach().cpu(), vmin=-1, vmax=1)
            ax[i].set_title(f"input")
        for i in range(self.tw,2*self.tw):
            ax[i].imshow(temp_pred[0, i-self.tw, :, :].detach().cpu(), vmin=-1, vmax=1)
            ax[i].set_title(f"pred")
        for i in range(2*self.tw,3*self.tw):
            ax[i].imshow(temp_label[0, i-2*self.tw, :, :].detach().cpu(), vmin=-1, vmax=1)
            ax[i].set_title(f"labels")
        
        plt.tight_layout()
        plt.savefig(output_path)
        plt.close()

    def train(self):
        self.prepare_dataloader()
        self._initialize_model()

        if self.wandb_enabled:
            wandb.init(project="BubbleML_DS_PF", name=self.model_name + datetime.now().strftime("_%Y-%m-%d_%H-%M"), config=self.wandb_config)
            wandb.config.update(self.config)

        best_val_loss_timestep = float('inf')
        best_val_loss_unrolled = float('inf')
        best_train_loss = float('inf')

        start_time = time.time()

        for self.epoch in range(self.epochs):

            self.model.train()
            train_losses = self.train_one_epoch()
            self.model.eval()
            val_loss_timestep, val_loss_unrolled = self.validate()
            
            makeviz = self.epoch % 10 == 0
            if makeviz:
                if self.makegif_val:
                    self.make_gif("output/tempgif.gif", on_val=True)
                if self.makeplot_val:
                    self.make_plot("output/tempplot.png", on_val=True)
                if self.makeplot_train:
                    self.make_plot("output/tempplot_train.png", on_val=False)

                #print(val_loss_timestep[0])
                #print(val_loss_timestep)
                #print()
                #print(val_loss_timestep.dtype)
                if self.save_on:
                    if torch.mean(val_loss_timestep) < best_val_loss_timestep:
                        best_val_loss_timestep = torch.mean(val_loss_timestep)
                        
                        torch.save(self.model.state_dict(), f"models/best_val_loss_timestep_{self.model_name}_E{self.epoch}_3.pth")
                    if torch.mean(train_losses) < best_train_loss:
                        best_train_loss = torch.mean(train_losses)
                        torch.save(self.model.state_dict(), f"models/best_train_loss_{self.model_name}_E{self.epoch}_3.pth")

            if self.wandb_enabled:
                wandb.log({
                    "epoch": self.epoch,
                    "train_loss_mean": torch.mean(train_losses),
                    "val_loss_timestep": torch.mean(val_loss_timestep),
                    #"val_loss_unrolled": val_loss_unrolled.item(),
                    "elapsed_time": time.time() - start_time,
                    "rollout_val_gif": wandb.Video('output/tempgif.gif', fps=5, format="gif") if self.makegif_val and makeviz else None,
                    "rollout_val_plot": wandb.Image('output/tempplot.png') if self.makeplot_val and makeviz else None,
                    "rollout_train_plot": wandb.Image('output/tempplot_train.png') if self.makeplot_train and makeviz else None,
                    "learning_rate": self.optimizer.param_groups[0]['lr']
                })
            

            self.scheduler.step(torch.mean(val_loss_timestep))
                
            print(f"Epoch {self.epoch}: Train Loss Mean = {torch.mean(train_losses):.8f}, "
                    f"Val Loss Timestep = {torch.mean(val_loss_timestep):.8f}, "
                    f"LR: {self.optimizer.param_groups[0]['lr']:.2e}")

        if self.wandb_enabled:
            wandb.finish()
