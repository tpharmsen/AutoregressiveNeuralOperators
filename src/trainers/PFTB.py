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
        self.batch_size = self.config['loader']['batch_size']
        self.max_unrolling = self.config['training']['max_unrolling']
        self.tw = self.config['model']['tw']
        self.init_learning_rate = float(self.config['training']['init_learning_rate'])
        self.weight_decay = float(self.config['training']['weight_decay'])
        self.wandb_enabled = self.config['wandb'] in ["True", 1]
        self.model_name = self.config['model']['name']
        self.discard_first = self.config['loader']['discard_first']
        self.gif_length = self.config['validation']['gif_length']
        self.pin_memory = self.config['loader']['pin_memory'] in ["True", 1]
        self.prefetch_factor = self.config['loader']['prefetch_factor']
        self.num_workers = self.config['loader']['num_workers']
        self.makegif_val = self.config['validation']['makegif_val'] in ["True", 1]
        self.makeplot_val = self.config['validation']['makeplot_val'] in ["True", 1]
        self.makeplot_train = self.config['validation']['makeplot_train'] in ["True", 1]
        self.use_coords = self.config['model']['use_coords'] in ["True", 1]
        self.makegif_vertical = self.config['validation']['makegif_vertical'] in ["True", 1]
        self.save_on = self.config['save_on'] in ["True", 1]
        self.path_gif = self.config['validation']['path_gif']
        self.path_plot = self.config['validation']['path_plot']
        self.pushforward_step = self.config['training']['pushforward_step']
        self.val_rollout_length = self.config['validation']['rollout_length']
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
        with open(path, 'r') as file:
            data = yaml.safe_load(file)
        return data

    def _initialize_model(self):
        if self.model_name == 'UNet_classic':
            #from modelComp.UNetplusplus import UNetPlusPlus
            from modelComp.UNet import UNet2D
            #from modelComp.FNO import FNO2d
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
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.init_learning_rate, weight_decay=self.weight_decay)
        self.criterion = nn.MSELoss(reduction='mean')
        #self.scheduler = StepLR(self.optimizer, step_size=20, gamma=0.1)
        self.scheduler = ReduceLROnPlateau(self.optimizer, mode='min', factor=0.1, patience=25, min_lr=1e-6)

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
        self.train_max_phase = self.train_dataset.normalize_phase_()

        self.val_dataset = HDF5ConcatDataset([PFLoader(file, 
                                                    discard_first=self.discard_first, 
                                                    use_coords=self.use_coords, 
                                                    time_window = self.tw, 
                                                    push_forward_steps=1) for file in val_files])
        self.val_dataset.normalize_temp_(self.train_max_temp)
        self.val_dataset.normalize_vel_(self.train_max_vel)
        self.val_dataset.normalize_phase_(self.train_max_phase)

        self.in_channels = self.train_dataset.datasets[0].in_channels
        self.out_channels = self.train_dataset.datasets[0].out_channels

        self.train_loader = DataLoader(self.train_dataset,
                                       batch_size=self.batch_size,
                                       shuffle=True,
                                       num_workers=self.num_workers,
                                       pin_memory=self.pin_memory,
                                       prefetch_factor=self.prefetch_factor)
        self.val_loader = DataLoader(self.val_dataset, 
                                     batch_size=self.batch_size, 
                                     shuffle=False,
                                     num_workers=self.num_workers,
                                     pin_memory=self.pin_memory,
                                     prefetch_factor=self.prefetch_factor)

    def push_forward_prob(self):

        steps = self.epoch // self.pushforward_step
        if steps == 0:
            return 1
        else:
            return random.choice(range(1, 1 + min(steps + 1, self.max_unrolling)))
        
    def _index_push(self, idx, coords, temp, vel, phase):
        return (coords[:, idx], temp[:, idx], vel[:, idx], phase[:, idx])


    def _forward_int(self, coords, temp, vel, phase):

        input = torch.cat((temp, vel, phase), dim=1)
        if self.use_coords:
            input = torch.cat((coords, input), dim=1)
        pred = self.model(input)


        temp_pred = pred[:, :self.tw]
        vel_pred = pred[:, self.tw:3*self.tw]
        phase_pred = pred[:, 3*self.tw:]

        return temp_pred, vel_pred, phase_pred

    def push_forward_trick(self, coords, temp, vel, phase, push_forward_steps):
        coords_input, temp_input, vel_input, phase_input = self._index_push(0, coords, temp, vel, phase)
        with torch.no_grad():
            for idx in range(push_forward_steps - 1):
                #print('pf', end='_')
                temp_input, vel_input, phase_input = self._forward_int(coords_input, temp_input, vel_input, phase_input)
        temp_pred, vel_pred, phase_pred = self._forward_int(coords_input, temp_input, vel_input, phase_input)
        return temp_pred, vel_pred, phase_pred

    def train_one_epoch(self):
        losses = []

        for idx, (coords, temp, vel, phase, temp_label, vel_label, phase_label) in enumerate(self.train_loader):
            
            #print(f"{idx/len(self.train_loader):2f}, {idx}, {coords.shape[0]}", end='\r')
            coords, temp, vel, phase = coords.to(self.device), temp.to(self.device), vel.to(self.device), phase.to(self.device)
            #print(torch.min(phase), torch.max(phase))
            push_forward_steps = self.push_forward_prob()
            #print('push_forward_steps', push_forward_steps)
            temp_pred, vel_pred, phase_pred = self.push_forward_trick(coords, temp, vel, phase, push_forward_steps)

            idx = (push_forward_steps - 1)
            temp_label = temp_label[:, idx].to(self.device)
            #idx = (push_forward_steps - 1)
            vel_label = vel_label[:, idx].to(self.device)
            #idx = (push_forward_steps - 1)
            phase_label = phase_label[:, idx].to(self.device)

            temp_loss = self.criterion(temp_pred, temp_label)
            vel_loss = self.criterion(vel_pred, vel_label)
            phase_loss = self.criterion(phase_pred, phase_label)
            loss = (temp_loss + vel_loss + phase_loss) / 3
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            losses.append(loss.detach())
            
            del temp, vel, phase, temp_label, vel_label, phase_label
            
        return torch.mean(torch.stack(losses))
    
    def validate(self):
        val_loss_timestep = self._validate_timestep()
        dataset = self.val_dataset.get_validation_stacks(0)
        val_loss_unrolled = self._validate_unrolled(dataset)
        return val_loss_timestep, val_loss_unrolled

    def _validate_timestep(self):
        losses = []
        for idx, (coords, temp, vel, phase, temp_label, vel_label, phase_label) in enumerate(self.val_loader):
            coords, temp, vel, phase = coords.to(self.device), temp.to(self.device), vel.to(self.device), phase.to(self.device)
            #if idx==0:
            #    print(temp.shape, vel.shape)
            temp_label = temp_label[:, 0].to(self.device)
            vel_label = vel_label[:, 0].to(self.device)
            phase_label = phase_label[:, 0].to(self.device)

            #print(torch.mean(coords), torch.std(coords))

            with torch.no_grad():
                #print(coords[:, 0].shape, temp[:, 0].shape, vel[:, 0].shape)    
                temp_pred, vel_pred, phase_pred = self._forward_int(coords[:, 0], temp[:, 0], vel[:, 0], phase[:, 0])
                temp_loss = self.criterion(temp_pred, temp_label)
                vel_loss = self.criterion(vel_pred, vel_label)
                phase_loss = self.criterion(phase_pred, phase_label)
                loss = (temp_loss + vel_loss + phase_loss) / 3
            
            losses.append(loss.detach())
            del temp, vel, phase, temp_label, vel_label, phase_label
        return torch.mean(torch.stack(losses))
    
    def _validate_unrolled(self, dataset):
        
        for idx, (coords, temp, vel, phase, temp_label, vel_label, phase_label) in enumerate(self.val_loader):
            #print(temp.shape, vel.shape)
            coords = coords[0, 0].unsqueeze(0).to(self.device)
            break

        losses = []
        list_temp_pred, list_vel_pred, list_phase_pred = [], [], []

        temp_true, vel_true, phase_true = dataset
        temp_true, vel_true, phase_true = temp_true.to(self.device), vel_true.to(self.device), phase_true.to(self.device)
        #print(temp_true.shape, vel_true.shape)
        #temp_pred, velx_pred, vely_pred = temp_true[:, 0], velx_true[:, 0], vely_true[:, :, 0]
        temp_pred, vel_pred, phase_pred = temp_true[:, :self.tw], vel_true[:, :2*self.tw], phase_true[:, :self.tw]
        #print(temp_pred.shape, vel_pred.shape, coords.shape)
        for t in range(0, self.val_rollout_length, self.tw):
            with torch.no_grad():
                #print('test')
                temp_pred, vel_pred, phase_pred = self._forward_int(coords, temp_pred, vel_pred, phase_pred)
                #print('prediction')    
                list_temp_pred.append(temp_pred)
                list_vel_pred.append(vel_pred)
                list_phase_pred.append(phase_pred)
        #print(len(list_temp_pred), len(list_vel_pred))
        #print('now stacking preds')
        temp_preds = torch.cat(list_temp_pred, dim=1)
        vel_preds = torch.cat(list_vel_pred, dim=1)    
        phase_preds = torch.cat(list_phase_pred, dim=1)
        #print(temp_preds.shape, vel_preds.shape)
        temp_loss = self.criterion(temp_preds, temp_true[:, :self.val_rollout_length])
        vel_loss = self.criterion(vel_preds, vel_true[:, :2*self.val_rollout_length])
        phase_loss = self.criterion(phase_preds, phase_true[:, :self.val_rollout_length])
        loss = (temp_loss + vel_loss + phase_loss) / 3
        losses.append(loss.detach())

        return torch.mean(torch.stack(losses))
    
    def make_gif(self, output_path, on_val=True):
        if on_val:
            for coords, temp, vel, phase, temp_label, vel_label, phase_label in self.val_loader:
                break
            stacked_true = self.val_dataset._get_temp_full(num=0)
        else:
            for coords, temp, vel, phase, temp_label, vel_label, phase_label in self.train_loader:
                break
            stacked_true = self.train_dataset._get_temp_full(num=6)
            #TODO: this does not align with random selection of train sample. Fix this

        coords, temp, vel, phase = coords[0].unsqueeze(0), temp[0].unsqueeze(0), vel[0].unsqueeze(0), phase[0].unsqueeze(0)
        coords, temp, vel, phase = coords.to(self.device), temp.to(self.device), vel.to(self.device), phase.to(self.device)

        preds = []
        with torch.no_grad():
            coords_input, temp_input, vel_input, phase_input = self._index_push(0, coords, temp, vel, phase)
            preds.append(temp_input)
            for i in range(1, self.gif_length//self.tw + 1):
                temp_input, vel_input, phase_input = self._forward_int(coords_input, temp_input, vel_input, phase_input)
                preds.append(temp_input)
        stacked_pred = torch.cat(preds, dim=1).squeeze(0)
        #print(stacked_true.shape, stacked_pred.shape)

        create_gif2(stacked_true.cpu(), stacked_pred.cpu(), output_path, timesteps=self.gif_length, vertical=self.makegif_vertical)
    
    def make_plot(self, output_path, on_val=True):
        if on_val:
            for coords, temp, vel, phase, temp_label, vel_label, phase_label in self.val_loader:
                break           
        else:
            for coords, temp, vel, phase, temp_label, vel_label, phase_label in self.train_loader:
                break  
        coords, temp, vel, phase = coords[0].unsqueeze(0), temp[0].unsqueeze(0), vel[0].unsqueeze(0), phase[0].unsqueeze(0)
        coords, temp, vel, phase = coords.to(self.device), temp.to(self.device), vel.to(self.device), phase.to(self.device)
            
        with torch.no_grad():
            coords_input, temp_input, vel_input, phase_input = self._index_push(0, coords, temp, vel, phase)
            temp_pred, vel_pred, phase_pred = self._forward_int(coords_input, temp_input, vel_input, phase_input)

        temp_label = temp_label[0, 0].unsqueeze(0).to(self.device)
        #vel_label = vel_label[0, 0].unsqueeze(0).to(self.device)
        #phase_label = phase_label[0, 0].unsqueeze(0).to(self.device)

        fig, ax = plt.subplots(3, self.tw, figsize=(3*self.tw, 9))
        fig.suptitle(f"Epoch {self.epoch}")
        ax = ax.flatten()
        for i in range(self.tw):
            ax[i].imshow(temp_input[0, i, :, :].detach().cpu(), vmin=-1, vmax=1)
            #ax[i].imshow(vel_input[0, 2* i, :, :].detach().cpu(), vmin=-1, vmax=1)
            ax[i].set_title(f"input")
        for i in range(self.tw,2*self.tw):
            ax[i].imshow(temp_pred[0, i-self.tw, :, :].detach().cpu(), vmin=-1, vmax=1)
            #ax[i].imshow(vel_pred[0, i-self.tw, :, :].detach().cpu(), vmin=-1, vmax=1)
            ax[i].set_title(f"pred")
        for i in range(2*self.tw,3*self.tw):
            ax[i].imshow(temp_label[0, i-2*self.tw, :, :].detach().cpu(), vmin=-1, vmax=1)
            #ax[i].imshow(vel_label[0, i-2*self.tw, :, :].detach().cpu(), vmin=-1, vmax=1)
            ax[i].set_title(f"labels")
        
        plt.tight_layout()
        plt.savefig(output_path)
        plt.close()

    def train(self):
        self.prepare_dataloader()
        self._initialize_model()

        if self.wandb_enabled:
            #wandb.init(project="BubbleML_DS_PF", name=self.model_name + datetime.now().strftime("_%Y-%m-%d_%H-%M"), config=self.wandb_config)
            wandb.init(project="BubbleML_DS_PF", name=self.model_name + '_wd=' + str(self.weight_decay) + '_maxpf=' + str(self.max_unrolling) + '_bs=' + str(self.batch_size) + '_pfst=' + str(self.pushforward_step) + '_dp=' + str(self.modelprop[1]), config=self.wandb_config)
            wandb.config.update(self.config)

        best_val_loss_timestep = float('inf')
        best_val_loss_unrolled = float('inf')
        best_train_loss = float('inf')

        start_time = time.time()

        for self.epoch in range(self.epochs):
            train_start_time = time.time()
            self.model.train()
            train_losses = self.train_one_epoch()
            #train_losses = -1
            val_start_time = time.time()
            self.model.eval()
            val_loss_timestep, val_loss_unrolled = self.validate()
            val_end_time = time.time()

            makeviz = self.epoch % 10 == 0
            if makeviz:
                viz_start_time = time.time()
                if self.makegif_val:
                    self.make_gif(self.path_gif, on_val=True)
                if self.makeplot_val:
                    self.make_plot("output/tempplot_val.png", on_val=True)
                if self.makeplot_train:
                    self.make_plot(self.path_plot, on_val=False)
                viz_end_time = time.time()
                #print(val_loss_timestep[0])
                #print(val_loss_timestep)
                #print()
                #print(val_loss_timestep.dtype)
                if self.save_on:
                    if val_loss_timestep < best_val_loss_timestep:
                        best_val_loss_timestep = val_loss_timestep
                        
                        torch.save(self.model.state_dict(), f"models/best_val_loss_timestep_{self.model_name}_E{self.epoch}_3.pth")
                    if train_losses < best_train_loss:
                        best_train_loss = train_losses
                        torch.save(self.model.state_dict(), f"models/best_train_loss_{self.model_name}_E{self.epoch}_3.pth")

            if self.wandb_enabled:
                wandb.log({
                    "epoch": self.epoch,
                    "train_loss_mean": torch.mean(train_losses),
                    "val_loss_timestep": torch.mean(val_loss_timestep),
                    "val_loss_unrolled": val_loss_unrolled.item(),
                    "elapsed_time": time.time() - start_time,
                    "train_time": val_start_time - train_start_time,
                    "val_time": val_end_time - val_start_time,
                    "viz_time": viz_end_time - viz_start_time if makeviz else None,
                    "rollout_val_gif": wandb.Video(self.path_gif, fps=5, format="gif") if self.makegif_val and makeviz else None,
                    "rollout_val_plot": wandb.Image('output/tempplot_val.png') if self.makeplot_val and makeviz else None,
                    "rollout_train_plot": wandb.Image(self.path_plot) if self.makeplot_train and makeviz else None,
                    "learning_rate": self.optimizer.param_groups[0]['lr']
                })
            

            self.scheduler.step(torch.mean(val_loss_timestep))
                
            print(f"Epoch {self.epoch}: "
                    f"Train Loss Mean = {train_losses:.8f}, "
                    f"Val Loss Timestep = {val_loss_timestep:.8f}, "
                    f"Val Loss Unrolled = {val_loss_unrolled:.8f}, "
                    f"LR: {self.optimizer.param_groups[0]['lr']:.2e}")

        if self.wandb_enabled:
            wandb.finish()
