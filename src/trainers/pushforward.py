import torch 
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import StepLR
from datetime import datetime
import time
import wandb
import argparse
import yaml
import random
import matplotlib.pyplot as plt

from dataloaders.FullLoaderBubbleML import get_dataloaders, get_datasets
from trainers.utils import rollout_temp, create_gif2

class PFTrainer:
    def __init__(self, config_path):
        self.config = self.load_config(config_path)
        self.device = torch.device(self.config['device'])
        self.epochs = self.config['training']['epochs']
        self.batch_size = self.config['batch_size']
        self.max_unrolling = self.config['training']['max_unrolling']
        self.tw = self.config['tw']
        self.learning_rate = self.config['training']['learning_rate']
        self.wandb_enabled = self.config['wandb'] in ["True", 1]
        self.model_name = self.config['modelname']
        self.discard_first = self.config['discard_first']
        self.gif_length = self.config['validation']['gif_length']
        self.tres = None
        self.makegif_val = True
        self.makegif_train = True
        self.makeplot_val = True
        self.makeplot_train = True

        self._initialize_model()

    def load_config(self, path):
        with open(path) as file:
            return yaml.safe_load(file)

    def _initialize_model(self):
        if self.model_name == "UNet_classic":
            from modelComp.UNet import UNet2DTest
            #self.model = UNet2D(in_channels=self.tw * 3, out_channels=self.tw * 3).to(self.device)
            self.model = UNet2DTest(n_class=3 * self.tw).to(self.device)
        elif self.model_name == "UNet_plusplus":
            from modelComp.UNetplusplus import UNetPlusPlus
            self.model = UNetPlusPlus(in_channels=self.tw * 3, out_channels=self.tw * 3).to(self.device)
        else:
            raise ValueError('MODEL NOT RECOGNIZED')
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        self.criterion = nn.MSELoss(reduction='mean')
        self.scheduler = StepLR(self.optimizer, step_size=20, gamma=0.1)

    def prepare_dataloader(self):
        train_files = [self.config['data_path'] + file for file in self.config['training']['files']]
        val_files = [self.config['data_path'] + file for file in self.config['validation']['files']]
        train_dataset, val_dataset = get_datasets(train_files, val_files, discard_first=self.discard_first, norm=True)
        self.train_loader, self.val_loader = get_dataloaders(train_dataset, val_dataset, 
                                                             train_batch_size=self.batch_size, train_shuffle=True)
        self.tres = train_dataset.tres

    def train_one_epoch(self):
        max_unrolling = min(self.epoch, self.max_unrolling)
        unrolling_choices = list(range(max_unrolling + 1))
        epoch_losses = []
        #print(len(self.train_loader))
        

        for i in range(self.tres):
            print(f"{i / self.tres:.2f}", end='\r')
            
            batch_losses = self._train_one_batch(unrolling_choices)
            epoch_losses.append(batch_losses.mean().item())

        return epoch_losses

    def _train_one_batch(self, unrolling_choices):
        losses = []
        
        #print('trainloader training')
        for raw_data in self.train_loader:
            #print(raw_data.shape)
            #for name, param in self.model.named_parameters():
            #    print(f"Parameter: {name}, dtype: {param.dtype}")
            unrolled_choice = random.choice(unrolling_choices)
            steps = [t for t in range(self.tw, self.tres - self.tw - (self.tw * unrolled_choice) + 1)]
            random_steps = random.choices(steps, k=self.batch_size)
            
            #print(random_steps)

            data, labels = self.create_data(raw_data, random_steps)
            data, labels = data.to(self.device), labels.to(self.device)

            with torch.no_grad():
                for _ in range(unrolled_choice):
                    random_steps = [rs + self.tw for rs in random_steps]
                    _, labels = self.create_data(raw_data, random_steps)
                    data = self.model(data)
                    labels = labels.to(self.device)

            self.optimizer.zero_grad()
            pred = self.model(data)
            loss = self.criterion(pred, labels)
            loss.backward()
            self.optimizer.step()
            losses.append(loss.detach() / self.batch_size)
            
        return torch.stack(losses)

    def validate(self):
        val_loss_timestep = self._validate_timestep()
        val_loss_unrolled = self._validate_unrolled()
        return val_loss_timestep, val_loss_unrolled

    def _validate_timestep(self):
        steps = [t for t in range(self.tw, self.tres - self.tw + 1)]
        losses = []
        #print('valloader training timestep')
        for raw_data in self.val_loader:
            step_losses = []
            
            for step in steps:
                with torch.no_grad():
                    same_steps = [step] * self.batch_size
                    data, labels = self.create_data(raw_data, same_steps)
                    data, labels = data.to(self.device), labels.to(self.device)
                    pred = self.model(data)
                    loss = self.criterion(pred, labels)
                    step_losses.append(loss / self.batch_size)

            losses.append(torch.mean(torch.stack(step_losses)))

        return torch.mean(torch.stack(losses))

    def _validate_unrolled(self):
        losses = []
        #print('valloader training   unrolled')
        for raw_data in self.val_loader:
            unrolled_losses = []
            with torch.no_grad():
                same_steps = [self.tw] * self.batch_size
                data, labels = self.create_data(raw_data, same_steps)
                data, labels = data.to(self.device), labels.to(self.device)
                pred = self.model(data)
                loss = self.criterion(pred, labels)
                unrolled_losses.append(loss / self.batch_size)

                for step in range(self.tw, self.tres - self.tw + 1, self.tw):
                    same_steps = [step] * self.batch_size
                    _, labels = self.create_data(raw_data, same_steps)
                    labels = labels.to(self.device)
                    pred = self.model(pred)
                    loss = self.criterion(pred, labels)
                    unrolled_losses.append(loss / self.batch_size)

            losses.append(torch.sum(torch.stack(unrolled_losses)))

        return torch.mean(torch.stack(losses))

    def create_data(self, raw_data, random_steps):
        data = torch.Tensor()
        labels = torch.Tensor()

        for (dp, step) in zip(raw_data, random_steps):
            d = dp[step - self.tw:step]
            l = dp[step:step + self.tw]
            #print(step, d.shape, l.shape)
            data = torch.cat((data, d[None, :]), 0)
            labels = torch.cat((labels, l[None, :]), 0)

        data = data.permute(0, 2, 1, 3, 4).reshape(data.shape[0], -1, data.shape[3], data.shape[4])
        labels = labels.permute(0, 2, 1, 3, 4).reshape(labels.shape[0], -1, labels.shape[3], labels.shape[4])
        return data, labels
    
    def make_gif(self, output_path, on_val=True):
        if on_val:
            #print('valloader gif')
            for raw_data in self.val_loader:
                break
        else:
            #print('trainloader gif')
            for raw_data in self.train_loader:
                raw_data = raw_data[0, :, :, :, :].unsqueeze(0)
                break
        input_rollout, _ = self.create_data(raw_data, [self.tw])
        #print(input_rollout.shape)
        input_rollout = input_rollout.to(self.device)
        stacked_pred = rollout_temp(self.model, input_rollout, self.device, self.tw, steps=self.gif_length)
        stacked_true = raw_data[0, :, 0, :, :]
        #print(stacked_true.shape, stacked_pred.shape)
        create_gif2(stacked_true, stacked_pred.cpu(), output_path, timesteps=self.gif_length,  vertical=True)
    
    def make_plot(self, output_path, on_val=True):
        if on_val:
            #print('valloader plot')
            for raw_data in self.val_loader:
                break
        else:
            #print('trainloader plot')
            for raw_data in self.train_loader:
                raw_data = raw_data[0, :, :, :, :].unsqueeze(0)
                break
        data, labels = self.create_data(raw_data, [random.choice(range(self.tw, self.tres - self.tw + 1))])
        data, labels = data.to(self.device), labels.to(self.device)
        pred = self.model(data)
        
        fig, ax = plt.subplots(3, 3, figsize=(10, 10))
        fig.suptitle(f"Epoch {self.epoch}")
        ax = ax.flatten()
        for i in range(3):
            ax[i].imshow(data[0, i, :, :].detach().cpu(), vmin=-1, vmax=1)
            ax[i].set_title(f"input")
        for i in range(3,6):
            ax[i].imshow(pred[0, i-3, :, :].detach().cpu(), vmin=-1, vmax=1)
            ax[i].set_title(f"pred")
        for i in range(6,9):
            ax[i].imshow(labels[0, i-6, :, :].detach().cpu(), vmin=-1, vmax=1)
            ax[i].set_title(f"labels")
        
        plt.savefig(output_path)
        plt.close()

    def train(self):
        self.prepare_dataloader()

        if self.wandb_enabled:
            wandb.init(project="BubbleML_DS_PF", name=self.model_name)
            wandb.config.update(self.config)

        best_val_loss_timestep = float('inf')
        best_val_loss_unrolled = float('inf')

        assert(self.tw * (self.max_unrolling + 2) <= self.tres)

        start_time = time.time()

        for self.epoch in range(self.epochs):
            #print('learning rate:', self.optimizer.param_groups[0]['lr'])
            self.model.train()
            train_losses = self.train_one_epoch()
            self.model.eval()
            val_loss_timestep, val_loss_unrolled = self.validate()
            self.scheduler.step()
            
            if self.epoch % 5 == 0:
                if self.makegif_val:
                    self.make_gif("output/wandb_temp_rollout_val.gif", on_val=True)
                if self.makegif_train:
                    self.make_gif("output/wandb_temp_rollout_train.gif", on_val=False)

                if self.makeplot_val:
                    self.make_plot("output/wandb_temp_plot_val.png", on_val=True)
                if self.makeplot_train:
                    self.make_plot("output/wandb_temp_plot_train.png", on_val=False)

                if val_loss_timestep < best_val_loss_timestep:
                    best_val_loss_timestep = val_loss_timestep
                    torch.save(self.model.state_dict(), f"models/best_val_loss_timestep_{self.model_name}_E{self.epoch}.pth")

                if val_loss_unrolled < best_val_loss_unrolled:
                    best_val_loss_unrolled = val_loss_unrolled
                    torch.save(self.model.state_dict(), f"models/best_val_loss_unrolled_{self.model_name}_E{self.epoch}.pth")

            if self.wandb_enabled:
                wandb.log({
                    "epoch": self.epoch,
                    "train_loss_mean": sum(train_losses) / len(train_losses),
                    "val_loss_timestep": val_loss_timestep.item(),
                    "val_loss_unrolled": val_loss_unrolled.item(),
                    "elapsed_time": time.time() - start_time,
                    "rollout_val_gif": wandb.Video('output/wandb_temp_rollout.gif', fps=5, format="gif"),
                    "learning_rate": self.optimizer.param_groups[0]['lr']
                })
                for train_loss_elem in train_losses:
                    wandb.log({"train_loss": train_loss_elem})

            print(f"Epoch {self.epoch}: Train Loss Mean = {sum(train_losses) / len(train_losses):.8f}, "
                  f"Val Loss Timestep = {val_loss_timestep:.8f}, Val Loss Unrolled = {val_loss_unrolled:.8f}, lr={self.optimizer.param_groups[0]['lr']}")

        if self.wandb_enabled:
            wandb.finish()


