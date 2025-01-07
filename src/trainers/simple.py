import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
import datetime
import wandb
import yaml
from modelComp.UNet import UNet2D
from dataloaders.SimpleLoaderBubbleML import SimpleLoaderBubbleML, get_dataloaders
from trainers.utils import create_gif2, rollout_temp

class SimpleTrainer:
    def __init__(self, config_path):
        self.config = self.load_config(config_path)
        self.EPOCHS = self.config['training']['epochs']
        self.DEVICE = torch.device(self.config["device"])
        self.modelname = self.config['modelname']

        self.train_loader, self.val_loader = self.prepare_dataloader()
        self.model = UNet2D(in_channels=3, out_channels=3, features=[64, 128, 256, 512]).to(self.DEVICE)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.config['training']['learning_rate'])

        wandb.init(project="bubbleml_DS", name=self.modelname)
        wandb.config.update(self.config)

        self.best_val_loss = float('inf')
        self.best_train_loss = float('inf')
        self.makegif = True

    def load_config(self, config_path):
        with open(config_path) as file:
            config = yaml.safe_load(file)
        return config

    def prepare_dataloader(self):
        train_files = [self.config['data_path'] + file for file in self.config['training']['files']]
        val_files = [self.config['data_path'] + file for file in self.config['training']['files']]
        return get_dataloaders(train_files, val_files, self.config['batch_size'])

    def one_epoch_train(self):
        epoch_train_loss = []
        self.model.train()

        for idx, (input, label) in enumerate(self.train_loader):
            print(f"{idx/len(self.train_loader):2f}", end='\r')
            input = input.to(self.DEVICE).float()
            label = label.to(self.DEVICE).float()
            pred = self.model(input)
            self.optimizer.zero_grad()
            loss = F.mse_loss(pred, label)
            loss.backward()
            self.optimizer.step()
            epoch_train_loss.append(loss.item())

        return torch.mean(torch.tensor(epoch_train_loss))

    def one_epoch_val(self):
        epoch_val_loss = []
        self.model.eval()

        with torch.no_grad():
            for input, label in self.val_loader:
                input = input.to(self.DEVICE).float()
                label = label.to(self.DEVICE).float()
                pred = self.model(input)
                loss = F.mse_loss(pred, label)
                epoch_val_loss.append(loss.item())

        return torch.mean(torch.tensor(epoch_val_loss))
    
    def train(self):
        current_time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        print("Training is starting...", current_time)
        print(torch.cuda.is_available())

        for epoch in range(self.EPOCHS):
            avg_train_loss = self.one_epoch_train()
            avg_val_loss = self.one_epoch_val()
            
            # create gif for the log
            # TODO: make gif here

            print(f"Epoch {epoch + 1}/{self.EPOCHS} - Train Loss: {avg_train_loss:.4f}, Validation Loss: {avg_val_loss:.4f}")

            wandb.log({"epoch": epoch, 
                       "train_loss": avg_train_loss, 
                       "val_loss": avg_val_loss})

            if epoch % 10 == 0:
                if avg_val_loss < self.best_val_loss:
                    self.best_val_loss = avg_val_loss
                    best_model_path = f"models/best_model_val_{self.modelname}.pth"
                    torch.save(self.model.state_dict(), best_model_path)
                if avg_train_loss < self.best_train_loss:
                    self.best_train_loss = avg_train_loss
                    best_model_path = f"models/best_model_train_{self.modelname}.pth"
                    torch.save(self.model.state_dict(), best_model_path)

        filename = f"models/finalmodel_{self.modelname}.pth"
        torch.save(self.model.state_dict(), filename)
        print("Training is finished. Model is saved as:", filename)

        wandb.save(filename)
        wandb.finish()
