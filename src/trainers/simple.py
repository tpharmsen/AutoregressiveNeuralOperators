import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
import datetime
import wandb
import yaml
from modelComp.UNet import UNet2D
from dataloaders.SimpleLoaderBubbleML import SimpleLoaderBubbleML, get_datasets, get_dataloaders
from trainers.utils import create_gif2, rollout_temp

class SimpleTrainer:
    def __init__(self, config_path):
        self.config = self.load_config(config_path)
        self.EPOCHS = self.config['training']['epochs']
        self.DEVICE = torch.device(self.config["device"])
        self.modelname = self.config['modelname']
        self.wandb_enabled = self.config['wandb'] in ["True", 1]
        self.wandb_enabled = False

        self.train_loader, self.val_loader = self.prepare_dataloader()
        self.model = UNet2D(in_channels=3, out_channels=3, features=[64, 128, 256, 512]).to(self.DEVICE)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.config['training']['learning_rate'])

        if self.wandb_enabled:
            wandb.init(project="bubbleml_DS", name=self.modelname)
            wandb.config.update(self.config)

        self.best_val_loss = float('inf')
        self.best_train_loss = float('inf')
        self.gif_length = self.config['validation']['gif_length']
        self.makegif = True

    def load_config(self, config_path):
        with open(config_path) as file:
            config = yaml.safe_load(file)
        return config

    def prepare_dataloader(self):
        train_files = [self.config['data_path'] + file for file in self.config['training']['files']]
        val_files = [self.config['data_path'] + file for file in self.config['training']['files']]
        self.train_dataset, self.val_dataset = get_datasets(train_files, val_files)
        train_loader, val_loader = get_dataloaders(self.train_dataset, self.val_dataset, self.config['batch_size'])
        return train_loader, val_loader

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
    
    def make_gif(self):
        self.model.eval()
        stacked_pred = None
        stacked_true = None
    
        with torch.no_grad():  # Disable gradient tracking to save memory
            for i, (input, label) in enumerate(self.val_loader):
                print(i, self.gif_length)
                
                # Transfer input and label to the device and convert to float
                input = input.to(self.DEVICE).float()
                label = label.to(self.DEVICE).float()
    
                if i == 0:
                    stacked_pred = input[0, 0, :, :].unsqueeze(0)
                    stacked_true = input[0, 0, :, :].unsqueeze(0)
                    pred = self.model(input)  # Model prediction
                elif i < self.gif_length:
                    pred = self.model(input)
                    # Stack the predictions and ground truth
                    stacked_pred = torch.cat((stacked_pred, pred[0, 0, :, :].unsqueeze(0)), 0)
                    stacked_true = torch.cat((stacked_true, label[0, 0, :, :].unsqueeze(0)), 0)
                    print(stacked_pred.shape, stacked_true.shape)
                else:
                    print('breaking\n\n')
                    break
        create_gif2(stacked_true.cpu(), stacked_pred.cpu(), 'output/temp_log.gif', timesteps=10, vertical=False)
    
    def train(self):
        current_time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        print("Training is starting...", current_time)
        print(torch.cuda.is_available())

        for epoch in range(self.EPOCHS):
            
            
            avg_train_loss = self.one_epoch_train()
            avg_val_loss = self.one_epoch_val()

            self.make_gif()          
            

            print(f"Epoch {epoch + 1}/{self.EPOCHS} - Train Loss: {avg_train_loss:.4f}, Validation Loss: {avg_val_loss:.4f}")
            if self.wandb_enabled:
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
        if self.wandb_enabled:
            wandb.save(filename)
            wandb.finish()
