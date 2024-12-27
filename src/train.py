import torch 
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import datetime
import wandb

import yaml

from modelComp.UNet import UNet2D
from trainers.simpletrainer import one_epoch_train_simple, one_epoch_val_simple
from dataloaders.SimpleLoaderBubbleML import SimpleLoaderBubbleML, get_dataloaders


def load_config(config_path):
    with open(config_path) as file:
        config = yaml.safe_load(file)
    return config

def prepare_dataloader(config):
    train_files = [config['data_path'] + file for file in config['training']['files']]
    val_files = [config['data_path'] + file for file in config['training']['files']]
    train_loader, val_loader = get_dataloaders(train_files, val_files, config['batch_size'])
    return train_loader, val_loader

def train(config_path):
    current_time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    print("Training is starting...:", current_time)
    print(torch.cuda.is_available())
    
    config = load_config(config_path)
    EPOCHS = config['training']['epochs']
    DEVICE = torch.device(config["device"])

    train_loader, val_loader = prepare_dataloader(config)

    model = UNet2D(in_channels=3, out_channels=3, features=[64, 128, 256, 512]).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=config['training']['learning_rate'])
    
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    best_train_loss = float('inf')
    modelname = config['modelname']
    print("Model name:", modelname)
    
    wandb.init(project="bubbleml_DS", name=modelname)
    wandb.config.update(config)

    for epoch in range(EPOCHS):
        avg_train_loss = one_epoch_train_simple(model, train_loader, optimizer, DEVICE)
        train_losses.append(avg_train_loss)

        avg_val_loss = one_epoch_val_simple(model, val_loader, DEVICE)
        val_losses.append(avg_val_loss)

        print(f"Epoch {epoch + 1}/{EPOCHS} - Train Loss: {avg_train_loss:.4f}, Validation Loss: {avg_val_loss:.4f}")

        wandb.log({"epoch": epoch, 
                   "train_loss": avg_train_loss, 
                   "val_loss": avg_val_loss})
        
        if epoch % 10 == 0:
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                best_model_path = f"models/best_model_train_{modelname}.pth"
                torch.save(model.state_dict(), best_model_path)
            if avg_train_loss < best_train_loss:
                best_train_loss = avg_train_loss
                best_model_path = f"models/best_model_epoch_train_{modelname}.pth"
                torch.save(model.state_dict(), best_model_path)

    filename = f"models/finalmodel_{modelname}.pth"
    torch.save(model.state_dict(), filename)
    print("Training is finished. Model is saved as:", filename)

    wandb.save(filename)
    wandb.save(best_model_path)
    wandb.finish()

if __name__ == "__main__":
    train("conf/example.yaml")