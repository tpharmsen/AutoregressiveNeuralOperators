import torch 
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

import yaml

from modelComp.UNet import UNet2D
from trainers.simpletrainer import one_epoch_train_simple, one_epoch_val_simple
from dataloaders.SimpleLoaderBubbleML import SimpleLoaderBubbleML


def load_config(config_path):
    with open(config_path) as file:
        config = yaml.safe_load(file)
    return config

def prepare_dataloader(config):
    # blabla
    #return SimpleLoaderBubbleML()
    pass

def train(config_path):

    config = load_config(config_path)
    EPOCHS = config['training']['epochs']
    DEVICE = torch.device(config["device"])

    #train_loader, val_loader = prepare_dataloader(config)

    model = UNet2D(in_channels=3, out_channels=3, features=[64, 128, 256, 512]).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=config['training']['learning_rate'])
    
    train_losses = []
    val_losses = []
    '''
    for epoch in range(EPOCHS):
        avg_train_loss = one_epoch_train_simple(model, train_loader, optimizer, DEVICE)
        train_losses.append(avg_train_loss)
    
        avg_val_loss = one_epoch_val_simple(model, val_loader, DEVICE)
        val_losses.append(avg_val_loss)
        
        print(f"Epoch {epoch + 1}/{EPOCHS} - Train Loss: {avg_train_loss:.4f}, Validation Loss: {avg_val_loss:.4f}")
    '''

    print(EPOCHS, DEVICE)
    print("Training is done!")


if __name__ == "__main__":
    train("../conf/example.yaml")