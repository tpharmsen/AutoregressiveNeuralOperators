import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, ConcatDataset, DataLoader
import h5py


TEMPERATURE = 'temperature'
VELX = 'velx'
VELY = 'vely'
PRESSURE = 'pressure'

def get_datasets(train_files, val_files, discard_first):
    train_dataset = ConcatDataset((SimpleLoaderBubbleML(file, discard_first) for file in train_files))
    val_dataset = ConcatDataset((SimpleLoaderBubbleML(file, discard_first) for file in val_files))
    return train_dataset, val_dataset

def get_dataloaders(train_dataset, val_dataset, batch_size):
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)
    return train_loader, val_loader

class SimpleLoaderBubbleML(Dataset):
    def __init__(self, filename, discard_first):
        self.filename = filename
        self.discard_first = discard_first
        self.data = h5py.File(self.filename, 'r')
        self.timesteps = self.data[TEMPERATURE][self.discard_first:].shape[0]
    
    def __len__(self):
        return self.timesteps - 1 - self.discard_first

    def _get_state(self, idx):
        
        temp = torch.from_numpy(self.data[TEMPERATURE][idx+self.discard_first])
        velx = torch.from_numpy(self.data[VELX][idx+self.discard_first])
        vely = torch.from_numpy(self.data[VELY][idx+self.discard_first])
        
        return torch.stack((temp, velx, vely), dim=0)
    
    def __getitem__(self, idx):
        
        input = self._get_state(idx)
        label = self._get_state(idx+1)
        return input, label
    
    def get_full_stack(self):
        
        temp_data = torch.from_numpy(self.data[TEMPERATURE][self.discard_first:]) 
        velx_data = torch.from_numpy(self.data[VELX][self.discard_first:])        
        vely_data = torch.from_numpy(self.data[VELY][self.discard_first:])         
        
        full_stack = torch.stack((temp_data, velx_data, vely_data), dim=1) 
        return full_stack

    
