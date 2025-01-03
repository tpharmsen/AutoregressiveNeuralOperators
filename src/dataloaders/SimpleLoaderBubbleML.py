import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, ConcatDataset, DataLoader
import h5py


TEMPERATURE = 'temperature'
VELX = 'velx'
VELY = 'vely'
PRESSURE = 'pressure'

def get_dataloaders(train_files, val_files, batch_size):
    train_dataset = ConcatDataset((SimpleLoaderBubbleML(file) for file in train_files))
    val_dataset = ConcatDataset((SimpleLoaderBubbleML(file) for file in val_files))
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    return train_loader, val_loader

class SimpleLoaderBubbleML(Dataset):
    def __init__(self, filename):
        self.filename = filename
        self.data = h5py.File(self.filename, 'r')
        self.timesteps = self.data[TEMPERATURE][:].shape[0]
    
    def __len__(self):
        return self.timesteps - 1

    def _get_state(self, idx):
        
        temp = torch.from_numpy(self.data[TEMPERATURE][idx])
        velx = torch.from_numpy(self.data[VELX][idx])
        vely = torch.from_numpy(self.data[VELY][idx])
        
        return torch.stack((temp, velx, vely), dim=0)
    
    def __getitem__(self, idx):
        
        input = self._get_state(idx)
        label = self._get_state(idx+1)
        return input, label
    
    def get_full_stack(self):
        
        temp_data = torch.from_numpy(self.data[TEMPERATURE][:]) 
        velx_data = torch.from_numpy(self.data[VELX][:])        
        vely_data = torch.from_numpy(self.data[VELY][:])         
        
        full_stack = torch.stack((temp_data, velx_data, vely_data), dim=1) 
        return full_stack

    
