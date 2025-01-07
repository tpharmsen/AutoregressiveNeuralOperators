import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, ConcatDataset, DataLoader, RandomSampler, BatchSampler
import h5py


TEMPERATURE = 'temperature'
VELX = 'velx'
VELY = 'vely'
PRESSURE = 'pressure'

def get_datasets(train_files, val_files, discard_first, norm):
    
    train_dataset = [SimpleLoaderBubbleML(file, discard_first) for file in train_files]
    val_dataset = [SimpleLoaderBubbleML(file, discard_first) for file in val_files]
    
    if norm:
        global_min_temp, global_max_temp = float('inf'), float('-inf')
        global_min_vel, global_max_vel = float('inf'), float('-inf')
        
        for dataset in train_dataset + val_dataset:
            global_min_temp = min(global_min_temp, dataset.get_min_temperature())
            global_max_temp = max(global_max_temp, dataset.get_max_temperature())
            global_min_vel = min(global_min_vel, dataset.get_min_velocity())
            global_max_vel = max(global_max_vel, dataset.get_max_velocity())
        
        for dataset in train_dataset + val_dataset:
            dataset.set_normalization_params(global_min_temp, global_max_temp, global_min_vel, global_max_vel)
    
    return ConcatDataset(train_dataset), ConcatDataset(val_dataset)

def get_dataloaders(train_dataset, val_dataset, train_batch_size, train_shuffle=False):
    train_sampler = RandomSampler(train_dataset, replacement=True, num_samples=train_batch_size)
    train_batch_sampler = BatchSampler(train_sampler, batch_size=train_batch_size, drop_last=False)
    train_loader = DataLoader(train_dataset, batch_sampler=train_batch_sampler)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)
    return train_loader, val_loader

class SimpleLoaderBubbleML(Dataset):
    def __init__(self, filename, discard_first):
        self.filename = filename
        self.discard_first = discard_first
        self.data = h5py.File(self.filename, 'r')
        self.timesteps = self.data[TEMPERATURE][self.discard_first:].shape[0]
        self.min_temp = None
        self.max_temp = None
        self.min_vel = None
        self.max_vel = None
    
    def __len__(self):
        return self.timesteps - 1 - self.discard_first

    def _get_state(self, idx):
        temp = torch.from_numpy(self.data[TEMPERATURE][idx + self.discard_first]).float()
        velx = torch.from_numpy(self.data[VELX][idx + self.discard_first]).float()
        vely = torch.from_numpy(self.data[VELY][idx + self.discard_first]).float()
        
        if self.min_temp is not None and self.max_temp is not None:
            temp = self.normalize(temp, self.min_temp, self.max_temp)
        if self.min_vel is not None and self.max_vel is not None:
            velx = self.normalize(velx, self.min_vel, self.max_vel)
            vely = self.normalize(vely, self.min_vel, self.max_vel)
        
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

    @staticmethod
    def normalize(data, min_val, max_val):
        """Normalize data to the range [-1, 1]."""
        return 2 * (data - min_val) / (max_val - min_val) - 1

    # Methods to compute min/max for temperature and velocity
    def get_max_temperature(self):
        return self.data[TEMPERATURE][self.discard_first:].max()
    
    def get_max_velocity(self):
        velx_max = self.data[VELX][self.discard_first:].max()
        vely_max = self.data[VELY][self.discard_first:].max()
        return max(velx_max, vely_max)
    
    def get_min_temperature(self):
        return self.data[TEMPERATURE][self.discard_first:].min()
    
    def get_min_velocity(self):
        velx_min = self.data[VELX][self.discard_first:].min()
        vely_min = self.data[VELY][self.discard_first:].min()
        return min(velx_min, vely_min)
