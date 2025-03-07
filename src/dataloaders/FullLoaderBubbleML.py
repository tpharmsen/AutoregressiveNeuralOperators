import torch
from torch.utils.data import Dataset, DataLoader, SequentialSampler, RandomSampler, BatchSampler
from torch.utils.data.distributed import DistributedSampler

import h5py

TEMPERATURE = 'temperature'
VELX = 'velx'
VELY = 'vely'
PRESSURE = 'pressure'

def get_datasets(train_files, val_files, discard_first, norm):
    train_dataset = FullLoaderBubbleML(train_files, discard_first)
    val_dataset = FullLoaderBubbleML(val_files, discard_first)
    train_dataset.tres = train_dataset.get_time_length()
    if norm:
        max_temp = max(train_dataset.get_max_temperature(), val_dataset.get_max_temperature())
        max_vel = max(train_dataset.get_max_velocity(), val_dataset.get_max_velocity())
        min_temp = min(train_dataset.get_min_temperature(), val_dataset.get_min_temperature())
        min_vel = min(train_dataset.get_min_velocity(), val_dataset.get_min_velocity())
        
        train_dataset.set_normalization_params(min_temp, max_temp, min_vel, max_vel)
        val_dataset.set_normalization_params(min_temp, max_temp, min_vel, max_vel)

    return train_dataset, val_dataset

def get_dataloaders(train_dataset, val_dataset, train_batch_size, train_shuffle=False, distributed=True):
    '''
    train_sampler = RandomSampler(train_dataset, replacement=True, num_samples=train_batch_size)
    train_batch_sampler = BatchSampler(train_sampler, batch_size=train_batch_size, drop_last=False)
    train_loader = DataLoader(train_dataset, batch_sampler=train_batch_sampler)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)
    return train_loader, val_loader
    '''
    
    train_sampler = RandomSampler(train_dataset) if train_shuffle else SequentialSampler(train_dataset)
    val_sampler = SequentialSampler(val_dataset)

    train_loader = DataLoader(
        train_dataset, 
        batch_size=train_batch_size, 
        sampler=train_sampler, 
        drop_last=False
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=train_batch_size, 
        sampler=val_sampler, 
        drop_last=False
    )
    return train_loader, val_loader

class FullLoaderBubbleML(Dataset):
    def __init__(self, files, discard_first):
        self.files = files
        self.discard_first = discard_first
        self.min_temp = None
        self.max_temp = None
        self.min_vel = None
        self.max_vel = None
        self.tlen = self.get_time_length()

    def __len__(self):
        #print(len(self.files))
        return len(self.files) #* (self.tlen - self.discard_first)

    def __getitem__(self, idx):
        #print(f"Loading file {idx}")
        file = self.files[idx]
        with h5py.File(file, 'r') as filedata:
            temp = torch.from_numpy(filedata[TEMPERATURE][self.discard_first:]).float()
            velx = torch.from_numpy(filedata[VELX][self.discard_first:]).float()
            vely = torch.from_numpy(filedata[VELY][self.discard_first:]).float()
        
        if self.min_temp is not None and self.max_temp is not None:
            temp = self.normalize(temp, self.min_temp, self.max_temp)
        if self.min_vel is not None and self.max_vel is not None:
            velx = self.normalize(velx, self.min_vel, self.max_vel)
            vely = self.normalize(vely, self.min_vel, self.max_vel)

        data = torch.stack((temp, velx, vely), dim=1)
        return data
    
    def set_normalization_params(self, min_temp, max_temp, min_vel, max_vel):
        """Set normalization parameters."""
        self.min_temp = min_temp
        self.max_temp = max_temp
        self.min_vel = min_vel
        self.max_vel = max_vel

    @staticmethod
    def normalize(data, min_val, max_val):
        """Normalize data to the range [-1, 1]."""
        return 2 * (data - min_val) / (max_val - min_val) - 1

    def get_time_length(self):
            file = self.files[0]
            with h5py.File(file, 'r') as data:
                tres = data[TEMPERATURE][self.discard_first:].shape[0]  # Time dimension size
            return tres
    
    def get_max_temperature(self):
        max_temp = float('-inf')
        for file in self.files:
            with h5py.File(file, 'r') as data:
                temp_max = data[TEMPERATURE][self.discard_first:].max()
                max_temp = max(max_temp, temp_max)
        return max_temp
    
    def get_max_velocity(self):
        max_vel = float('-inf')
        for file in self.files:
            with h5py.File(file, 'r') as data:
                velx_max = data[VELX][self.discard_first:].max()
                vely_max = data[VELY][self.discard_first:].max()
                max_vel = max(max_vel, velx_max, vely_max)
        return max_vel
    
    def get_min_temperature(self):
        min_temp = float('inf')
        for file in self.files:
            with h5py.File(file, 'r') as data:
                temp_min = data[TEMPERATURE][self.discard_first:].min()
                min_temp = min(min_temp, temp_min)
        return min_temp
    
    def get_min_velocity(self):
        min_vel = float('inf')
        for file in self.files:
            with h5py.File(file, 'r') as data:
                velx_min = data[VELX][self.discard_first:].min()
                vely_min = data[VELY][self.discard_first:].min()
                min_vel = min(min_vel, velx_min, vely_min)
        return min_vel
    def _get_coords(self, timestep):
        with h5py.File(self.files[0], 'r') as data:
            x = torch.from_numpy(data['x'][self.discard_first:]).float()
            x /= x.max()
            y = torch.from_numpy(data['y'][self.discard_first:]).float()
            y /= y.max()
        coords = torch.stack([x, y], dim=0)
        return coords