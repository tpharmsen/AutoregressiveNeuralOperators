import torch
from torch.utils.data import Dataset, DataLoader
import h5py

TEMPERATURE = 'temperature'
VELX = 'velx'
VELY = 'vely'
PRESSURE = 'pressure'

def get_dataset(files):
    dataset = FullLoaderBubbleML(files)
    dataset.tres = dataset.get_time_length()
    return dataset
def get_dataloader(dataset, batch_size=1, shuffle=False):
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    return dataloader

class FullLoaderBubbleML(Dataset):
    def __init__(self, files):
        self.files = files

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        file = self.files[idx]
        with h5py.File(file, 'r') as data:
            temp = torch.from_numpy(data[TEMPERATURE][:])
            velx = torch.from_numpy(data[VELX][:])
            vely = torch.from_numpy(data[VELY][:])
        
        data = torch.stack((temp, velx, vely), dim=1)
        return data
    
    def get_time_length(self):
            file = self.files[0]
            with h5py.File(file, 'r') as data:
                return data[TEMPERATURE].shape[0]  # Time dimension size
