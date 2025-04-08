import torch
from torch.utils.data import Dataset
import h5py
from pathlib import Path
from dataloaders.utils import spatial_resample

class AmiraDatasetFromH5(Dataset):
    def __init__(self, filepaths, resample_shape=128, resample_mode='fourier', timesample=5):
        self.data_list = []
        self.traj_list = []
        self.ts = None
        self.resample_shape = resample_shape
        self.resample_mode = resample_mode
        self.name = None
        self.vel_scale = None
        self.dt = timesample
        
        for filepath in filepaths:
            with h5py.File(filepath, 'r') as f:
                data = torch.from_numpy(f['velocity'][:])
                data = data.permute(0,3,1,2)
                data = spatial_resample(data, self.resample_shape, self.resample_mode)
                self.data_list.append(data)
                self.traj_list.append(torch.tensor(1))
                if self.ts is None:
                    self.ts = data.shape[0]
        
        self.data = torch.stack(self.data_list, dim=0)
        self.traj = sum(self.traj_list)

    def __len__(self):
        return self.traj * (self.ts - self.dt)

    def __getitem__(self, idx):
        traj_idx = idx // (self.ts - self.dt)
        ts_idx = idx % (self.ts - self.dt)
        
        front = self.data[traj_idx][ts_idx]
        label = self.data[traj_idx][ts_idx + self.dt]

        #front = spatial_resample(front, self.resample_shape, mode=self.resample_mode)
        #label = spatial_resample(label, self.resample_shape, mode=self.resample_mode)
        return front, label #front.unsqueeze(0), label.unsqueeze(0)
    
        
    def get_single_traj(self, idx):
        full = self.data[idx][::self.dt]
        #full = spatial_resample(full, self.resample_shape, self.resample_mode)
        return full
    
    def normalize_velocity(self, vel_scale):
        self.data = self.data / vel_scale
        self.vel_scale = vel_scale

    def absmax_vel(self):
        return self.data.abs().max()