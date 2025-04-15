import torch
from torch.utils.data import Dataset
import h5py
from pathlib import Path
from dataloaders.utils import spatial_resample

class AmiraReaderFromH5(Dataset):
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
        
    def get_single_traj(self, idx):
        full = self.data[idx][::self.dt]
        #full = spatial_resample(full, self.resample_shape, self.resample_mode)
        return full
    
    def normalize_velocity(self, vel_scale):
        self.data = self.data / vel_scale
        self.vel_scale = vel_scale

    def absmax_vel(self):
        return self.data.abs().max()

class AmiraDatasetFromH5(Dataset):
    def __init__(self, reader: AmiraReaderFromH5, forward_steps = 1):
        self.reader = reader
        self.traj = reader.traj
        self.dt = reader.dt
        self.ts = reader.ts
        self.fs = forward_steps

    def __len__(self):
        return self.reader.traj * (self.reader.ts - self.reader.dt)

    def __getitem__(self, idx):
        traj_idx = idx // (self.reader.ts - self.reader.dt)
        ts_idx = idx % (self.reader.ts - self.reader.dt)
        
        front = self.reader.data[traj_idx][ts_idx]
        label = self.reader.data[traj_idx][ts_idx + self.fs * self.reader.dt]
        return front, label
