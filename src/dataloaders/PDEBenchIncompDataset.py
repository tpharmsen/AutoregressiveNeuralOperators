import torch
import h5py
from torch.utils.data import Dataset
import numpy as np
from pathlib import Path
from dataloaders.utils import spatial_resample


class PDEBenchIncompReader(Dataset):
    def __init__(self, filepaths, resample_shape=128, resample_mode='fourier', timesample=10):
        self.data_list = []
        self.traj_list = []
        self.ts = None
        self.resample_shape = resample_shape
        self.resample_mode = resample_mode
        self.name = None
        self.vel_scale = None
        self.dt = timesample
        
        for filepath in filepaths:
            with h5py.File(filepath, "r") as f:
                keys = list(f.keys())
                #print(f"Keys in {filepath}: {keys}")
                
                if "velocity" in keys:
                    data = torch.from_numpy(f['velocity'][:].astype(np.float32))
                    #print(data.shape)
                    data = data.permute(0, 1, 4, 2, 3)  
                    
                    data = spatial_resample(data, self.resample_shape, self.resample_mode)
                    if self.ts is None:
                        self.ts = data.shape[1]
                    elif self.ts != data.shape[1]:
                        raise ValueError("Mismatch in timestep dimensions across files.")
                    
                    self.data_list.append(data)
                    self.traj_list.append(data.shape[0])
        
        self.data = torch.cat(self.data_list, dim=0)
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

class PDEBenchIncompDataset(Dataset):
    def __init__(self, reader: PDEBenchIncompReader, temporal_bundling = 1, forward_steps = 1):
        self.reader = reader
        self.traj = reader.traj
        self.dt = reader.dt
        self.ts = reader.ts
        self.tb = temporal_bundling
        self.fs = forward_steps
        self.lenpertraj = self.reader.ts - (1 + self.fs) * self.reader.dt * self.tb + self.reader.dt
        self.idx_window = self.reader.dt * self.tb

    def __len__(self):
        return self.reader.traj * (self.reader.ts - self.reader.dt)

    def __len__(self):
        return self.reader.traj * self.lenpertraj

    def __getitem__(self, idx):
        traj_idx = idx // self.lenpertraj
        ts_idx = idx % self.lenpertraj
        
        front = self.reader.data[traj_idx][ts_idx : ts_idx + self.idx_window : self.reader.dt]
        label = self.reader.data[traj_idx][ts_idx + self.fs * self.idx_window : ts_idx + (self.fs + 1) * self.idx_window : self.reader.dt]
        return front, label
