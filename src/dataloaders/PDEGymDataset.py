import torch
from torch.utils.data import Dataset
import netCDF4 as nc
from dataloaders.utils import spatial_resample

class PDEGymReader:
    def __init__(self, filepaths, resample_shape=128, resample_mode='fourier', timesample=1):

        self.data_list = []
        self.resample_shape = resample_shape
        self.resample_mode = resample_mode
        self.name = None
        self.vel_scale = None
        self.dt = timesample
        
        for filepath in filepaths:
            with nc.Dataset(filepath, "r") as f:
                data = torch.from_numpy(f['velocity'][:,:,:2])  # remove passive tracer
                data = spatial_resample(data, self.resample_shape, self.resample_mode)
                self.data_list.append(data)  

        self.data = torch.cat(self.data_list, dim=0) 
        self.traj = self.data.shape[0]
        self.ts = self.data.shape[1]
        
    def get_single_traj(self, idx):
        full = self.data[idx][::self.dt]
        #full = spatial_resample(full, self.resample_shape, self.resample_mode)
        return full
    
    def normalize_velocity(self, vel_scale):
        self.data = self.data / vel_scale
        self.vel_scale = vel_scale

    def absmax_vel(self):
        return self.data.abs().max()

class PDEGymDataset(Dataset):
    def __init__(self, reader: PDEGymReader, temporal_bundling = 1, forward_steps = 1):
        self.reader = reader
        self.traj = reader.traj
        self.dt = reader.dt
        self.ts = reader.ts
        self.tb = temporal_bundling
        self.fs = forward_steps
        self.lenpertraj = self.reader.ts - (1 + self.fs) * self.reader.dt * self.tb + self.reader.dt
        self.idx_window = self.reader.dt * self.tb

    def __len__(self):
        return self.reader.traj * self.lenpertraj

    def __getitem__(self, idx):
        traj_idx = idx // self.lenpertraj
        ts_idx = idx % self.lenpertraj
        
        front = self.reader.data[traj_idx][ts_idx : ts_idx + self.idx_window : self.reader.dt]
        label = self.reader.data[traj_idx][ts_idx + self.fs * self.idx_window : ts_idx + (self.fs + 1) * self.idx_window : self.reader.dt]
        return front, label
