import torch
from torch.utils.data import Dataset
import h5py
from pathlib import Path
from dataloaders.utils import spatial_resample
import numpy as np

class AmiraReaderFromAM:
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
            #print(filepath)
            data = torch.from_numpy(self.read_amira_binary_mesh(filepath).copy())
            data = data.permute(0,3,1,2)
            data = spatial_resample(data, self.resample_shape, self.resample_mode)
            self.data_list.append(data)
            self.traj_list.append(torch.tensor(1))
            if self.ts is None:
                self.ts = data.shape[0]
        
        self.data = torch.stack(self.data_list, dim=0)
        #print(self.data.shape)
        self.traj = sum(self.traj_list)

    def read_amira_binary_mesh(self, filename):
        with open(filename, 'rb') as f:
            raw_data = f.read()
        # first occurrence of "@1"
        first_marker_idx = raw_data.find(b'@1')
        if first_marker_idx == -1:
            raise ValueError("Could not find binary data section in Amira file.")
        # second occurrence of "@1"
        second_marker_idx = raw_data.find(b'@1', first_marker_idx + 2)
        if second_marker_idx == -1:
            raise ValueError("Could not find second binary data section in Amira file.")
        data_start = second_marker_idx + 4  # Skip '@1\n'
        binary_data = raw_data[data_start:]
        lattice_shape = (1001, 512, 512, 2)
        float_data = np.frombuffer(binary_data, dtype=np.float32)
        float_data = float_data.reshape(lattice_shape)
        return float_data
        
    def get_single_traj(self, idx):
        full = self.data[idx][::self.dt]
        return full
    
    def normalize_velocity(self, vel_scale):
        self.data = self.data / vel_scale
        self.vel_scale = vel_scale

    def absmax_vel(self):
        return self.data.abs().max()

class AmiraDatasetFromAM(Dataset):
    def __init__(self, reader: AmiraReaderFromAM, temporal_bundling = 1, forward_steps = 1):
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
        
        #print(idx, traj_idx, ts_idx, self.lenpertraj, self.reader.traj, self.reader.ts, self.reader.dt, self.tb, self.fs)
        front = self.reader.data[traj_idx][ts_idx : ts_idx + self.idx_window : self.reader.dt]
        label = self.reader.data[traj_idx][ts_idx + self.fs * self.idx_window : ts_idx + (self.fs + 1) * self.idx_window : self.reader.dt]
        return front, label