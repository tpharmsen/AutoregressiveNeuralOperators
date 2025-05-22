import torch
import h5py
from torch.utils.data import Dataset
import numpy as np
from pathlib import Path
from dataloaders.utils import spatial_resample

class DiskDataset(Dataset):
    def __init__(self, preproc_path, temporal_bundling = 1, forward_steps = 1):
        self.filepath = preproc_path
        with h5py.File(self.filepath, 'r') as f:
            self.avg = float(f['avg'][()])
            self.std = float(f['std'][()])
            self.resample_shape = int(f['resample_shape'][()])
            self.resample_mode = str(f['resample_mode'][()])
            self.dt = int(f['timesample'][()])
            self.name = str(f['name'][()])
            self.traj = int(f['traj'][()])
            self.ts = int(f['ts'][()])
            self.datashape = tuple(f['datashape'][()])
        self.tb = temporal_bundling
        self.fs = forward_steps
        self.lenpertraj = self.ts - (1 + self.fs) * self.dt * self.tb + self.dt
        self.idx_window = self.dt * self.tb
        
    def __len__(self):
        return self.traj * self.lenpertraj

    def __getitem__(self, idx):
        traj_idx = idx // self.lenpertraj
        ts_idx = idx % self.lenpertraj
        
        with h5py.File(self.filepath, 'r') as f:
            front = f['data'][traj_idx][ts_idx : ts_idx + self.idx_window : self.dt]
            label = f['data'][traj_idx][ts_idx + self.fs * self.idx_window : ts_idx + (self.fs + 1) * self.idx_window : self.dt]

        return torch.tensor(front), torch.tensor(label)