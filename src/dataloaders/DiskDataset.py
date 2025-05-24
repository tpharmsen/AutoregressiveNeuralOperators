import torch
import h5py
from torch.utils.data import Dataset
import numpy as np
from pathlib import Path
from dataloaders.utils import spatial_resample

class DiskDataset(Dataset):
    def __init__(self, preproc_path, temporal_bundling = 1, forward_steps = 1):
        self.filepath = preproc_path
        self._file = None

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
        self.avgnorm = None
        self.stdnorm = None
        
    def _get_file(self):
        if self._file is None:
            self._file = h5py.File(self.filepath, 'r')
        return self._file

        
    def __len__(self):
        return self.traj * self.lenpertraj

    def __getitem__(self, idx):
        
        f = self._get_file()
        traj_idx = idx // self.lenpertraj
        ts_idx = idx % self.lenpertraj
        
        front = f['data'][traj_idx][ts_idx : ts_idx + self.idx_window : self.dt]
        label = f['data'][traj_idx][ts_idx + self.fs * self.idx_window : ts_idx + (self.fs + 1) * self.idx_window : self.dt]
        if self.avgnorm is not None:
            #print('normalising\n')
            front = (front - self.avgnorm) / self.stdnorm
            label = (label - self.avgnorm) / self.stdnorm
        return torch.tensor(front), torch.tensor(label)

    def __del__(self):
        if self._file is not None:
            self._file.close()

    def get_single_traj(self, idx):
        f = self._get_file()
        full = f['data'][idx][::self.dt]
        return torch.tensor(full)