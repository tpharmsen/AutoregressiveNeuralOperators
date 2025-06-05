import torch
import h5py
from torch.utils.data import Dataset
import numpy as np
from pathlib import Path
from dataloaders.utils import spatial_resample

class DiskDatasetDiv(Dataset):
    def __init__(self, preproc_path, temporal_bundling = 1, forward_steps = 1):
        self.filepath = preproc_path
        self._file = None
        # If the file ends with .h5, remove it
        if self.filepath.endswith('.h5'):
            self.filepath = self.filepath[:-3]
        metafile = Path(self.filepath)
        metafile = metafile / 'meta.h5'
        #check if metafile exists
        if not metafile.exists():
            raise FileNotFoundError(f"Metadata file {metafile} does not exist. Please preprocess the data first.")
        with h5py.File(metafile, 'r') as f:
            self.avg = float(f['avg'][()])
            self.std = float(f['std'][()])
            self.resample_shape = int(f['resample_shape'][()])
            self.resample_mode = str(f['resample_mode'][()].decode('utf-8'))
            self.dt = int(f['timesample'][()])
            self.name = str(f['name'][()].decode('utf-8'))
            self.traj = int(f['traj'][()])
            self.ts = int(f['ts'][()])
            self.datashape = tuple(f['datashape'][()])
        #print(f"Dataset {self.name} loaded with {self.traj} trajectories, each with {self.ts} time steps.")
        #print(f"reshape method: {self.resample_mode}, shape: {self.resample_shape}")
        self.tb = temporal_bundling
        self.fs = forward_steps
        self.lenpertraj = self.ts - (1 + self.fs) * self.dt * self.tb + self.dt
        self.idx_window = self.dt * self.tb
        self.avgnorm = None
        self.stdnorm = None
    """    
    def _get_file(self):
        if self._file is None:
            self._file = h5py.File(self.filepath, 'r')
        return self._file
    """
        
    def __len__(self):
        return self.traj * self.lenpertraj

    def __getitem__(self, idx):
        
        #f = self._get_file()
        traj_idx = idx // self.lenpertraj
        ts_idx = idx % self.lenpertraj
        filename = self.filepath
        # if ends with .h5, remove it
        filename = Path(filename)
        filename = filename / f'traj{traj_idx:05d}.h5'
        with h5py.File(filename, 'r') as f:
            front = f['data'][ts_idx : ts_idx + self.idx_window : self.dt]
            label = f['data'][ts_idx + self.fs * self.idx_window : ts_idx + (self.fs + 1) * self.idx_window : self.dt]
        if self.avgnorm is not None:
            #print('normalising\n')
            front = (front - self.avgnorm) / self.stdnorm
            label = (label - self.avgnorm) / self.stdnorm
        return torch.tensor(front, dtype=torch.float32), torch.tensor(label, dtype=torch.float32)
    """
    def __del__(self):
        if self._file is not None:
            self._file.close()
    """

    def get_single_traj(self, idx):
        #f = self._get_file()
        filename = self.filepath
        filename = Path(filename)
        filename = filename / f'traj{idx:05d}.h5'
        with h5py.File(filename, 'r') as f:
            full = f['data'][::self.dt]
        if self.avgnorm is not None:
            full = (full - self.avgnorm) / self.stdnorm
        return torch.tensor(full, dtype=torch.float32)