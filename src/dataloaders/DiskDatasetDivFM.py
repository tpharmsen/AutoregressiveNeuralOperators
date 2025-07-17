import torch
import h5py
from torch.utils.data import Dataset
import numpy as np
from pathlib import Path
from dataloaders.utils import spatial_resample

from scipy.ndimage import gaussian_filter

class DiskDatasetDivFM(Dataset):
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
            target = f['data'][ts_idx : ts_idx + self.idx_window : self.dt]
            #label = f['data'][ts_idx + self.fs * self.idx_window : ts_idx + (self.fs + 1) * self.idx_window : self.dt]
        if self.avgnorm is not None:
            #print('normalising\n')
            target = (target - self.avgnorm) / self.stdnorm
        #print(target.shape)
        prior = self.prior_prefix(target, fromframe=4, sigma=4.5, scale=0.8)

        #label = (label - self.avgnorm) / self.stdnorm
        return torch.tensor(prior, dtype=torch.float32), torch.tensor(target, dtype=torch.float32)
    
    def prior_prefix(self, x, fromframe=3, sigma=4.0, scale=1):
            xnoise = x.copy()
            #print('function called')
            sigma = (0, sigma, sigma)
            for i in range(fromframe, xnoise.shape[0]):
                noise = xnoise[i-1] + scale * np.random.normal(size = xnoise[i].shape) #* torch.randn_like(xnoise[i])
                #print(noise.shape)
                noise = gaussian_filter(noise, sigma=sigma)
                xnoise[i, :] = noise
            return xnoise

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