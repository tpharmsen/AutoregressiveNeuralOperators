import torch
import h5py
from torch.utils.data import Dataset
import numpy as np
from pathlib import Path
from dataloaders.utils import spatial_resample


class PDEBenchIncompPreProc(Dataset):
    def __init__(self, filepaths, preproc_savepath, resample_shape=128, resample_mode='fourier', timesample=5):
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
        self.avg = float(self.data.mean())
        self.std = float(self.data.std())
        # save the data in quick readable format in h5py
        with h5py.File(preproc_savepath, 'w') as f:
            f.create_dataset('data', data=self.data.numpy())
            f.create_dataset('avg', data=self.avg)
            f.create_dataset('std', data=self.std)
            f.create_dataset('resample_shape', data=self.resample_shape)
            f.create_dataset('resample_mode', data=self.resample_mode)
            f.create_dataset('timesample', data=self.dt)
            f.create_dataset('name', data=self.name)
            f.create_dataset('traj', data=self.traj)
            f.create_dataset('ts', data=self.ts)
            f.create_dataset('datashape', data=self.data.shape)