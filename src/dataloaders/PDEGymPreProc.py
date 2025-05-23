import torch
import h5py
from torch.utils.data import Dataset
import numpy as np
from pathlib import Path
from dataloaders.utils import spatial_resample
import netCDF4 as nc


class PDEGymPreProc(Dataset):
    def __init__(self, filepaths, preproc_savepath, resample_shape=128, resample_mode='fourier', timesample=5, dataset_name='pdegym'):
        self.data_list = []
        self.resample_shape = resample_shape
        self.resample_mode = resample_mode
        self.name = dataset_name
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