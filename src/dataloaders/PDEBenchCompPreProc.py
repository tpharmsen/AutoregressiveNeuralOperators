import torch
import h5py
from torch.utils.data import Dataset
import numpy as np
from pathlib import Path
from dataloaders.utils import spatial_resample



class PDEBenchCompPreProc(Dataset):
    def __init__(self, filepaths, preproc_savepath, resample_shape=128, resample_mode='fourier', timesample=5, dataset_name='pdebenchcomp'):
        self.data_list = []
        self.traj_list = []
        self.ts = None
        self.resample_shape = resample_shape
        self.resample_mode = resample_mode
        self.name = dataset_name
        self.vel_scale = None
        self.dt = timesample
        
        batchread = 50

        for filepath in filepaths:
            with h5py.File(filepath, "r") as f:
                keys = list(f.keys())
                #print(f"Keys in {filepath}: {keys}")
                
                if "Vx" in keys and "Vy" in keys:
                    vx = f["Vx"]
                    vy = f["Vy"]
                    num_samples = vx.shape[0]
                    
                    for i in range(0, num_samples, batchread):
                        end = min(i + batchread, num_samples)

                        vx_batch = vx[i:end] 
                        vy_batch = vy[i:end] 

                        batch = np.stack((vx_batch, vy_batch), axis=2) 
                        batch = torch.from_numpy(batch.astype(np.float32)) 

                        B, T, C, H, W = batch.shape

                        batch = spatial_resample(batch, self.resample_shape, self.resample_mode)
                        #print(batch.shape)

                        if self.ts is None:
                            self.ts = batch.shape[1]

                        self.data_list.append(batch)
                        self.traj_list.append(batch.shape[0])
        
        self.data = torch.concat(self.data_list, dim=0)
        #print(self.data.shape)
        self.traj = int(sum(self.traj_list))
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