import torch
import h5py
from torch.utils.data import Dataset
import numpy as np
from pathlib import Path
from dataloaders.utils import spatial_resample
import netCDF4 as nc

import torch
import h5py
from torch.utils.data import Dataset
import numpy as np
from pathlib import Path
from dataloaders.utils import spatial_resample
import netCDF4 as nc


class PDEGymPreProcDiv(Dataset):
    def __init__(self, filepaths, preproc_save_dir, resample_shape=128, resample_mode='fourier', timesample=5, dataset_name='pdegym'):
        self.resample_shape = resample_shape
        self.resample_mode = resample_mode
        self.name = dataset_name
        self.vel_scale = None
        self.dt = timesample
        self.trajectories = []

        if preproc_save_dir.endswith('.h5'):
            preproc_save_dir = preproc_save_dir[:-3]
        print(f"Preprocessing and saving to {preproc_save_dir}")
        preproc_save_dir = Path(preproc_save_dir)
        preproc_save_dir.mkdir(parents=True, exist_ok=True)

        global_sum, global_sq_sum, global_count = 0.0, 0.0, 0
        traj_counter = 0

        for filepath in filepaths:
            with nc.Dataset(filepath, "r") as f:
                raw_data = torch.from_numpy(f['velocity'][:,:,:2])
                raw_data = spatial_resample(raw_data, self.resample_shape, self.resample_mode) 

                for i in range(raw_data.shape[0]):
                    data = raw_data[i]  
                    save_path = preproc_save_dir / f"traj{traj_counter:05d}.h5"
                    with h5py.File(save_path, 'w') as hf:
                        hf.create_dataset('data', data=data.numpy())
                    #print(data.shape)
                    global_sum += data.sum().item()
                    global_sq_sum += (data ** 2).sum().item()
                    global_count += data.numel()

                    self.trajectories.append(data)
                    traj_counter += 1

        # Compute dataset-wide stats
        avg = global_sum / global_count
        std = (global_sq_sum / global_count - avg ** 2) ** 0.5
        total_trajectories = traj_counter

        # Save metadata in a separate file
        meta_path = preproc_save_dir / 'meta.h5'
        with h5py.File(meta_path, 'w') as f:
            f.create_dataset('avg', data=avg)
            f.create_dataset('std', data=std)
            f.create_dataset('resample_shape', data=self.resample_shape)
            f.create_dataset('resample_mode', data=self.resample_mode)
            f.create_dataset('timesample', data=self.dt)
            f.create_dataset('name', data=self.name)
            f.create_dataset('traj', data=total_trajectories)
            f.create_dataset('ts', data=len(self.trajectories[0]))
            f.create_dataset('datashape', data=tuple([total_trajectories, 21,2, resample_shape, resample_shape]))

        self.total_trajectories = total_trajectories

    def __len__(self):
        return self.total_trajectories

    def __getitem__(self, idx):
        return self.trajectories[idx]