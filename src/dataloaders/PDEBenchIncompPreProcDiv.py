import torch
import h5py
from torch.utils.data import Dataset
import numpy as np
from pathlib import Path
from dataloaders.utils import spatial_resample


class PDEBenchIncompPreProcDiv(Dataset):
    def __init__(self, filepaths, preproc_save_dir, resample_shape=128, resample_mode='fourier', timesample=5, dataset_name='pdebenchincomp'):
        self.data_list = []
        self.traj_list = []
        self.ts = None
        self.resample_shape = resample_shape
        self.resample_mode = resample_mode
        self.name = dataset_name
        self.vel_scale = None
        self.dt = timesample
        
        preproc_save_dir = Path(preproc_save_dir)
        preproc_save_dir.mkdir(parents=True, exist_ok=True)

        global_sum, global_sq_sum, global_count = 0.0, 0.0, 0
        traj_counter = 0

        for filepath in filepaths:
            with h5py.File(filepath, "r") as f:
                keys = list(f.keys())
                #print(f"Keys in {filepath}: {keys}")
                
                if "velocity" in keys:
                    raw_data = torch.from_numpy(f['velocity'][:].astype(np.float32))
                    #print(data.shape)
                    raw_data = raw_data.permute(0, 1, 4, 2, 3)  
                    
                    raw_data = spatial_resample(raw_data, self.resample_shape, self.resample_mode)
                    if self.ts is None:
                        self.ts = raw_data.shape[1]
                        #print(self.ts)
                    for i in range(raw_data.shape[0]):
                        data = raw_data[i]  
                        save_path = preproc_save_dir / f"traj{traj_counter:05d}.h5"
                        with h5py.File(save_path, 'w') as hf:
                            hf.create_dataset('data', data=data.numpy())
                        print(data.shape)
                        global_sum += data.sum().item()
                        global_sq_sum += (data ** 2).sum().item()
                        global_count += data.numel()
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
            f.create_dataset('traj', data=traj_counter)
            f.create_dataset('ts', data=self.ts)
            f.create_dataset('datashape', data=tuple([traj_counter, self.ts,2, resample_shape, resample_shape]))
