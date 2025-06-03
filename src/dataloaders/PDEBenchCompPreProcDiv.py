import torch
import h5py
from torch.utils.data import Dataset
import numpy as np
from pathlib import Path
from dataloaders.utils import spatial_resample



class PDEBenchCompPreProcDiv(Dataset):
    def __init__(self, filepaths, preproc_save_dir, resample_shape=128, resample_mode='fourier', timesample=5, dataset_name='pdebenchcomp'):
        self.data_list = []
        self.traj_list = []
        self.ts = None
        self.resample_shape = resample_shape
        self.resample_mode = resample_mode
        self.name = dataset_name
        self.vel_scale = None
        self.dt = timesample
        
        batchread = 50
        preproc_save_dir = Path(preproc_save_dir)
        preproc_save_dir.mkdir(parents=True, exist_ok=True)

        global_sum, global_sq_sum, global_count = 0.0, 0.0, 0
        traj_counter = 0

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
                        for i in range(batch.shape[0]):
                            data = batch[i]  
                            save_path = preproc_save_dir / f"traj{traj_counter:05d}.h5"
                            with h5py.File(save_path, 'w') as hf:
                                hf.create_dataset('data', data=data.numpy())
                            #print(data.shape)
                            global_sum += data.sum().item()
                            global_sq_sum += (data ** 2).sum().item()
                            global_count += data.numel()

                    self.trajectories.append(batch)
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