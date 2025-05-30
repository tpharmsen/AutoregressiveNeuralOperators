import torch
import h5py
from torch.utils.data import Dataset
import numpy as np
from pathlib import Path
from dataloaders.utils import spatial_resample


class AmiraPreProcDiv(Dataset):
    def __init__(self, filepaths, preproc_save_dir, resample_shape=128, resample_mode='fourier', timesample=5, dataset_name='amira'):
        #self.data_list = []
        #self.traj_list = []
        self.ts = None
        self.resample_shape = resample_shape
        self.resample_mode = resample_mode
        self.name = dataset_name
        self.vel_scale = None
        self.dt = timesample

        if preproc_save_dir.endswith('.h5'):
            preproc_save_dir = preproc_save_dir[:-3]
        print(f"Preprocessing and saving to {preproc_save_dir}")
        preproc_save_dir = Path(preproc_save_dir)
        preproc_save_dir.mkdir(parents=True, exist_ok=True)

        global_sum, global_sq_sum, global_count = 0.0, 0.0, 0
        traj_counter = 0
        
        for filepath in filepaths:
            #print(filepath)
            raw_data = torch.from_numpy(self.read_amira_binary_mesh(filepath).copy())
            raw_data = raw_data.permute(0,3,1,2)
            raw_data = spatial_resample(raw_data, self.resample_shape, self.resample_mode)

            save_path = Path(preproc_save_dir) / f"traj{traj_counter:05d}.h5"
            with h5py.File(save_path, 'w') as hf:
                hf.create_dataset('data', data=raw_data.numpy())
            global_sum += raw_data.sum().item()
            global_sq_sum += (raw_data ** 2).sum().item()
            global_count += raw_data.numel()
            if self.ts is None:
                self.ts = raw_data.shape[0]
            traj_counter += 1
        
        avg = global_sum / global_count
        std = (global_sq_sum / global_count - avg ** 2) ** 0.5
        total_trajectories = traj_counter

        meta_path = preproc_save_dir / 'meta.h5'
        with h5py.File(meta_path, 'w') as f:
            f.create_dataset('avg', data=avg)
            f.create_dataset('std', data=std)
            f.create_dataset('resample_shape', data=self.resample_shape)
            f.create_dataset('resample_mode', data=self.resample_mode)
            f.create_dataset('timesample', data=self.dt)
            f.create_dataset('name', data=self.name)
            f.create_dataset('traj', data=total_trajectories)
            f.create_dataset('ts', data=self.ts)
            f.create_dataset('datashape', data=tuple([total_trajectories, self.ts, 2, self.resample_shape, self.resample_shape]))

    def read_amira_binary_mesh(self, filename):
        with open(filename, 'rb') as f:
            raw_data = f.read()
        # first occurrence of "@1"
        first_marker_idx = raw_data.find(b'@1')
        if first_marker_idx == -1:
            raise ValueError("Could not find binary data section in Amira file.")
        # second occurrence of "@1"
        second_marker_idx = raw_data.find(b'@1', first_marker_idx + 2)
        if second_marker_idx == -1:
            raise ValueError("Could not find second binary data section in Amira file.")
        data_start = second_marker_idx + 4  # Skip '@1\n'
        binary_data = raw_data[data_start:]
        lattice_shape = (1001, 512, 512, 2)
        float_data = np.frombuffer(binary_data, dtype=np.float32)
        float_data = float_data.reshape(lattice_shape)
        return float_data
    
    