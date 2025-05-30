import torch
import h5py
from torch.utils.data import Dataset
import numpy as np
from pathlib import Path
from dataloaders.utils import spatial_resample


class AmiraPreProc(Dataset):
    def __init__(self, filepaths, preproc_savepath, resample_shape=128, resample_mode='fourier', timesample=5, dataset_name='amira'):
        self.data_list = []
        self.traj_list = []
        self.ts = None
        self.resample_shape = resample_shape
        self.resample_mode = resample_mode
        self.name = dataset_name
        self.vel_scale = None
        self.dt = timesample
        
        for filepath in filepaths:
            #print(filepath)
            data = torch.from_numpy(self.read_amira_binary_mesh(filepath).copy())
            data = data.permute(0,3,1,2)
            data = spatial_resample(data, self.resample_shape, self.resample_mode)
            self.data_list.append(data)
            self.traj_list.append(torch.tensor(1))
            if self.ts is None:
                self.ts = data.shape[0]
        
        self.data = torch.stack(self.data_list, dim=0)
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
    
    