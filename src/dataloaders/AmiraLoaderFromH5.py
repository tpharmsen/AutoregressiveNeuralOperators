import torch
from torch.utils.data import Dataset
import h5py
from pathlib import Path
from src.dataloaders.utils import spatial_resample

class AmiraDatasetFromH5(Dataset):
    def __init__(self, filepaths, resample_shape=(256, 256), resample_mode='fourier', timesample=5):
        self.data_list = []
        self.traj_list = []
        self.ts = None
        self.resample_shape = resample_shape
        self.resample_mode = resample_mode
        
        for filepath in filepaths:
            with h5py.File(filepath, 'r') as f:
                data = torch.from_numpy(f['velocity'][::timesample])
                data = data.permute(0,3,1,2)
                self.data_list.append(data)
                self.traj_list.append(torch.tensor(1))
                if self.ts is None:
                    self.ts = data.shape[0]
        
        self.data = torch.stack(self.data_list, dim=0)
        self.traj = sum(self.traj_list)

    def __len__(self):
        return self.traj * (self.ts - 1)

    def __getitem__(self, idx):
        traj_idx = idx // (self.ts - 1)
        ts_idx = idx % (self.ts - 1)
        
        front = self.data[traj_idx][ts_idx]
        label = self.data[traj_idx][ts_idx + 1]

        front = spatial_resample(front, self.resample_shape, mode=self.resample_mode)
        label = spatial_resample(label, self.resample_shape, mode=self.resample_mode)
        return front.unsqueeze(0), label.unsqueeze(0)