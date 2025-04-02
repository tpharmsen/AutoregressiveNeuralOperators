import torch
from torch.utils.data import Dataset
import netCDF4 as nc
from dataloaders.utils import spatial_resample


class PDEGymDataset(Dataset):
    def __init__(self, filepaths, resample_shape=(256, 256), resample_mode='fourier', timesample=1):

        self.data = []
        self.resample_shape = resample_shape
        self.resample_mode = resample_mode
        for filepath in filepaths:
            with nc.Dataset(filepath, "r") as f:
                velocity = torch.from_numpy(f['velocity'][:,::timesample,:2])  # remove passive tracer
                self.data.append(velocity)  

        self.data = torch.cat(self.data, dim=0) 
        self.traj = self.data.shape[0]
        self.ts = self.data.shape[1]
        #print(self.ts)

    def __len__(self):
        return self.traj * (self.ts - 1)

    def __getitem__(self, idx):
        traj_idx = idx // (self.ts - 1)
        ts_idx = idx % (self.ts - 1)

        front = self.data[traj_idx][ts_idx]
        label = self.data[traj_idx][ts_idx + 1]
        front = spatial_resample(front, self.resample_shape, self.resample_mode)
        label = spatial_resample(label, self.resample_shape, self.resample_mode)
        return front, label #front.unsqueeze(0), label.unsqueeze(0)
        
    def get_single_traj(self, idx):
        full = self.data[idx]
        full = spatial_resample(full, self.resample_shape, self.resample_mode)
        return full