import torch
from torch.utils.data import Dataset
import netCDF4 as nc

class PDEGymDataset(Dataset):
    def __init__(self, filepaths):
        

        self.data = []
        for filepath in filepaths:
            with nc.Dataset(filepath, "r") as f:
                velocity = torch.from_numpy(f['velocity'][:,:,:2,:,:])  # remove passive tracer
                self.data.append(velocity)  

        self.data = torch.cat(self.data, dim=0) 
        self.traj = self.data.shape[0]
        self.ts = self.data.shape[1]
        print(self.ts)

    def __len__(self):
        return self.traj * (self.ts - 1)

    def __getitem__(self, idx):
        traj_idx = idx // (self.ts - 1)
        ts_idx = idx % (self.ts - 1)

        return self.data[traj_idx][ts_idx].unsqueeze(0), self.data[traj_idx][ts_idx + 1].unsqueeze(0)