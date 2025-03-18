import torch
from torch.utils.data import Dataset
import h5py
import numpy as np


class PDEBenchCompDataset(Dataset):
    def __init__(self, filepath):

        with h5py.File(filepath, "r") as f:
            keys = list(f.keys())
            print(keys)
            '''
            if "density" in keys:
                self.density = torch.from_numpy(np.array(f["density"], dtype=np.float32))
            if "pressure" in keys:
                self.pres = torch.from_numpy(np.array(f['pressure'], dtype=np.float32))
            '''
            
            self.data = torch.from_numpy(
                np.stack((f["Vx"][:], f["Vy"][:]), axis=2).astype(np.float32)
            )
            self.traj, self.ts = self.data.shape[0], self.data.shape[1]
                

    def __len__(self):
        return self.traj * (self.ts - 1)

    def __getitem__(self, idx):
        traj_idx = idx // (self.ts - 1)
        ts_idx = idx % (self.ts - 1)
        
        return self.data[traj_idx][ts_idx].unsqueeze(0), self.data[traj_idx][ts_idx].unsqueeze(0)
