import torch
from torch.utils.data import Dataset
import h5py
import numpy as np


class PDEBenchCompDataset(Dataset):
    def __init__(self, filepaths):
        self.data_list = []
        self.traj_list = []
        self.ts = None

        for filepath in filepaths:
            with h5py.File(filepath, "r") as f:
                keys = list(f.keys())
                print(f"Keys in {filepath}: {keys}")
                
                if "Vx" in keys and "Vy" in keys:
                    data = torch.from_numpy(
                        np.stack((f["Vx"][:], f["Vy"][:]), axis=2).astype(np.float32)
                    )
                    
                    if self.ts is None:
                        self.ts = data.shape[1]
                    elif self.ts != data.shape[1]:
                        raise ValueError("Mismatch in timestep dimensions across files.")
                    
                    self.data_list.append(data)
                    self.traj_list.append(data.shape[0])
        
        self.data = torch.cat(self.data_list, dim=0)
        self.traj = sum(self.traj_list)
        
    def __len__(self):
        return self.traj * (self.ts - 1)

    def __getitem__(self, idx):
        traj_idx = idx // (self.ts - 1)
        ts_idx = idx % (self.ts - 1)
        
        return self.data[traj_idx][ts_idx].unsqueeze(0), self.data[traj_idx][ts_idx + 1].unsqueeze(0)
