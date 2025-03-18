import torch
import h5py
from torch.utils.data import Dataset
import numpy as np

class PDEBenchDatasetINCOMP(Dataset):
    def __init__(self, filepaths):
        self.data_list = []
        self.traj_list = []
        self.ts = None
        
        for filepath in filepaths:
            with h5py.File(filepath, "r") as f:
                keys = list(f.keys())
                print(f"Keys in {filepath}: {keys}")
                
                if "velocity" in keys:
                    data = torch.from_numpy(f['velocity'][:].astype(np.float32))
                    data = data.permute(0, 1, 4, 2, 3)  # Adjust dimensions
                    
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
