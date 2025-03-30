import torch
import h5py
from torch.utils.data import Dataset
import numpy as np
from pathlib import Path
from src.dataloaders.utils import spatial_resample

def get_dataset(folderPath):
    dir = Path(folderPath)
    assert dir.exists(), 'doesnt exist homie'
    files = list(dir.glob("*.h5"))
    return PDEBenchCompDataset(files)

class PDEBenchInCompDataset(Dataset):
    def __init__(self, filepaths, resample_shape=(256, 256), resample_mode='fourier', timesample=10):
        self.data_list = []
        self.traj_list = []
        self.ts = None
        self.resample_shape = resample_shape
        self.resample_mode = resample_mode
        
        for filepath in filepaths:
            with h5py.File(filepath, "r") as f:
                keys = list(f.keys())
                print(f"Keys in {filepath}: {keys}")
                
                if "velocity" in keys:
                    data = torch.from_numpy(f['velocity'][:,::timesample].astype(np.float32))
                    print(data.shape)
                    data = data.permute(0, 1, 4, 2, 3)  # Adjust dimensions
                    
                    if self.ts is None:
                        self.ts = data.shape[1]
                    elif self.ts != data.shape[1]:
                        raise ValueError("Mismatch in timestep dimensions across files.")
                    
                    self.data_list.append(data)
                    self.traj_list.append(data.shape[0])
                #if "t" in keys:
                #    self.data_t = torch.from_numpy(f['t'][:].astype(np.float32))
                #    print(self.data_t.shape)
        
        self.data = torch.cat(self.data_list, dim=0)
        self.traj = sum(self.traj_list)
        print(self.traj)
        print(self.ts)
        
    def __len__(self):
        return self.traj * (self.ts - 1)

    def __getitem__(self, idx):
        traj_idx = idx // (self.ts - 1)
        ts_idx = idx % (self.ts - 1)
        
        front = self.data[traj_idx][ts_idx]
        label = self.data[traj_idx][ts_idx + 1]
        front = spatial_resample(front, self.resample_shape, self.resample_mode)
        label = spatial_resample(label, self.resample_shape, self.resample_mode)
        return front.unsqueeze(0), label.unsqueeze(0)
