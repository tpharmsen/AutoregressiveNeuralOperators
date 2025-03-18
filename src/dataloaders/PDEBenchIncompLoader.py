import torch
import h5py
from torch.utils.data import Dataset

class PDEBenchDatasetINCOMP(Dataset):
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
            if "velocity" in keys:
                self.data = torch.from_numpy(f['velocity'][:].astype(np.float32))
                print(self.data.shape)
                self.data = self.data.permute(0,1,4,2,3)
            #print(torch.from_numpy(f['force'][:].astype(np.float32)).shape)
            #print(torch.from_numpy(f['particles'][:].astype(np.float32)).shape)
            #print(torch.from_numpy(f['t'][:].astype(np.float32)).shape)
            self.traj, self.ts = self.data.shape[0], self.data.shape[1]
                

    def __len__(self):
        return self.traj * (self.ts - 1)

    def __getitem__(self, idx):
        traj_idx = idx // (self.ts - 1)
        ts_idx = idx % (self.ts - 1)

        return self.data[traj_idx][ts_idx].unsqueeze(0), self.data[traj_idx][ts_idx].unsqueeze(0)

