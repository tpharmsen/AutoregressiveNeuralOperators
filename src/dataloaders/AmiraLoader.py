import torch
from torch.utils.data import Dataset
import h5py

class AmiraDataset(Dataset):
    def __init__(self, filepath):

        with h5py.File(filepath, 'r') as f:
            self.data = torch.from_numpy(f['velocity'][:])
        #print(self.data.shape)

    def __len__(self):
        return len(self.data) - 1

    def __getitem__(self, idx):
        data = self.data[idx]
        label = self.data[idx+1]
        
        return data, label