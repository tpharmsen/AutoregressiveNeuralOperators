import torch
from torch.utils.data import Dataset
import h5py
from pathlib import Path


class AmiraDataset(Dataset):
    def __init__(self, filepaths):
        self.data_list = []
        self.lengths = []
        
        
        for filepath in filepaths:
            with h5py.File(filepath, 'r') as f:
                data = torch.from_numpy(f['velocity'][:])
                self.data_list.append(data)
                self.lengths.append(len(data) - 1)
        
        self.cumulative_lengths = torch.cumsum(torch.tensor(self.lengths), dim=0)

    def __len__(self):
        return sum(self.lengths)

    def __getitem__(self, idx):
        # Determine which file the index belongs to
        file_idx = next(i for i, cl in enumerate(self.cumulative_lengths) if idx < cl)
        local_idx = idx if file_idx == 0 else idx - self.cumulative_lengths[file_idx - 1]
        
        data = self.data_list[file_idx][local_idx]
        label = self.data_list[file_idx][local_idx + 1]
        
        return data.unsqueeze(0), label.unsqueeze(0)
