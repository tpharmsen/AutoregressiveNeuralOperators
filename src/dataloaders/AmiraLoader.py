import torch
from torch.utils.data import Dataset
import h5py
from pathlib import Path
from src.dataloaders.utils import spatial_resample

def get_dataset(folderPath):
    dir = Path(folderPath)
    assert dir.exists(), 'doesnt exist homie'
    files = list(dir.glob("*.h5"))
    return AmiraDataset(files)

class AmiraDataset(Dataset):
    def __init__(self, filepaths, resample_shape=(256, 256), resample_mode='fourier'):
        self.data_list = []
        self.lengths = []
        self.resample_shape = resample_shape
        self.resample_mode = resample_mode
        
        for filepath in filepaths:
            with h5py.File(filepath, 'r') as f:
                data = torch.from_numpy(f['velocity'][:])
                data = data.permute(0,3,1,2)
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
        #print(data.shape, label.shape)
        data = spatial_resample(data, self.resample_shape, mode=self.resample_mode)
        label = spatial_resample(label, self.resample_shape, mode=self.resample_mode)
        return data.unsqueeze(0), label.unsqueeze(0)
    