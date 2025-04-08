import torch
from pathlib import Path
import numpy as np
from torch.utils.data import Sampler

class ZeroShotSampler(Sampler):
    def __init__(self, dataset, train_ratio=0.8, split="train", seed=227):
        torch.manual_seed(seed) 
        num_train = int(dataset.traj * train_ratio)
        shuffled_trajs = torch.randperm(dataset.traj).tolist() 
        train_trajs = shuffled_trajs[:num_train]
        self.val_trajs = shuffled_trajs[num_train:]
        train_indices = [t * (dataset.ts - dataset.dt) + ts for t in train_trajs for ts in range(dataset.ts - dataset.dt)]
        val_indices = [t * (dataset.ts - dataset.dt) + ts for t in self.val_trajs for ts in range(dataset.ts - dataset.dt)]
        self.indices = train_indices if split == "train" else val_indices
    def __iter__(self):
        return iter(self.indices)
    def __len__(self):
        return len(self.indices)
    def random_val_traj(self):
        return self.val_trajs[torch.randint(0, len(self.val_trajs), (1,)).item()]

def bicubic_resample(data, target_shape):
    assert data.shape[-2] == data.shape[-1], 'Only square images supported'
    assert target_shape[0] == target_shape[1], 'Only square output images supported'
    assert data.shape[-2] != target_shape[0], 'Image already in target shape'
    if data.dim() == 3:
        data = data.unsqueeze(0)
        return torch.nn.functional.interpolate(data, size=target_shape, mode='bicubic', align_corners=False).squeeze(0)
    else:
        return torch.nn.functional.interpolate(data, size=target_shape, mode='bicubic', align_corners=False)

def fourier_resample(data, target_shape):
    assert data.dim() in (3, 4)
    
    four_dim = data.dim() == 4
    if four_dim:
        batch_size, c, x, y = data.shape
    else:
        c, x, y = data.shape
        batch_size = 1
        data = data.unsqueeze(0)

    target_x, target_y = target_shape
    assert x == y, 'Only square images supported'
    assert target_x == target_y, 'Only square images supported'
    if x == target_x:
        return data if four_dim else data.squeeze(0) 

    device = data.device
    sampled_data = []

    for b in range(batch_size):
        batch_samples = []
        for i in range(c):
            signal = data[b, i].to(dtype=torch.complex64, device=device)

            pos0 = torch.linspace(0.5, x - 0.5, x, device=device)
            pos1 = torch.linspace(0.5, target_x - 0.5, target_x, device=device)
            freq0 = torch.fft.fftfreq(x, device=device)
            freq1 = torch.fft.fftfreq(target_x, device=device)

            exp_matrix_0 = torch.exp(-2j * torch.pi * torch.outer(pos0, freq0))

            freq_coeff = torch.matmul(torch.matmul(exp_matrix_0.H, signal), exp_matrix_0) / x**2
            scaled_coeff = torch.zeros((target_x, target_x), dtype=torch.complex64, device=device)

            if target_x > x:  # Upsample
                min_idx = (target_x - x) // 2
                scaled_coeff[min_idx:min_idx + x, min_idx:min_idx + x] = torch.fft.fftshift(freq_coeff)
                scaled_coeff = torch.fft.ifftshift(scaled_coeff)
            else:  # Downsample
                min_idx = (x - target_x) // 2
                scaled_coeff = torch.fft.fftshift(freq_coeff)[min_idx:min_idx + target_x, min_idx:min_idx + target_x]
                scaled_coeff = torch.fft.ifftshift(scaled_coeff)

            exp_matrix_1 = torch.exp(2j * torch.pi * torch.outer(freq1, pos1))
            scaled_signal = torch.matmul(torch.matmul(exp_matrix_1.H, scaled_coeff), exp_matrix_1)
            batch_samples.append(scaled_signal.real)

        sampled_data.append(torch.stack(batch_samples, dim=0))

    scaled_signal = torch.stack(sampled_data, dim=0)
    return scaled_signal if four_dim else scaled_signal.squeeze(0)


def spatial_resample(data, target_shape, mode='bicubic'):
    #print(data.shape, target_shape)
    if data.shape[-1] == target_shape and data.shape[-2] == target_shape:
        return data
    if mode == 'bicubic':
        return bicubic_resample(data, (target_shape, target_shape))
    elif mode == 'fourier':
        return fourier_resample(data, (target_shape, target_shape))
    else:
        raise ValueError(f'Unknown mode: {mode}')

def get_dataset(dataset_obj, folderPath, file_ext, resample_shape, resample_mode, timesample):
    subdir = Path(folderPath)
    assert subdir.exists(), 'subdir doesnt exist'
    files = list(subdir.glob("*." + str(file_ext)))
    #print(files)
    return dataset_obj(files, resample_shape, resample_mode, timesample)

