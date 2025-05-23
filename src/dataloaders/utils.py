import torch
from pathlib import Path
import numpy as np
from torch.utils.data import Sampler

class ZeroShotSampler(torch.utils.data.Sampler):
    def __init__(self, dataset, train_ratio=0.8, split="train", seed=227, forward_steps=1):
        torch.manual_seed(seed) 
        num_train = int(dataset.traj * train_ratio)
        shuffled_trajs = torch.randperm(dataset.traj).tolist() 
        train_trajs = shuffled_trajs[:num_train]
        self.val_trajs = shuffled_trajs[num_train:]
        train_indices = [t * dataset.lenpertraj + ts for t in train_trajs for ts in range(0, dataset.lenpertraj, forward_steps)] # we take only every forward_steps timestep for quickness
        val_indices = [t * dataset.lenpertraj + ts for t in self.val_trajs for ts in range(0, dataset.lenpertraj, forward_steps)]
        self.indices = train_indices if split == "train" else val_indices
    def __iter__(self):
        return iter(self.indices)
    def __len__(self):
        return len(self.indices)
    def random_val_traj(self):
        return self.val_trajs[torch.randint(0, len(self.val_trajs), (1,)).item()]
  
def bicubic_resample(data, target_shape, device):
    assert data.dim() >= 2, "Input must have at least 2 dimensions"
    assert data.shape[-2] == data.shape[-1], 'Only square images supported'
    assert target_shape[0] == target_shape[1], 'Only square output images supported'
    assert data.shape[-2] != target_shape[0], 'Image already in target shape'
    
    original_shape = data.shape
    flattened = data.reshape(-1, *original_shape[-2:]).to(device)
    
    if flattened.dim() == 2:
        flattened = flattened.unsqueeze(0).unsqueeze(0) 
    elif flattened.dim() == 3:
        flattened = flattened.unsqueeze(1) 
    
    resampled = torch.nn.functional.interpolate(
        flattened, 
        size=target_shape, 
        mode='bicubic', 
        align_corners=False
    )
    
    if data.dim() == 2:
        resampled = resampled.squeeze(0).squeeze(0)
    elif data.dim() == 3:
        resampled = resampled.squeeze(1)
    else:
        resampled = resampled.reshape(*original_shape[:-2], *target_shape)
    
    return resampled

def fourier_resample(data, target_shape, device):
    assert data.dim() >= 2, "Input must have at least 2 dimensions"
    
    original_shape = data.shape
    x, y = data.shape[-2], data.shape[-1]
    target_x, target_y = target_shape
    assert x == y, 'Only square images supported'
    assert target_x == target_y, 'Only square images supported'

    #device = data.device
    flattened = data.reshape(-1, x, y)
    signal = flattened.to(dtype=torch.complex64, device=device)

    pos0 = torch.linspace(0.5, x - 0.5, x, device=device)
    pos1 = torch.linspace(0.5, target_x - 0.5, target_x, device=device)
    freq0 = torch.fft.fftfreq(x, device=device)
    freq1 = torch.fft.fftfreq(target_x, device=device)

    exp_matrix_0 = torch.exp(-2j * torch.pi * torch.outer(pos0, freq0))
    exp_matrix_1 = torch.exp(2j * torch.pi * torch.outer(freq1, pos1))

    results = []
    for item in signal:
        freq_coeff = torch.matmul(torch.matmul(exp_matrix_0.H, item), exp_matrix_0) / x**2
        scaled_coeff = torch.zeros((target_x, target_x), dtype=torch.complex64, device=device)

        if target_x > x:  # Upsample
            min_idx = (target_x - x) // 2
            scaled_coeff[min_idx:min_idx + x, min_idx:min_idx + x] = torch.fft.fftshift(freq_coeff)
            scaled_coeff = torch.fft.ifftshift(scaled_coeff)
        else:  # Downsample
            min_idx = (x - target_x) // 2
            scaled_coeff = torch.fft.fftshift(freq_coeff)[min_idx:min_idx + target_x, min_idx:min_idx + target_x]
            scaled_coeff = torch.fft.ifftshift(scaled_coeff)

        scaled_signal = torch.matmul(torch.matmul(exp_matrix_1.H, scaled_coeff), exp_matrix_1)
        results.append(scaled_signal.real)

    output = torch.stack(results).reshape(*original_shape[:-2], target_x, target_y)
    return output


def spatial_resample(data, target_shape, mode):
    if data.shape[-1] == target_shape and data.shape[-2] == target_shape:
        return data
    old_device = data.device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if mode == 'bicubic':
        return bicubic_resample(data, (target_shape, target_shape), device).to(old_device)
    elif mode == 'fourier':
        return fourier_resample(data, (target_shape, target_shape), device).to(old_device)
    else:
        raise ValueError(f'Unknown mode: {mode}')


def get_dataset(dataset_obj, folderPath, preproc_savepath, file_ext, resample_shape, resample_mode, timesample, dataset_name):
    subdir = Path(folderPath)
    assert subdir.exists(), 'subdir doesnt exist'
    files = list(subdir.glob("*." + str(file_ext)))
    return dataset_obj(files, preproc_savepath, resample_shape, resample_mode, timesample, dataset_name)

