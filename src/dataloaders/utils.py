import torch


def bicubic_resample(data, target_shape):
    return torch.nn.functional.interpolate(data, size=target_shape, mode='bicubic')

def fourier_resample(data: torch.Tensor, target_shape: tuple) -> torch.Tensor:
    B, T, channels, x, y = data.shape
    new_x, new_y = target_shape
    
    data_fft = torch.fft.fft2(data, norm='forward')
    data_fft = torch.fft.fftshift(data_fft, dim=(-2, -1))
    x_min = (new_x - x) // 2
    y_min = (new_y - y) // 2
    
    if new_x >= x and new_y >= y:
        # Upsampling: Zero-padding in Fourier domain
        resampled_fft = torch.zeros((channels, new_x, new_y), dtype=torch.complex64, device=data.device)
        resampled_fft[:, x_min:x_min + x, y_min:y_min + y] = data_fft
    else:
        # Downsampling: Cropping in Fourier domain
        resampled_fft = data_fft[:, -x_min:new_x - x_min, -y_min:new_y - y_min]
    resampled_fft = torch.fft.ifftshift(resampled_fft, dim=(-2, -1))
    resampled_data = torch.fft.ifft2(resampled_fft, norm='forward').real
    return resampled_data

def spatial_resample(data, target_shape, mode='bicubic'):
    if mode == 'bicubic':
        return bicubic_resample(data, target_shape)
    elif mode == 'fourier':
        return fourier_resample(data.squeeze(), target_shape).unsqueeze(0).unsqueeze(0)
    else:
        raise ValueError(f'Unknown mode: {mode}')