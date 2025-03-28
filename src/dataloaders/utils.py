import torch

def bicubic_resample(data, target_shape):
    assert data.shape[-2] == data.shape[-1], 'Only square images supported'
    assert target_shape[0] == target_shape[1], 'Only square output images supported'
    assert data.shape[-2] != target_shape[0], 'Image already in target shape'
    return torch.nn.functional.interpolate(data, size=target_shape, mode='bicubic')

def fourier_resample(data, target_shape):
    x, y = data.shape[-2], data.shape[-1]
    target_x, target_y = target_shape
    assert x == y, 'Only square images supported'
    assert target_x == target_y, 'Only square images supported'
    data = data.to(torch.complex64)
    
    pos0 = torch.linspace(0.5, x-0.5, x, dtype=torch.complex64)
    pos1 = torch.linspace(0.5, target_x-0.5, target_x, dtype=torch.complex64)
    freq0 = torch.fft.fftfreq(x)
    freq1 = torch.fft.fftfreq(target_x)

    exp_matrix_0 = torch.exp(-2j * torch.pi * torch.outer(pos0, freq0))
    freq_coeff = torch.matmul(torch.matmul(exp_matrix_0.H, data), exp_matrix_0) / x**2

    if target_x > x: # upsample
        scaled_coeff = torch.zeros((target_x, target_x), dtype=torch.complex64)
        min_idx = (target_x - x) // 2
        scaled_coeff[min_idx:min_idx+x, min_idx:min_idx+x] = torch.fft.fftshift(freq_coeff)
        scaled_coeff = torch.fft.ifftshift(scaled_coeff)
    elif target_x < x: #downsample
        scaled_coeff = torch.zeros((target_x, target_x), dtype=torch.complex64)
        min_idx = (x - target_x) // 2
        scaled_coeff = torch.fft.fftshift(freq_coeff)[min_idx:min_idx+target_x, min_idx:min_idx+target_x]

        scaled_coeff = torch.fft.ifftshift(scaled_coeff)
    else:
        raise ValueError("Image already in target shape")
    exp_matrix_1 = torch.exp(2j * torch.pi * torch.outer(freq1, pos1))
    upscaled_signal = torch.matmul(torch.matmul(exp_matrix_1.H, scaled_coeff), exp_matrix_1) 
    return upscaled_signal.real


def spatial_resample(data, target_shape, mode='bicubic'):
    if mode == 'bicubic':
        return bicubic_resample(data, target_shape)
    elif mode == 'fourier':
        return fourier_resample(data.squeeze(), target_shape).unsqueeze(0).unsqueeze(0)
    else:
        raise ValueError(f'Unknown mode: {mode}')