import torch
# upscale/downscale with bicubic method or fourier spectral method

def bicubic_resample(data, target_shape):
    assert data.shape[-2] == data.shape[-1], 'Only square images supported'
    assert target_shape[0] == target_shape[1], 'Only square output images supported'
    assert data.shape[-2] != target_shape[0], 'Image already in target shape'
    return torch.nn.functional.interpolate(data, size=target_shape, mode='bicubic', align_corners=False)

def fourier_resample(data, target_shape):
    c, x, y = data.shape[-3], data.shape[-2], data.shape[-1]
    target_x, target_y = target_shape
    assert x == y, 'Only square images supported'
    assert target_x == target_y, 'Only square images supported'
    #if x == target_x:
    #    return data
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    sampled_data = []

    for i in range(c):

        signal = data[i].to(dtype=torch.complex64, device=device)
        
        pos0 = torch.linspace(0.5, x-0.5, x, device=device)
        pos1 = torch.linspace(0.5, target_x-0.5, target_x, device=device)
        freq0 = torch.fft.fftfreq(x, device=device)
        freq1 = torch.fft.fftfreq(target_x, device=device)

        exp_matrix_0 = torch.exp(-2j * torch.pi * torch.outer(pos0, freq0))


        freq_coeff = torch.matmul(torch.matmul(exp_matrix_0.H, signal), exp_matrix_0) / x**2
        scaled_coeff = torch.zeros((target_x, target_x), dtype=torch.complex64, device=device)

        if target_x > x: # upsample
            min_idx = (target_x - x) // 2
            scaled_coeff[min_idx:min_idx+x, min_idx:min_idx+x] = torch.fft.fftshift(freq_coeff)
            scaled_coeff = torch.fft.ifftshift(scaled_coeff)
        elif target_x < x: #downsample
            min_idx = (x - target_x) // 2
            scaled_coeff = torch.fft.fftshift(freq_coeff)[min_idx:min_idx+target_x, min_idx:min_idx+target_x]   
            scaled_coeff = torch.fft.ifftshift(scaled_coeff)
        else:
            raise ValueError("Image already in target shape")
        #print(freq1.shape, pos1.shape, scaled_coeff.shape)
        exp_matrix_1 = torch.exp(2j * torch.pi * torch.outer(freq1, pos1))
        scaled_signal = torch.matmul(torch.matmul(exp_matrix_1.H, scaled_coeff), exp_matrix_1) 
        sampled_data.append(scaled_signal)

    scaled_signal = torch.stack(sampled_data, dim=0)
    return scaled_signal.real.cpu()


def spatial_resample(data, target_shape, mode='bicubic'):
    if mode == 'bicubic':
        return bicubic_resample(data, target_shape)
    elif mode == 'fourier':
        return fourier_resample(data.squeeze(), target_shape)
    else:
        raise ValueError(f'Unknown mode: {mode}')