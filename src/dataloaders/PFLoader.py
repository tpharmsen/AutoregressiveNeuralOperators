
import torch
from torch.utils.data import ConcatDataset, Dataset
import torchvision
import h5py
import random

class HDF5ConcatDataset(ConcatDataset):
    def __init__(self, datasets):
        super().__init__(datasets)

    def time_window(self):
        return self.datasets[0].time_window

    def absmax_vel(self):
        return max(d.absmax_vel() for d in self.datasets)

    def absmax_temp(self):
        return max(d.absmax_temp() for d in self.datasets)
    
    def absmax_phase(self):
        return max(d.absmax_phase() for d in self.datasets)
    
    def _get_temp_full(self, num):
        return self.datasets[num]._data['temp']

    def normalize_temp_(self, absmax_temp=None):
        if not absmax_temp:
            absmax_temp = self.absmax_temp()
        for d in self.datasets:
            d.normalize_temp_(absmax_temp)
        return absmax_temp

    def normalize_vel_(self, absmax_vel=None):
        if not absmax_vel:
            absmax_vel = self.absmax_vel()
        for d in self.datasets:
            d.normalize_vel_(absmax_vel)
        return absmax_vel
    
    def normalize_phase_(self, absmax_phase=None):
        if not absmax_phase:
            absmax_phase = self.absmax_phase()
        for d in self.datasets:
            d.normalize_phase_(absmax_phase)
        return absmax_phase

    def datum_dim(self):
        return self.datasets[0].datum_dim()
    
    def get_validation_stacks(self, dataset_num):
        
        if dataset_num >= len(self.datasets) or dataset_num < 0:
            raise IndexError(f"Dataset index {dataset_num} is out of range. Total datasets: {len(self.datasets)}")

        dataset = self.datasets[dataset_num]

        temp_data = dataset._data['temp'] 
        velx_data = dataset._data['velx']  
        vely_data = dataset._data['vely'] 
        phase_data = dataset._data['dfun']


        interleaved_vel = torch.empty(
        velx_data.size(0) * 2, velx_data.size(1), velx_data.size(2),
        dtype=velx_data.dtype, device=velx_data.device
        )  

        interleaved_vel[0::2] = velx_data  # Fill even indices with velx
        interleaved_vel[1::2] = vely_data  # Fill odd indices with vely

        temp_data = temp_data.unsqueeze(0)        
        interleaved_vel = interleaved_vel.unsqueeze(0)  
        phase_data = phase_data.unsqueeze(0)         
        #del velx_data, vely_data
        return temp_data, interleaved_vel, phase_data
    
    

class PFLoader(Dataset):
    def __init__(self,
                 filename,
                 discard_first,
                 use_coords,
                 time_window,
                 push_forward_steps=0,
                 transform=False):
        #super().__init__()
        self.time_window = time_window
        self.push_forward_steps = push_forward_steps
        self.filename = filename
        self.discard_first = discard_first
        self.temp_scale = None
        self.vel_scale = None
        self.phase_scale = None

        if use_coords:
            self.coords_dim = 2
        else:
            self.coords_dim = 0
        self.temp_channels = self.time_window
        self.vel_channels = self.time_window * 2
        self.phase_channels = self.time_window

        self.in_channels = self.coords_dim + self.temp_channels + self.vel_channels + self.phase_channels
        self.out_channels = 4 * self.time_window

        self.transform = transform
        self.read_files()
    
    def read_files(self):
        self._data = {}
        with h5py.File(self.filename, 'r') as f:
            self._data['temp'] = torch.nan_to_num(torch.from_numpy(f['temperature'][:][self.discard_first:])).float()
            self._data['velx'] = torch.nan_to_num(torch.from_numpy(f['velx'][:][self.discard_first:])).float()
            self._data['vely'] = torch.nan_to_num(torch.from_numpy(f['vely'][:][self.discard_first:])).float()
            self._data['dfun'] = torch.nan_to_num(torch.from_numpy(f['dfun'][:][self.discard_first:])).float()
            self._data['x'] = torch.from_numpy(f['x'][:][self.discard_first:]).float()
            self._data['y'] = torch.from_numpy(f['y'][:][self.discard_first:]).float()

        if self.temp_scale and self.vel_scale and self.phase_scale:
            self.normalize_temp_(self.temp_scale)
            self.normalize_vel_(self.vel_scale)
            self.normalize_phase_(self.phase_scale)

    def _transform(self, coords, temp, vel, phase, temp_label, vel_label, phase_label):
        if self.transform:
            if random.random() > 0.5:
                
                #coords = torchvision.transforms.functional.hflip(coords)
                temp = torchvision.transforms.functional.hflip(temp)
                vel = torchvision.transforms.functional.hflip(vel)
                phase = torchvision.transforms.functional.hflip(phase)
                temp_label = torchvision.transforms.functional.hflip(temp_label)
                vel_label = torchvision.transforms.functional.hflip(vel_label)
                phase_label = torchvision.transforms.functional.hflip(phase_label)
                
                # Flip x-velocity (even indices of the interleaved velocity tensor)
                #print(vel.shape)
                vel[0::2] *= -1  
                vel_label[0::2] *= -1 
        return (coords, temp, vel, phase, temp_label, vel_label, phase_label)

    def absmax_temp(self):
        return self._data['temp'].abs().max()

    def absmax_vel(self):
        return max(self._data['velx'].abs().max(), self._data['vely'].abs().max())
    
    def absmax_phase(self):
        return self._data['dfun'].abs().max()
    
    def normalize_temp_(self, scale):
        self._data['temp'] = 2 * (self._data['temp'] / scale) - 1
        self.temp_scale = scale

    def normalize_vel_(self, scale):
        for v in ('velx', 'vely'):
            self._data[v] = self._data[v] / scale
        self.vel_scale = scale

    def normalize_phase_(self, scale):
        self._data['dfun'] = self._data['dfun'] / scale
        self.phase_scale = scale

    def _get_temp(self, timestep):
        return self._data['temp'][timestep]
    
    def _get_phase(self, timestep):
        return torch.heaviside(self._data['dfun'][timestep], values=torch.Tensor([1]))  # pass phase as 0 to 1 values

    def _get_vel_stack(self, timestep):
        return torch.stack([
            self._data['velx'][timestep],
            self._data['vely'][timestep],
        ], dim=0)

    def _get_coords(self, timestep):
        x = self._data['x'][timestep]
        x /= x.max()
        y = self._data['y'][timestep]
        y /= y.max()
        #print('x:', x.shape, x[0], x[-1], x.min(), x.max(), x.mean())
        #print()
        #print('y:', y.shape, y[0], y[-1], y.min(), y.max(), y.mean())
        coords = torch.stack([x, y], dim=0)
        return coords

    def __len__(self):
        return self._data['temp'].size(0) - self.time_window - (self.time_window * self.push_forward_steps - 1) 

    def _get_timestep(self, timestep):
        
        coords = self._get_coords(timestep)
        temp = torch.stack([self._get_temp(timestep + k) for k in range(self.time_window)], dim=0)
        vel = torch.cat([self._get_vel_stack(timestep + k) for k in range(self.time_window)], dim=0) 
        phase = torch.stack([self._get_phase(timestep + k) for k in range(self.time_window)], dim=0)
        
        base_time = timestep + self.time_window 
        temp_label = torch.stack([self._get_temp(base_time + k) for k in range(self.time_window)], dim=0)
        vel_label = torch.cat([self._get_vel_stack(base_time + k) for k in range(self.time_window)], dim=0)
        phase_label = torch.stack([self._get_phase(base_time + k) for k in range(self.time_window)], dim=0)
        '''
        if self.time_window == 1:
            coords = coords.unsqueeze(0)  # Add the time dimension
            temp = temp.unsqueeze(0)
            vel = vel.unsqueeze(0)
            phase = phase.unsqueeze(0)
            temp_label = temp_label.unsqueeze(0)
            vel_label = vel_label.unsqueeze(0)
            phase_label = phase_label.unsqueeze(0)
        '''
        return self._transform(coords, temp, vel, phase, temp_label, vel_label, phase_label)

    def __getitem__(self, timestep):
        
        args = list(zip(*[self._get_timestep(timestep + k * self.time_window) for k in range(self.push_forward_steps)]))
        return tuple([torch.stack(arg, dim=0) for arg in args])

    