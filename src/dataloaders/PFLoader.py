
import torch
from torch.utils.data import ConcatDataset, Dataset
import h5py

class HDF5ConcatDataset(ConcatDataset):
    def __init__(self, datasets):
        super().__init__(datasets)

    def time_window(self):
        return self.datasets[0].time_window

    def absmax_vel(self):
        return max(d.absmax_vel() for d in self.datasets)

    def absmax_temp(self):
        return max(d.absmax_temp() for d in self.datasets)
    
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

    def datum_dim(self):
        return self.datasets[0].datum_dim()

class PFLoader(Dataset):
    def __init__(self,
                 filename,
                 discard_first,
                 use_coords,
                 time_window,
                 push_forward_steps=0):
        #super().__init__()
        self.time_window = time_window
        self.push_forward_steps = push_forward_steps
        self.filename = filename
        self.discard_first = discard_first
        self.temp_scale = None
        self.vel_scale = None

        if use_coords:
            self.coords_dim = 2
        else:
            self.coords_dim = 0
        self.temp_channels = self.time_window
        self.vel_channels = self.time_window * 2

        self.in_channels = self.coords_dim + self.temp_channels + self.vel_channels 
        self.out_channels = 3 * self.time_window

        self.read_files()
    
    def read_files(self):
        self._data = {}
        with h5py.File(self.filename, 'r') as f:
            self._data['temp'] = torch.nan_to_num(torch.from_numpy(f['temperature'][:][self.discard_first:])).float()
            self._data['velx'] = torch.nan_to_num(torch.from_numpy(f['velx'][:][self.discard_first:])).float()
            self._data['vely'] = torch.nan_to_num(torch.from_numpy(f['vely'][:][self.discard_first:])).float()
            #self._data['dfun'] = torch.nan_to_num(torch.from_numpy(f['dfun'][:][self.discard_first:]))
            self._data['x'] = torch.from_numpy(f['x'][:][self.discard_first:]).float()
            self._data['y'] = torch.from_numpy(f['y'][:][self.discard_first:]).float()

        if self.temp_scale and self.vel_scale:
            self.normalize_temp_(self.temp_scale)
            self.normalize_vel_(self.vel_scale)

    def absmax_temp(self):
        return self._data['temp'].abs().max()

    def absmax_vel(self):
        return max(self._data['velx'].abs().max(), self._data['vely'].abs().max())
    
    def normalize_temp_(self, scale):
        self._data['temp'] = 2 * (self._data['temp'] / scale) - 1
        self.temp_scale = scale

    def normalize_vel_(self, scale):
        for v in ('velx', 'vely'):
            self._data[v] = self._data[v] / scale
        self.vel_scale = scale

    def _get_temp(self, timestep):
        return self._data['temp'][timestep]
    
    def _get_temp_full(self):
        return self._data['temp']

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
        coords = torch.stack([x, y], dim=0)
        return coords

    def __len__(self):
        return self._data['temp'].size(0) - self.time_window - (self.time_window * self.push_forward_steps - 1) 

    def _get_timestep(self, timestep):
        r"""
        Get the window rooted at timestep.
        This includes the {timestep - self.time_window, ..., timestep - 1} as input
        and {timestep, ..., timestep + future_window - 1} as output
        """
        coords = self._get_coords(timestep)
        temp = torch.stack([self._get_temp(timestep + k) for k in range(self.time_window)], dim=0)
        vel = torch.cat([self._get_vel_stack(timestep + k) for k in range(self.time_window)], dim=0) 
        

        base_time = timestep + self.time_window 
        temp_label = torch.stack([self._get_temp(base_time + k) for k in range(self.time_window)], dim=0)
        vel_label = torch.cat([self._get_vel_stack(base_time + k) for k in range(self.time_window)], dim=0)
        return coords, temp, vel, temp_label, vel_label

    def __getitem__(self, timestep):
        r"""
        Get the windows rooted at {timestep, timestep + self.future_window, ...}
        For each variable, the windows are concatenated into one tensor.
        """
        args = list(zip(*[self._get_timestep(timestep + k * self.time_window) for k in range(self.push_forward_steps)]))
        return tuple([torch.stack(arg, dim=0) for arg in args])

    def write_vel(self, vel, timestep):
        base_time = timestep + self.time_window
        self._data['velx'][base_time:base_time + self.time_window] = vel[0::2]
        self._data['vely'][base_time:base_time + self.time_window] = vel[1::2]

    def write_temp(self, temp, timestep):
        if temp.dim() == 2:
            temp.unsqueeze_(-1)
        base_time = timestep + self.time_window
        self._data['temp'][base_time:base_time + self.time_window] = temp