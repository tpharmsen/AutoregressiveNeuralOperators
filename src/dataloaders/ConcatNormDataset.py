from torch.utils.data import ConcatDataset

class ConcatNormDataset(ConcatDataset):
    def __init__(self, datasets):
        super().__init__(datasets)

    def absmax_vel(self):
        return max(d.dataset.absmax_vel() for d in self.datasets)

    def normalize_velocity(self, absmax_vel=None):
        if not absmax_vel:
            absmax_vel = self.absmax_vel()
        for d in self.datasets:
            d.dataset.normalize_velocity(absmax_vel)
        return absmax_vel

    