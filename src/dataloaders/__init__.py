from .AmiraLoaderFromAM import AmiraDatasetFromAM
from .AmiraLoaderFromH5 import AmiraDatasetFromH5
from .PDEBenchCompLoader import PDEBenchCompDataset
from .PDEBenchIncompLoader import PDEBenchIncompDataset
from .PDEGymLoader import PDEGymDataset

# Optional: Define what gets imported when using `from dataloaders import *`
__all__ = [
    "AmiraLoaderFromAM",
    "AmiraLoaderFromH5",
    "PDEBenchCompLoader",
    "PDEBenchIncompLoader",
    "PDEGymLoader"
]