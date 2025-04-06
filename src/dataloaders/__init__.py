from .AmiraDatasetFromAM import AmiraDatasetFromAM
from .AmiraDatasetFromH5 import AmiraDatasetFromH5
from .PDEBenchCompDataset import PDEBenchCompDataset
from .PDEBenchIncompDataset import PDEBenchIncompDataset
from .PDEGymDataset import PDEGymDataset
from .ConcatNormDataset import ConcatNormDataset

# Optional: Define what gets imported when using `from dataloaders import *`
__all__ = [
    "AmiraDatasetFromAM",
    "AmiraDatasetFromH5",
    "PDEBenchCompDataset",
    "PDEBenchIncompDataset",
    "PDEGymDataset",
    "ConcatNormDataset",
]

DATASET_MAPPER = {"AmiraDatasetFromAM": AmiraDatasetFromAM,
                  "AmiraDatasetFromH5": AmiraDatasetFromH5,
                  "PDEBenchCompDataset": PDEBenchCompDataset,
                  "PDEBenchIncompDataset": PDEBenchIncompDataset,
                  "PDEGymDataset": PDEGymDataset}

