from .AmiraPreProc import AmiraPreProc
from .PDEBenchCompPreProc import PDEBenchCompPreProc
from .PDEBenchIncompPreProc import PDEBenchIncompPreProc
from .PDEGymPreProc import PDEGymPreProc

from .ConcatNormDataset import ConcatNormDataset

# Optional: Define what gets imported when using `from dataloaders import *`
__all__ = [
    "DiskDataset"
    "AmiraPreProc", 
    "PDEBenchCompPreProc",
    "PDEBenchIncompPreProc",
    "PDEGymPreProc",
    "ConcatNormDataset",
]
PREPROC_MAPPER = {"AmiraPreProc": AmiraPreProc,
                  "PDEBenchCompPreProc": PDEBenchCompPreProc,
                "PDEBenchIncompPreProc": PDEBenchIncompPreProc,
                "PDEGymPreProc": PDEGymPreProc
                  }
"""
DATASET_MAPPER = {"AmiraDatasetFromAM": AmiraDatasetFromAM,
                  "AmiraDatasetFromH5": AmiraDatasetFromH5,
                  "PDEBenchCompDataset": PDEBenchCompDataset,
                  "PDEBenchIncompDataset": PDEBenchIncompDataset,
                  "PDEGymDataset": PDEGymDataset}

"""