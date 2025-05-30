from .AmiraPreProc import AmiraPreProc
from .AmiraPreProcDiv import AmiraPreProcDiv
from .PDEBenchCompPreProc import PDEBenchCompPreProc
from .PDEBenchIncompPreProc import PDEBenchIncompPreProc
from .PDEGymPreProc import PDEGymPreProc
from .PDEGymPreProcDiv import PDEGymPreProcDiv
from .DiskDataset import DiskDataset
from .DiskDatasetDiv import DiskDatasetDiv
from .ConcatNormDataset import ConcatNormDataset

# Optional: Define what gets imported when using `from dataloaders import *`
__all__ = [
    "DiskDataset",
    "DiskDatasetDiv",
    "AmiraPreProc", 
    "AmiraPreProcDiv",
    "PDEBenchCompPreProc",
    "PDEBenchIncompPreProc",
    "PDEGymPreProc",
    "PDEGymPreProcDiv",
    "ConcatNormDataset",
]
PREPROC_MAPPER = {"AmiraPreProc": AmiraPreProc,
                "AmiraPreProcDiv": AmiraPreProcDiv,
                "PDEBenchCompPreProc": PDEBenchCompPreProc,
                "PDEBenchIncompPreProc": PDEBenchIncompPreProc,
                "PDEGymPreProcDiv": PDEGymPreProcDiv,
                "PDEGymPreProc": PDEGymPreProc
                  }
"""
DATASET_MAPPER = {"AmiraDatasetFromAM": AmiraDatasetFromAM,
                  "AmiraDatasetFromH5": AmiraDatasetFromH5,
                  "PDEBenchCompDataset": PDEBenchCompDataset,
                  "PDEBenchIncompDataset": PDEBenchIncompDataset,
                  "PDEGymDataset": PDEGymDataset}

"""