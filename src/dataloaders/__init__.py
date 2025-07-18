from .AmiraPreProc import AmiraPreProc
from .AmiraPreProcDiv import AmiraPreProcDiv
from .PDEBenchCompPreProc import PDEBenchCompPreProc
from .PDEBenchCompPreProcDiv import PDEBenchCompPreProcDiv
from .PDEBenchIncompPreProc import PDEBenchIncompPreProc
from .PDEBenchIncompPreProcDiv import PDEBenchIncompPreProcDiv
from .PDEGymPreProc import PDEGymPreProc
from .PDEGymPreProcDiv import PDEGymPreProcDiv

from .DiskDataset import DiskDataset
from .DiskDatasetDiv import DiskDatasetDiv
from .DiskDatasetDivFM import DiskDatasetDivFM
from .ConcatNormDataset import ConcatNormDataset


# Optional: Define what gets imported when using `from dataloaders import *`
__all__ = [
    "DiskDataset",
    "DiskDatasetDiv",
    "DiskDatasetDivFM",
    "AmiraPreProc", 
    "AmiraPreProcDiv",
    "PDEBenchCompPreProc",
    "PDEBenchCompPreProcDiv",
    "PDEBenchIncompPreProc",
    "PDEBenchIncompPreProcDiv",
    "PDEGymPreProc",
    "PDEGymPreProcDiv",
    "ConcatNormDataset",
]
PREPROC_MAPPER = {"AmiraPreProc": AmiraPreProc,
                "AmiraPreProcDiv": AmiraPreProcDiv,
                "PDEBenchCompPreProc": PDEBenchCompPreProc,
                "PDEBenchCompPreProcDiv": PDEBenchCompPreProcDiv,
                "PDEBenchIncompPreProc": PDEBenchIncompPreProc,
                "PDEBenchIncompPreProcDiv": PDEBenchIncompPreProcDiv,
                "PDEGymPreProc": PDEGymPreProc,
                "PDEGymPreProcDiv": PDEGymPreProcDiv,
                "DiskDatasetDivFM": DiskDatasetDivFM
                  }
"""
DATASET_MAPPER = {"AmiraDatasetFromAM": AmiraDatasetFromAM,
                  "AmiraDatasetFromH5": AmiraDatasetFromH5,
                  "PDEBenchCompDataset": PDEBenchCompDataset,
                  "PDEBenchIncompDataset": PDEBenchIncompDataset,
                  "PDEGymDataset": PDEGymDataset}

"""