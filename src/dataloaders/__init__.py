from .AmiraDatasetFromAM import AmiraReaderFromAM, AmiraDatasetFromAM
from .AmiraDatasetFromH5 import AmiraReaderFromH5, AmiraDatasetFromH5
from .PDEBenchCompDataset import PDEBenchCompReader, PDEBenchCompDataset
from .PDEBenchIncompDataset import PDEBenchIncompReader, PDEBenchIncompDataset
from .PDEGymDataset import PDEGymReader, PDEGymDataset
from .ConcatNormDataset import ConcatNormDataset

# Optional: Define what gets imported when using `from dataloaders import *`
__all__ = [
    "AmiraReaderFromAM",
    "AmiraDatasetFromAM",
    "AmiraReaderFromH5",
    "AmiraDatasetFromH5",
    "PDEBenchCompReader",
    "PDEBenchCompDataset",
    "PDEBenchIncompReader",
    "PDEBenchIncompDataset",
    "PDEGymReader",
    "PDEGymDataset",
    "ConcatNormDataset",
]
READER_MAPPER = {"AmiraDatasetFromAM": AmiraReaderFromAM,
                  "AmiraDatasetFromH5": AmiraReaderFromH5,
                  "PDEBenchCompDataset": PDEBenchCompReader,
                  "PDEBenchIncompDataset": PDEBenchIncompReader,
                  "PDEGymDataset": PDEGymReader}

DATASET_MAPPER = {"AmiraDatasetFromAM": AmiraDatasetFromAM,
                  "AmiraDatasetFromH5": AmiraDatasetFromH5,
                  "PDEBenchCompDataset": PDEBenchCompDataset,
                  "PDEBenchIncompDataset": PDEBenchIncompDataset,
                  "PDEGymDataset": PDEGymDataset}

