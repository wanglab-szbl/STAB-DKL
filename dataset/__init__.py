"""
Dataset module for protein stability prediction.
"""

from .stability_dataset import (
    FeatureExtractor,
    StabilityDataset,
    DatasetConfig
)

__all__ = [
    'FeatureExtractor',
    'StabilityDataset',
    'DatasetConfig'
]
