"""Dataset classes for AsTorch."""

import numpy as np
from typing import Any, List, Tuple


class Dataset:
    """Base dataset class."""
    
    def __len__(self) -> int:
        raise NotImplementedError
    
    def __getitem__(self, idx) -> Any:
        raise NotImplementedError


class MNISTDataset(Dataset):
    """MNIST dataset from numpy arrays."""
    
    def __init__(self, images: np.ndarray, labels: np.ndarray):
        """
        Args:
            images: Array of shape (N, 28, 28)
            labels: Array of shape (N,)
        """
        self.images = images
        self.labels = labels
        assert len(self.images) == len(self.labels)
    
    def __len__(self) -> int:
        return len(self.images)
    
    def __getitem__(self, idx) -> Tuple[np.ndarray, int]:
        return self.images[idx], self.labels[idx]


class Subset(Dataset):
    """Subset of a dataset."""
    
    def __init__(self, dataset: Dataset, indices: List[int]):
        self.dataset = dataset
        self.indices = indices
    
    def __len__(self) -> int:
        return len(self.indices)
    
    def __getitem__(self, idx) -> Any:
        return self.dataset[self.indices[idx]]
