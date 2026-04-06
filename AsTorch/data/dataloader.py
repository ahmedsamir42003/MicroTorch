"""DataLoader for AsTorch."""

import numpy as np
from typing import Optional, Callable, Any, List


class DataLoader:
    """DataLoader for batching and iterating over datasets."""
    
    def __init__(
        self,
        dataset: Any,
        batch_size: int = 1,
        shuffle: bool = False,
        drop_last: bool = False,
        collate_fn: Optional[Callable] = None,
    ):
        """
        Args:
            dataset: Dataset with __len__ and __getitem__
            batch_size: Number of samples per batch
            shuffle: Whether to shuffle each epoch
            drop_last: Whether to drop the last incomplete batch
            collate_fn: Custom collate function
        """
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.collate_fn = collate_fn if collate_fn else self._default_collate
        self.indices = np.arange(len(dataset))
    
    def _default_collate(self, batch: List[Any]) -> Any:
        """Default collate function."""
        if isinstance(batch[0], (tuple, list)):
            components = list(zip(*batch))
            return tuple(np.stack(c) for c in components)
        else:
            return np.stack(batch)
    
    def __iter__(self):
        """Iterate over batches."""
        indices = self.indices.copy()
        if self.shuffle:
            np.random.shuffle(indices)
        
        for start_idx in range(0, len(self.dataset), self.batch_size):
            end_idx = min(start_idx + self.batch_size, len(self.dataset))
            
            if self.drop_last and end_idx - start_idx < self.batch_size:
                break
            
            batch_indices = indices[start_idx:end_idx]
            batch = [self.dataset[int(i)] for i in batch_indices]
            yield self.collate_fn(batch)
    
    def __len__(self) -> int:
        """Return number of batches."""
        if self.drop_last:
            return len(self.dataset) // self.batch_size
        else:
            return (len(self.dataset) + self.batch_size - 1) // self.batch_size
