"""
DataLoader creation utilities.
"""

import torch
from torch.utils.data import DataLoader, Subset
from typing import Tuple, Optional
import logging

logger = logging.getLogger(__name__)


def create_dataloaders(
    train_dataset,
    val_dataset,
    test_dataset,
    batch_size: int = 4,
    num_workers: int = 0,
    pin_memory: bool = False,
    shuffle_train: bool = True,
    shuffle_val: bool = False,
    shuffle_test: bool = False
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create DataLoaders for train, validation, and test sets.
    
    Args:
        train_dataset: Training dataset
        val_dataset: Validation dataset
        test_dataset: Test dataset
        batch_size: Batch size for all loaders
        num_workers: Number of worker processes
        pin_memory: Pin tensors to GPU memory
        shuffle_train: Shuffle training data
        shuffle_val: Shuffle validation data
        shuffle_test: Shuffle test data
    
    Returns:
        Tuple of (train_loader, val_loader, test_loader)
    """
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=shuffle_train,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=shuffle_val,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=shuffle_test,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False
    )
    
    logger.info(
        f"Created DataLoaders:"
        f" train={len(train_loader)} batches, "
        f"val={len(val_loader)} batches, "
        f"test={len(test_loader)} batches"
    )
    
    return train_loader, val_loader, test_loader
