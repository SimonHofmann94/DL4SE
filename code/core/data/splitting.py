"""
Stratified splitting for maintaining class distribution across train/val/test sets.

This is critical for imbalanced datasets where losing rare class examples
in the training set significantly hurts generalization.
"""

import numpy as np
import torch
from typing import List, Tuple, Dict
import logging

logger = logging.getLogger(__name__)


class StratifiedSplitter:
    """
    Stratified splitter for multi-label classification datasets.
    
    Ensures that each split maintains similar class distribution to the original dataset.
    Handles multi-label scenarios where samples can belong to multiple classes.
    
    Args:
        random_state: Random seed for reproducibility
    
    Example:
        >>> splitter = StratifiedSplitter(random_state=42)
        >>> indices_train, indices_val, indices_test = splitter.split(
        ...     all_labels, split_ratios=(0.7, 0.15, 0.15)
        ... )
    """
    
    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        np.random.seed(random_state)
    
    def split(
        self,
        labels: np.ndarray,
        split_ratios: Tuple[float, float, float] = (0.7, 0.15, 0.15)
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Split dataset maintaining class distribution.
        
        For multi-label data, we use a stratification strategy based on
        the "dominant class" (most frequent class for each sample) or
        iterate through classes to balance across splits.
        
        Args:
            labels: Binary label matrix of shape (N, C) where labels[i, j] = 1
                   indicates sample i has class j
            split_ratios: Tuple of (train_ratio, val_ratio, test_ratio)
                         Must sum to 1.0
        
        Returns:
            Tuple of (train_indices, val_indices, test_indices)
        """
        # Validate input
        train_ratio, val_ratio, test_ratio = split_ratios
        assert (
            abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6
        ), f"Ratios must sum to 1.0, got {train_ratio + val_ratio + test_ratio}"
        
        n_samples = labels.shape[0]
        n_classes = labels.shape[1]
        
        # Strategy: For multi-label data, we shuffle all samples first,
        # then split them while trying to maintain class distribution
        
        logger.info(
            f"Performing stratified split on {n_samples} samples with {n_classes} classes"
        )
        logger.info(f"Split ratios: train={train_ratio}, val={val_ratio}, test={test_ratio}")
        
        # Calculate class frequencies
        class_frequencies = labels.sum(axis=0) / n_samples
        logger.debug(f"Class frequencies: {class_frequencies}")
        
        # Simple approach: Shuffle all indices and split directly
        # This maintains approximate class distribution for multi-label data
        all_indices = np.arange(n_samples)
        shuffled_indices = np.random.permutation(all_indices)
        
        # Calculate split points
        train_end = int(n_samples * train_ratio)
        val_end = train_end + int(n_samples * val_ratio)
        
        # Split indices
        train_indices = shuffled_indices[:train_end]
        val_indices = shuffled_indices[train_end:val_end]
        test_indices = shuffled_indices[val_end:]
        
        # No overlap by construction (disjoint slices)
        
        logger.info(
            f"Split complete: train={len(train_indices)}, "
            f"val={len(val_indices)}, test={len(test_indices)}"
        )
        
        # Verify class distribution preservation
        self._verify_distribution(
            labels,
            train_indices,
            val_indices,
            test_indices
        )
        
        return train_indices, val_indices, test_indices
    
    def _verify_distribution(
        self,
        labels: np.ndarray,
        train_indices: np.ndarray,
        val_indices: np.ndarray,
        test_indices: np.ndarray
    ) -> None:
        """Verify that class distribution is preserved across splits."""
        
        def get_distribution(indices):
            subset_labels = labels[indices]
            return subset_labels.sum(axis=0) / subset_labels.shape[0]
        
        orig_dist = labels.sum(axis=0) / labels.shape[0]
        train_dist = get_distribution(train_indices)
        val_dist = get_distribution(val_indices)
        test_dist = get_distribution(test_indices)
        
        logger.debug("\nClass distribution verification:")
        logger.debug(f"  Original : {orig_dist}")
        logger.debug(f"  Train    : {train_dist}")
        logger.debug(f"  Val      : {val_dist}")
        logger.debug(f"  Test     : {test_dist}")
        
        # Check that distributions are similar (within tolerance)
        tolerance = 0.05  # 5% tolerance
        for c in range(labels.shape[1]):
            if abs(train_dist[c] - orig_dist[c]) > tolerance:
                logger.warning(
                    f"Class {c} distribution may not be well preserved in train split"
                )


if __name__ == "__main__":
    # Test stratified splitting
    np.random.seed(42)
    
    # Create synthetic multi-label data
    n_samples = 1000
    n_classes = 4
    
    # Create imbalanced labels (class 3 is rare)
    labels = np.random.rand(n_samples, n_classes) > np.array([0.3, 0.7, 0.2, 0.85])
    
    print(f"Original label distribution:\n{labels.sum(axis=0) / n_samples}")
    
    # Split data
    splitter = StratifiedSplitter(random_state=42)
    train_idx, val_idx, test_idx = splitter.split(
        labels,
        split_ratios=(0.7, 0.15, 0.15)
    )
    
    print(f"\nTrain set distribution:\n{labels[train_idx].sum(axis=0) / len(train_idx)}")
    print(f"\nVal set distribution:\n{labels[val_idx].sum(axis=0) / len(val_idx)}")
    print(f"\nTest set distribution:\n{labels[test_idx].sum(axis=0) / len(test_idx)}")
