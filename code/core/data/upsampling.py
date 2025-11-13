"""
Class-Balanced Upsampling Strategies for Imbalanced Datasets.

This module provides strategies to handle class imbalance by oversampling
minority classes during training. Multiple strategies are supported:

1. Class-Balanced Sampling: Sample probabilities inversely proportional to class frequency
2. Per-Class Upsampling: Replicate minority class samples to match majority class
3. Hybrid: Combine upsampling with weighted sampling

All strategies preserve the original validation/test sets.
"""

import numpy as np
from typing import List, Tuple, Dict, Optional
import logging
from collections import Counter

logger = logging.getLogger(__name__)


class ClassBalancedSampler:
    """
    Sampler that balances class distribution by oversampling minority classes.
    
    For multi-label classification, we compute sampling weights based on the
    rarest positive label for each sample.
    
    Args:
        labels: Binary label matrix of shape (N, C) where N=samples, C=classes
        mode: Sampling strategy
              - 'sqrt': weight ~ 1/sqrt(freq) (moderate balancing)
              - 'inv': weight ~ 1/freq (strong balancing)
              - 'effective_num': Use effective number (Class-Balanced Loss paper)
        beta: Hyperparameter for effective_num mode (default: 0.9999)
        min_weight: Minimum sampling weight to prevent extreme imbalance
        
    Example:
        >>> labels = np.array([[1,0,0], [1,0,0], [0,1,0], [0,0,1]])
        >>> sampler = ClassBalancedSampler(labels, mode='inv')
        >>> weights = sampler.get_sample_weights()
        >>> # Sample with replacement using these weights
        >>> indices = np.random.choice(len(labels), size=100, p=weights/weights.sum())
    """
    
    def __init__(
        self,
        labels: np.ndarray,
        mode: str = 'sqrt',
        beta: float = 0.9999,
        min_weight: float = 0.1
    ):
        self.labels = labels
        self.mode = mode
        self.beta = beta
        self.min_weight = min_weight
        
        self.num_samples = len(labels)
        self.num_classes = labels.shape[1]
        
        # Compute class frequencies
        self.class_counts = labels.sum(axis=0)  # (C,)
        self.class_freqs = self.class_counts / self.num_samples  # (C,)
        
        logger.info(f"ClassBalancedSampler initialized:")
        logger.info(f"  Mode: {mode}")
        logger.info(f"  Num samples: {self.num_samples}")
        logger.info(f"  Class counts: {self.class_counts}")
        logger.info(f"  Class frequencies: {self.class_freqs}")
    
    def _compute_class_weights(self) -> np.ndarray:
        """
        Compute per-class sampling weights.
        
        Returns:
            Array of shape (C,) with weights for each class
        """
        if self.mode == 'inv':
            # Inverse frequency: w_c = 1 / freq_c
            weights = 1.0 / (self.class_freqs + 1e-6)
        
        elif self.mode == 'sqrt':
            # Square root of inverse frequency (more moderate)
            weights = 1.0 / (np.sqrt(self.class_freqs) + 1e-6)
        
        elif self.mode == 'effective_num':
            # Effective number from Class-Balanced Loss paper
            # E_n = (1 - beta^n) / (1 - beta)
            # weight = 1 / E_n
            effective_num = (1.0 - np.power(self.beta, self.class_counts)) / (1.0 - self.beta)
            weights = 1.0 / (effective_num + 1e-6)
        
        else:
            raise ValueError(f"Unknown mode: {self.mode}")
        
        # Normalize weights to [min_weight, 1.0]
        weights = weights / weights.max()
        weights = np.maximum(weights, self.min_weight)
        
        logger.info(f"Class weights computed: {weights}")
        
        return weights
    
    def get_sample_weights(self) -> np.ndarray:
        """
        Compute per-sample weights based on their rarest positive label.
        
        For multi-label, we take the MAXIMUM weight across all positive labels
        (i.e., weight is determined by the rarest class present).
        
        Returns:
            Array of shape (N,) with sampling weight for each sample
        """
        class_weights = self._compute_class_weights()  # (C,)
        
        # For each sample, compute weight based on rarest positive label
        sample_weights = np.zeros(self.num_samples)
        
        for i in range(self.num_samples):
            # Get positive labels for this sample
            positive_labels = np.where(self.labels[i] > 0)[0]
            
            if len(positive_labels) == 0:
                # No positive labels (should not happen in multi-label)
                sample_weights[i] = 1.0
            else:
                # Weight = max weight of positive labels (rarest class dominates)
                sample_weights[i] = class_weights[positive_labels].max()
        
        # Normalize to sum to num_samples (so expected samples per epoch = num_samples)
        sample_weights = sample_weights / sample_weights.sum() * self.num_samples
        
        return sample_weights


class UpsamplingStrategy:
    """
    Upsample minority classes by replicating samples.
    
    This strategy creates a new dataset where minority class samples are
    replicated to match the frequency of the majority class.
    
    Args:
        labels: Binary label matrix (N, C)
        target_ratio: Target ratio for minority classes
                      - 'match_max': Replicate to match majority class count
                      - float (0-1): Replicate to achieve this ratio of majority
        min_replications: Minimum times to replicate each sample (default: 1)
        max_replications: Maximum times to replicate each sample (default: 10)
        
    Example:
        >>> labels = np.array([[1,0,0], [1,0,0], [0,1,0], [0,0,1]])
        >>> upsampler = UpsamplingStrategy(labels, target_ratio='match_max')
        >>> new_indices = upsampler.get_upsampled_indices()
        >>> # new_indices might be: [0, 1, 2, 2, 3, 3] (replicated minority)
    """
    
    def __init__(
        self,
        labels: np.ndarray,
        target_ratio: str | float = 'match_max',
        min_replications: int = 1,
        max_replications: int = 10
    ):
        self.labels = labels
        self.target_ratio = target_ratio
        self.min_replications = min_replications
        self.max_replications = max_replications
        
        self.num_samples = len(labels)
        self.num_classes = labels.shape[1]
        
        # Compute class counts
        self.class_counts = labels.sum(axis=0)
        self.max_count = self.class_counts.max()
        
        logger.info(f"UpsamplingStrategy initialized:")
        logger.info(f"  Target ratio: {target_ratio}")
        logger.info(f"  Class counts: {self.class_counts}")
        logger.info(f"  Max count: {self.max_count}")
    
    def get_upsampled_indices(self) -> np.ndarray:
        """
        Get indices for upsampled dataset.
        
        Returns:
            Array of indices (can contain duplicates)
        """
        # Count samples per class
        sample_replications = np.ones(self.num_samples, dtype=int)
        
        for i in range(self.num_samples):
            # Get positive labels
            positive_labels = np.where(self.labels[i] > 0)[0]
            
            if len(positive_labels) == 0:
                continue
            
            # Find rarest positive label
            counts = self.class_counts[positive_labels]
            min_count = counts.min()
            
            # Compute replication factor
            if self.target_ratio == 'match_max':
                target_count = self.max_count
            else:
                target_count = int(self.max_count * self.target_ratio)
            
            # Replicate to reach target
            replication_factor = int(np.ceil(target_count / min_count))
            replication_factor = np.clip(
                replication_factor,
                self.min_replications,
                self.max_replications
            )
            
            sample_replications[i] = replication_factor
        
        # Create upsampled indices
        upsampled_indices = []
        for i in range(self.num_samples):
            upsampled_indices.extend([i] * sample_replications[i])
        
        upsampled_indices = np.array(upsampled_indices)
        
        logger.info(f"Upsampling complete:")
        logger.info(f"  Original samples: {self.num_samples}")
        logger.info(f"  Upsampled samples: {len(upsampled_indices)}")
        logger.info(f"  Replication stats: min={sample_replications.min()}, "
                   f"max={sample_replications.max()}, "
                   f"mean={sample_replications.mean():.2f}")
        
        return upsampled_indices


def get_class_balanced_indices(
    labels: np.ndarray,
    strategy: str = 'weighted_sampling',
    **kwargs
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """
    Factory function to get class-balanced dataset indices.
    
    Args:
        labels: Binary label matrix (N, C)
        strategy: Balancing strategy
                  - 'weighted_sampling': Return weights for PyTorch WeightedRandomSampler
                  - 'upsample': Return replicated indices
                  - 'none': No balancing (return original indices)
        **kwargs: Strategy-specific parameters
        
    Returns:
        Tuple of (indices, weights)
        - indices: Array of sample indices (may contain duplicates if upsampled)
        - weights: Sampling weights (None if not using weighted sampling)
    
    Example:
        >>> # Weighted sampling
        >>> indices, weights = get_class_balanced_indices(labels, strategy='weighted_sampling', mode='inv')
        >>> sampler = torch.utils.data.WeightedRandomSampler(weights, len(weights))
        
        >>> # Upsampling
        >>> indices, _ = get_class_balanced_indices(labels, strategy='upsample', target_ratio=0.5)
        >>> dataset_upsampled = torch.utils.data.Subset(dataset, indices)
    """
    num_samples = len(labels)
    
    if strategy == 'none':
        return np.arange(num_samples), None
    
    elif strategy == 'weighted_sampling':
        sampler = ClassBalancedSampler(labels, **kwargs)
        weights = sampler.get_sample_weights()
        indices = np.arange(num_samples)
        return indices, weights
    
    elif strategy == 'upsample':
        upsampler = UpsamplingStrategy(labels, **kwargs)
        indices = upsampler.get_upsampled_indices()
        return indices, None
    
    else:
        raise ValueError(f"Unknown strategy: {strategy}")


if __name__ == "__main__":
    # Test upsampling strategies
    print("Testing Class-Balanced Sampling...")
    
    # Create imbalanced dataset
    labels = np.array([
        [1, 0, 0, 0, 0],  # Class 0 (common)
        [1, 0, 0, 0, 0],
        [1, 0, 0, 0, 0],
        [0, 1, 0, 0, 0],  # Class 1 (rare)
        [0, 0, 1, 0, 0],  # Class 2 (very rare)
        [0, 0, 0, 1, 0],  # Class 3 (common)
        [0, 0, 0, 1, 0],
        [0, 0, 0, 0, 1],  # Class 4 (rare)
    ])
    
    print(f"\nOriginal labels:\n{labels}")
    print(f"Class counts: {labels.sum(axis=0)}")
    
    # Test weighted sampling
    print("\n--- Weighted Sampling (inv) ---")
    indices, weights = get_class_balanced_indices(labels, strategy='weighted_sampling', mode='inv')
    print(f"Sample weights: {weights}")
    
    # Test upsampling
    print("\n--- Upsampling (match_max) ---")
    indices, _ = get_class_balanced_indices(labels, strategy='upsample', target_ratio='match_max')
    print(f"Upsampled indices: {indices}")
    print(f"New class counts: {labels[indices].sum(axis=0)}")
    
    print("\n✓ Tests complete!")
