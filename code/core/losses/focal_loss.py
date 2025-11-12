"""
Focal Loss implementation with dynamic parameter computation.

Focal Loss addresses class imbalance by down-weighting easy examples and
focusing on hard negatives, making it ideal for imbalanced datasets.

Two alpha computation methods are supported:
1. Standard inverse frequency weighting
2. Class-Balanced Loss using Effective Number of Samples (Cui et al., CVPR 2019)

Papers:
- Focal Loss: https://arxiv.org/abs/1708.02002
- Class-Balanced Loss: https://arxiv.org/abs/1901.05555
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, Optional
import logging
import numpy as np

from .base import BaseLoss

logger = logging.getLogger(__name__)


class FocalLoss(BaseLoss):
    """
    Focal Loss for multi-label classification with class imbalance.
    
    Formula:
        FL(p_t) = -α_t * (1 - p_t)^γ * log(p_t)
    
    Args:
        num_classes: Number of classes
        alpha: Class weighting (float or list). If float, applied uniformly.
               If None or 'dynamic', computed dynamically from class frequencies.
        gamma: Focusing parameter (default: 2.0). Higher values increase focus on hard examples.
        reduction: 'mean', 'sum', or 'none' (default: 'mean')
        use_logits: If True, input is logits; if False, input is probabilities (default: True)
        use_effective_num: If True, use Class-Balanced Loss (Cui et al., CVPR 2019).
                          If False, use standard inverse frequency weighting.
        beta: Hyperparameter for effective number calculation (default: 0.9999)
              Typical values: 0.999 (less aggressive), 0.9999 (more aggressive)
    
    Example:
        >>> # Use Class-Balanced Loss with gamma=3.0
        >>> loss_fn = FocalLoss(num_classes=5, alpha='dynamic', gamma=3.0, 
        ...                     use_effective_num=True, beta=0.9999)
        >>> logits = torch.randn(32, 5)  # Model output (B, C)
        >>> targets = torch.randint(0, 2, (32, 5)).float()  # Binary labels
        >>> loss = loss_fn(logits, targets)
    """
    
    def __init__(
        self,
        num_classes: int,
        alpha: Optional[float | str | list] = 'dynamic',
        gamma: float = 2.0,
        reduction: str = 'mean',
        use_logits: bool = True,
        use_effective_num: bool = True,
        beta: float = 0.9999
    ):
        super().__init__(num_classes)
        self.gamma = gamma
        self.reduction = reduction
        self.use_logits = use_logits
        self.use_effective_num = use_effective_num
        self.beta = beta
        
        # Store alpha configuration
        self.alpha_config = alpha
        
        # Handle different alpha initialization modes
        if isinstance(alpha, list):
            # Manual alpha values provided as list
            if len(alpha) != num_classes:
                raise ValueError(
                    f"Alpha list length ({len(alpha)}) must match num_classes ({num_classes})"
                )
            self.register_buffer('alpha_buffer', torch.tensor(alpha, dtype=torch.float32))
            self.alpha_computed = True
            logger.info(f"Using manual alpha values: {alpha}")
        elif isinstance(alpha, (int, float)):
            # Single alpha value for all classes
            self.register_buffer('alpha_buffer', torch.tensor([alpha] * num_classes, dtype=torch.float32))
            self.alpha_computed = True
            logger.info(f"Using uniform alpha value: {alpha}")
        else:
            # Will be set during first forward pass if alpha is 'dynamic'
            self.register_buffer('alpha_buffer', None)
            self.alpha_computed = False
    
    def _compute_alpha_from_frequencies(
        self,
        targets: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute alpha weights from class frequencies in targets.
        
        Two methods available:
        1. Standard inverse frequency: α_c = 1 - (freq_c / total_samples)
        2. Class-Balanced Loss using Effective Number (Cui et al., CVPR 2019):
           α_c = (1 - β) / (1 - β^n_c)
           Paper: https://arxiv.org/abs/1901.05555
        
        The effective number approach provides smoother weighting for highly
        imbalanced datasets by accounting for overlap in data coverage.
        
        Args:
            targets: Binary targets of shape (B, C)
        
        Returns:
            Alpha weights tensor of shape (C,)
        """
        # Compute positive counts per class
        pos_counts = targets.sum(dim=0)  # (C,)
        total_samples = targets.shape[0]
        
        # Positive ratio per class
        pos_ratio = pos_counts / total_samples  # (C,)
        
        if self.use_effective_num:
            # Method 2: Class-Balanced Loss (Cui et al., CVPR 2019)
            # Uses effective number of samples to compute alpha
            # Formula: E_n = (1 - β^n) / (1 - β)
            #          α_c = (1 - β) / (1 - β^n_c)
            
            # Effective number calculation
            # For small n_c: E_n ≈ 1
            # For large n_c: E_n ≈ n_c
            effective_num = (1.0 - torch.pow(self.beta, pos_counts)) / (1.0 - self.beta)
            
            # Alpha is inversely proportional to effective number
            # Classes with fewer samples get higher alpha
            alpha = (1.0 - self.beta) / (1.0 - torch.pow(self.beta, pos_counts) + 1e-7)
            
            # Normalize to [0, 1] range
            alpha = alpha / alpha.sum() * self.num_classes
            
            logger.info(
                f"Computed focal loss alpha using Class-Balanced Loss (Cui et al., CVPR 2019):\n"
                f"  Beta: {self.beta}\n"
                f"  Positive counts per class: {pos_counts.cpu().numpy()}\n"
                f"  Effective numbers: {effective_num.cpu().numpy()}\n"
                f"  Alpha values (normalized): {alpha.cpu().numpy()}"
            )
        else:
            # Method 1: Standard inverse frequency
            # α_pos = 1 - freq means higher alpha for rare classes
            alpha = (1.0 - pos_ratio)
            
            # Clamp to reasonable range [0.01, 0.99]
            alpha = torch.clamp(alpha, min=0.01, max=0.99)
            
            logger.info(
                f"Computed focal loss alpha from class frequencies (standard method):\n"
                f"  Positive ratios per class: {pos_ratio.cpu().numpy()}\n"
                f"  Alpha values: {alpha.cpu().numpy()}"
            )
        
        return alpha
    
    def forward(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        class_frequencies: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Compute focal loss.
        
        Args:
            predictions: Model predictions (logits) of shape (B, C)
            targets: Binary target labels of shape (B, C)
            class_frequencies: Optional class frequencies for dynamic alpha computation
        
        Returns:
            Scalar loss value
        """
        # Compute alpha if needed
        if self.alpha_config == 'dynamic' and not self.alpha_computed:
            self.alpha_buffer = self._compute_alpha_from_frequencies(targets)
            self.alpha_computed = True
        
        # Convert logits to probabilities if needed
        if self.use_logits:
            # Use numerically stable log-sum-exp trick
            p = torch.sigmoid(predictions)
        else:
            p = predictions
        
        # Ensure p is in valid range
        p = torch.clamp(p, min=1e-7, max=1 - 1e-7)
        
        # Compute focal term: (1 - p_t)^γ
        # For multi-label: use p for positive targets, (1-p) for negative
        ce = F.binary_cross_entropy(p, targets, reduction='none')
        p_t = torch.where(targets == 1, p, 1 - p)
        focal_weight = (1 - p_t) ** self.gamma
        
        # Apply alpha weighting if available
        if self.alpha_buffer is not None:
            alpha = self.alpha_buffer.to(predictions.device)
            # Apply alpha: higher weight for rare classes
            alpha_t = torch.where(targets == 1, alpha, 1 - alpha)
            focal_weight = alpha_t * focal_weight
        
        # Focal loss
        fl = focal_weight * ce
        
        # Apply reduction
        if self.reduction == 'mean':
            return fl.mean()
        elif self.reduction == 'sum':
            return fl.sum()
        else:
            return fl
    
    def get_params(self) -> Dict[str, Any]:
        """Get focal loss hyperparameters for logging."""
        alpha_val = "dynamic"
        if self.alpha_computed and self.alpha_buffer is not None:
            alpha_val = self.alpha_buffer.cpu().numpy().tolist()
        elif isinstance(self.alpha_config, (int, float)):
            alpha_val = self.alpha_config
        elif isinstance(self.alpha_config, list):
            alpha_val = self.alpha_config
        
        return {
            "gamma": self.gamma,
            "alpha": alpha_val,
            "reduction": self.reduction
        }
    
    def reset_alpha(self) -> None:
        """Reset computed alpha buffer for use with new dataset."""
        self.alpha_buffer = None
        self.alpha_computed = False
        logger.info("Focal loss alpha buffer reset")


if __name__ == "__main__":
    # Test focal loss
    B, C = 32, 4
    
    # Create synthetic data
    logits = torch.randn(B, C)
    targets = torch.randint(0, 2, (B, C)).float()
    
    # Create loss function with dynamic alpha
    loss_fn = FocalLoss(
        num_classes=C,
        alpha='dynamic',
        gamma=2.0,
        reduction='mean'
    )
    
    # Compute loss
    loss = loss_fn(logits, targets)
    print(f"Loss: {loss.item():.4f}")
    print(f"Loss params: {loss_fn.get_params()}")
    print(f"Loss info: {loss_fn.log_info()}")
