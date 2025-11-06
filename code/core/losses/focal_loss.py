"""
Focal Loss implementation with dynamic parameter computation.

Focal Loss addresses class imbalance by down-weighting easy examples and
focusing on hard negatives, making it ideal for imbalanced datasets.

Paper: https://arxiv.org/abs/1708.02002
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
               If None, computed dynamically from class frequencies.
        gamma: Focusing parameter (default: 2.0)
        reduction: 'mean', 'sum', or 'none' (default: 'mean')
        use_logits: If True, input is logits; if False, input is probabilities (default: True)
    
    Example:
        >>> loss_fn = FocalLoss(num_classes=4, alpha='dynamic', gamma=2.0)
        >>> logits = torch.randn(32, 4)  # Model output (B, C)
        >>> targets = torch.randint(0, 2, (32, 4)).float()  # Binary labels
        >>> loss = loss_fn(logits, targets)
    """
    
    def __init__(
        self,
        num_classes: int,
        alpha: Optional[float | str | list] = 'dynamic',
        gamma: float = 2.0,
        reduction: str = 'mean',
        use_logits: bool = True
    ):
        super().__init__(num_classes)
        self.gamma = gamma
        self.reduction = reduction
        self.use_logits = use_logits
        
        # Store alpha configuration
        self.alpha_config = alpha
        
        # Will be set during first forward pass if alpha is 'dynamic'
        self.register_buffer('alpha_buffer', None)
        self.alpha_computed = False
    
    def _compute_alpha_from_frequencies(
        self,
        targets: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute alpha weights from class frequencies in targets.
        
        For highly imbalanced classes, α should be higher for rare classes.
        Formula: α_c = 1 - (freq_c / total_samples)
        
        Args:
            targets: Binary targets of shape (B, C)
        
        Returns:
            Alpha weights tensor of shape (C,)
        """
        # Compute positive ratio per class
        pos_counts = targets.sum(dim=0)  # (C,)
        total_samples = targets.shape[0]
        
        # Positive ratio per class
        pos_ratio = pos_counts / total_samples  # (C,)
        
        # Alpha for positive class: higher for rare classes
        # α_pos = 1 if all samples have this class, else computed based on frequency
        alpha = (1.0 - pos_ratio)
        
        # Clamp to reasonable range [0.01, 0.99]
        alpha = torch.clamp(alpha, min=0.01, max=0.99)
        
        logger.info(
            f"Computed focal loss alpha from class frequencies:\n"
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
