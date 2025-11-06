"""
BCE (Binary Cross-Entropy) Loss wrapper for multi-label classification.

Provides a consistent interface with other loss functions for comparison.
"""

import torch
import torch.nn as nn
from typing import Dict, Any, Optional
import logging

from .base import BaseLoss

logger = logging.getLogger(__name__)


class BCEWithLogitsLossWrapper(BaseLoss):
    """
    Wrapper for BCEWithLogitsLoss with support for positive class weights.
    
    BCEWithLogitsLoss combines a sigmoid layer with BCE loss numerically stable way.
    Supports pos_weight to handle class imbalance.
    
    Args:
        num_classes: Number of classes
        pos_weight: Positive class weights (shape: (C,) or None).
                   If None, uniform weights (1.0) are used.
        reduction: 'mean', 'sum', or 'none' (default: 'mean')
    
    Example:
        >>> loss_fn = BCEWithLogitsLossWrapper(num_classes=4)
        >>> logits = torch.randn(32, 4)
        >>> targets = torch.randint(0, 2, (32, 4)).float()
        >>> loss = loss_fn(logits, targets)
    """
    
    def __init__(
        self,
        num_classes: int,
        pos_weight: Optional[torch.Tensor] = None,
        reduction: str = 'mean'
    ):
        super().__init__(num_classes)
        self.reduction = reduction
        self.pos_weight = pos_weight
        
        # Create underlying loss function
        self.bce_loss = nn.BCEWithLogitsLoss(
            pos_weight=pos_weight,
            reduction=reduction
        )
    
    def forward(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute BCE loss with logits.
        
        Args:
            predictions: Model predictions (logits) of shape (B, C)
            targets: Binary target labels of shape (B, C)
        
        Returns:
            Scalar loss value
        """
        return self.bce_loss(predictions, targets)
    
    def get_params(self) -> Dict[str, Any]:
        """Get loss hyperparameters for logging."""
        pos_weight_val = None
        if self.pos_weight is not None:
            pos_weight_val = self.pos_weight.cpu().numpy().tolist()
        
        return {
            "pos_weight": pos_weight_val,
            "reduction": self.reduction
        }


if __name__ == "__main__":
    # Test BCE loss
    B, C = 32, 4
    
    # Create synthetic data
    logits = torch.randn(B, C)
    targets = torch.randint(0, 2, (B, C)).float()
    
    # Create loss function
    loss_fn = BCEWithLogitsLossWrapper(
        num_classes=C,
        pos_weight=None,
        reduction='mean'
    )
    
    # Compute loss
    loss = loss_fn(logits, targets)
    print(f"Loss: {loss.item():.4f}")
    print(f"Loss params: {loss_fn.get_params()}")
    print(f"Loss info: {loss_fn.log_info()}")
