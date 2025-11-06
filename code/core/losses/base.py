"""
Base class for loss functions.

All loss function implementations should inherit from BaseLoss to ensure
consistent interface for training, logging, and UI integration.
"""

import torch
import torch.nn as nn
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)


class BaseLoss(nn.Module, ABC):
    """
    Abstract base class for all loss functions.
    
    Provides interface for:
    - Loss computation
    - Parameter management for UI display
    - Logging metadata about the loss
    
    Subclasses must implement:
    - forward(): Compute the loss
    - get_params(): Return loss hyperparameters for logging
    """
    
    def __init__(self, num_classes: int):
        super().__init__()
        self.num_classes = num_classes
    
    @abstractmethod
    def forward(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        **kwargs
    ) -> torch.Tensor:
        """
        Compute loss between predictions and targets.
        
        Args:
            predictions: Model predictions (shape: B x C or B x C x H x W)
            targets: Ground truth targets (same shape as predictions)
            **kwargs: Additional arguments specific to loss function
        
        Returns:
            Scalar loss tensor
        """
        pass
    
    @abstractmethod
    def get_params(self) -> Dict[str, Any]:
        """
        Get loss hyperparameters for logging and display.
        
        Returns:
            Dictionary of parameter names to values
        """
        pass
    
    def get_name(self) -> str:
        """
        Get human-readable name of the loss function.
        
        Returns:
            String name
        """
        return self.__class__.__name__
    
    def log_info(self) -> str:
        """
        Get formatted string with loss information.
        
        Returns:
            Formatted string for logging
        """
        name = self.get_name()
        params = self.get_params()
        params_str = ", ".join(f"{k}={v}" for k, v in params.items())
        return f"{name}({params_str})"
