"""
Loss function registry for easy selection and instantiation.
"""

import torch.nn as nn
from typing import Dict, Callable, Type, Any, Optional
import logging

logger = logging.getLogger(__name__)


class LossRegistry:
    """
    Registry for loss function classes.
    
    Allows loss functions to be registered by name and instantiated
    from configuration dictionaries.
    
    Example:
        >>> registry = LossRegistry()
        >>> registry.register("focal_loss", FocalLoss)
        >>> loss_fn = registry.get("focal_loss", num_classes=4, gamma=2.0)
    """
    
    def __init__(self):
        self._registry: Dict[str, Type[nn.Module]] = {}
    
    def register(
        self,
        name: str,
        loss_class: Type[nn.Module],
        description: str = ""
    ) -> None:
        """
        Register a loss function class.
        
        Args:
            name: Unique identifier for the loss function
            loss_class: The loss function class
            description: Optional description
        """
        if name in self._registry:
            logger.warning(f"Loss '{name}' already registered. Overwriting.")
        
        self._registry[name] = {
            "class": loss_class,
            "description": description
        }
        logger.info(f"Registered loss function: {name}")
    
    def get(self, name: str, **kwargs) -> nn.Module:
        """
        Instantiate a loss function from the registry.
        
        Args:
            name: Name of the loss function
            **kwargs: Arguments to pass to constructor
        
        Returns:
            Instance of the loss function
        
        Raises:
            ValueError: If loss function name is not found
        """
        if name not in self._registry:
            available = list(self._registry.keys())
            raise ValueError(
                f"Loss '{name}' not found. Available: {available}"
            )
        
        loss_class = self._registry[name]["class"]
        
        try:
            loss = loss_class(**kwargs)
            logger.info(f"Instantiated loss: {name}")
            return loss
        except Exception as e:
            raise RuntimeError(f"Failed to instantiate '{name}': {str(e)}")
    
    def list_losses(self) -> Dict[str, str]:
        """List all registered losses with descriptions."""
        return {
            name: info["description"]
            for name, info in self._registry.items()
        }
    
    def has_loss(self, name: str) -> bool:
        """Check if a loss function is registered."""
        return name in self._registry


# Global registry
_default_registry = None


def get_registry() -> LossRegistry:
    """Get the global loss registry (singleton)."""
    global _default_registry
    if _default_registry is None:
        _default_registry = LossRegistry()
        _register_default_losses()
    return _default_registry


def _register_default_losses() -> None:
    """Register default loss functions."""
    from .focal_loss import FocalLoss
    from .bce_loss import BCEWithLogitsLossWrapper
    
    registry = _default_registry
    
    registry.register(
        "focal_loss",
        FocalLoss,
        description="Focal Loss for imbalanced classification"
    )
    
    registry.register(
        "bce_with_logits",
        BCEWithLogitsLossWrapper,
        description="Binary Cross-Entropy with Logits"
    )
