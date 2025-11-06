"""
Model registry for easy model instantiation and discovery.

The registry allows models to be registered and retrieved by name,
making it easy to swap models in configuration files and UI.
"""

import torch.nn as nn
from typing import Dict, Callable, Any, Type
import logging

logger = logging.getLogger(__name__)


class ModelRegistry:
    """
    Registry for managing available model architectures.
    
    Features:
    - Register models with factory functions
    - Retrieve models by name
    - List all available models
    - Validate configurations before instantiation
    
    Example:
        >>> registry = ModelRegistry()
        >>> registry.register("convnext_tiny_cbam", ConvNextTinyCBAM)
        >>> model = registry.get("convnext_tiny_cbam", num_classes=4)
    """
    
    def __init__(self):
        self._registry: Dict[str, Callable] = {}
    
    def register(
        self,
        name: str,
        model_class: Type[nn.Module],
        description: str = ""
    ) -> None:
        """
        Register a model class in the registry.
        
        Args:
            name: Unique identifier for the model
            model_class: The model class (should inherit from nn.Module)
            description: Optional description of the model
        """
        if name in self._registry:
            logger.warning(f"Model '{name}' already registered. Overwriting.")
        
        self._registry[name] = {
            "class": model_class,
            "description": description
        }
        logger.info(f"Registered model: {name}")
    
    def get(self, name: str, **kwargs) -> nn.Module:
        """
        Instantiate a model from the registry.
        
        Args:
            name: Name of the model to retrieve
            **kwargs: Arguments to pass to the model constructor
        
        Returns:
            Instance of the requested model
        
        Raises:
            ValueError: If model name is not found in registry
        """
        if name not in self._registry:
            available = list(self._registry.keys())
            raise ValueError(
                f"Model '{name}' not found in registry. "
                f"Available models: {available}"
            )
        
        model_info = self._registry[name]
        model_class = model_info["class"]
        
        try:
            model = model_class(**kwargs)
            logger.info(f"Instantiated model: {name} with kwargs: {kwargs}")
            return model
        except Exception as e:
            raise RuntimeError(
                f"Failed to instantiate model '{name}': {str(e)}"
            )
    
    def list_models(self) -> Dict[str, str]:
        """
        List all registered models with descriptions.
        
        Returns:
            Dictionary mapping model names to descriptions
        """
        return {
            name: info["description"]
            for name, info in self._registry.items()
        }
    
    def has_model(self, name: str) -> bool:
        """
        Check if a model is registered.
        
        Args:
            name: Name of the model
        
        Returns:
            True if model is registered, False otherwise
        """
        return name in self._registry
    
    def clear(self) -> None:
        """Clear all registered models."""
        self._registry.clear()
        logger.info("Model registry cleared")


# Global registry instance
_default_registry = None


def get_registry() -> ModelRegistry:
    """
    Get the default model registry (singleton pattern).
    
    Returns:
        The global ModelRegistry instance
    """
    global _default_registry
    if _default_registry is None:
        _default_registry = ModelRegistry()
        _register_default_models()
    return _default_registry


def _register_default_models() -> None:
    """Register all default models on module import."""
    from .backbones import ConvNextTinyCBAM
    
    registry = _default_registry
    
    registry.register(
        "convnext_tiny_cbam",
        ConvNextTinyCBAM,
        description="ConvNext-Tiny with CBAM modules at stages 3-4 for defect detection"
    )
