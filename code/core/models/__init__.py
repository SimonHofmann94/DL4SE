"""
Model module containing architectures and model registry.
"""

from .backbones import ConvNextTinyCBAM
from .registry import ModelRegistry

__all__ = [
    "ConvNextTinyCBAM",
    "ModelRegistry",
]
