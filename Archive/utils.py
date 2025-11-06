"""
Utility functions for Severstal Steel Defect Classification project.
"""

import os
import random
import numpy as np
import torch
from typing import Dict, Any, Optional
import logging


def set_seed(seed: int = 42) -> None:
    """Set seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def setup_logging(level: str = "INFO") -> logging.Logger:
    """Setup logging configuration."""
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
        ]
    )
    return logging.getLogger(__name__)


def calculate_class_weights(class_counts: Dict[int, int], num_classes: int) -> torch.Tensor:
    """
    Calculate class weights for imbalanced dataset.
    
    Args:
        class_counts: Dictionary mapping class index to count
        num_classes: Total number of classes
    
    Returns:
        Tensor of class weights
    """
    total_samples = sum(class_counts.values())
    weights = torch.ones(num_classes)
    
    for class_idx in range(num_classes):
        if class_idx in class_counts and class_counts[class_idx] > 0:
            weights[class_idx] = total_samples / (num_classes * class_counts[class_idx])
    
    return weights


def create_dirs(paths: Dict[str, str]) -> None:
    """Create directories if they don't exist."""
    for name, path in paths.items():
        os.makedirs(path, exist_ok=True)
        print(f"Created/verified directory for {name}: {path}")


def format_metrics(metrics: Dict[str, float], prefix: str = "") -> str:
    """Format metrics dictionary for logging."""
    formatted = []
    for key, value in metrics.items():
        if isinstance(value, float):
            formatted.append(f"{prefix}{key}: {value:.4f}")
        else:
            formatted.append(f"{prefix}{key}: {value}")
    return ", ".join(formatted)


def count_parameters(model: torch.nn.Module) -> tuple:
    """Count total and trainable parameters in model."""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total_params, trainable_params


def save_model_info(model: torch.nn.Module, save_path: str) -> None:
    """Save model architecture and parameter info."""
    total_params, trainable_params = count_parameters(model)
    
    info = {
        "model_class": model.__class__.__name__,
        "total_parameters": total_params,
        "trainable_parameters": trainable_params,
        "model_size_mb": total_params * 4 / (1024 * 1024),  # Assuming float32
    }
    
    import json
    with open(save_path, 'w') as f:
        json.dump(info, f, indent=2)
    
    print(f"Model info saved to: {save_path}")
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Model size: {info['model_size_mb']:.2f} MB")


class AverageMeter:
    """Computes and stores the average and current value."""
    
    def __init__(self, name: str, fmt: str = ':f'):
        self.name = name
        self.fmt = fmt
        self.reset()
    
    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
    
    def update(self, val: float, n: int = 1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
    
    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


def get_device(device_str: str = "auto") -> torch.device:
    """Get appropriate device for training."""
    if device_str == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device_str)
    
    if device.type == "cuda":
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    else:
        print("Using CPU")
    
    return device