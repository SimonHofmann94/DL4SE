"""
Metric computation for multi-label classification.
"""

import numpy as np
import torch
from sklearn.metrics import (
    precision_score, recall_score, f1_score,
    hamming_loss, accuracy_score
)
from typing import Dict, Tuple
import logging

logger = logging.getLogger(__name__)


def compute_metrics(
    predictions: np.ndarray,
    targets: np.ndarray,
    threshold: float = 0.5,
    class_names: list = None
) -> Dict:
    """
    Compute multi-label classification metrics.
    
    Args:
        predictions: Binary predictions (shape: N x C) or logits
        targets: Binary targets (shape: N x C)
        threshold: Threshold for converting logits to binary predictions
        class_names: Optional names for classes
    
    Returns:
        Dictionary with metrics
    """
    
    # Convert logits to binary if needed
    if predictions.max() > 1:
        predictions = (predictions > threshold).astype(int)
    
    # Ensure binary format
    predictions = (predictions > 0).astype(int)
    targets = (targets > 0).astype(int)
    
    n_classes = predictions.shape[1]
    
    # Per-class metrics
    per_class_metrics = {}
    for c in range(n_classes):
        class_name = class_names[c] if class_names else f"class_{c}"
        
        precision = precision_score(
            targets[:, c], predictions[:, c], zero_division=0
        )
        recall = recall_score(
            targets[:, c], predictions[:, c], zero_division=0
        )
        f1 = f1_score(targets[:, c], predictions[:, c], zero_division=0)
        
        per_class_metrics[f"{class_name}_precision"] = precision
        per_class_metrics[f"{class_name}_recall"] = recall
        per_class_metrics[f"{class_name}_f1"] = f1
    
    # Macro-average
    precision_macro = precision_score(targets, predictions, average='macro', zero_division=0)
    recall_macro = recall_score(targets, predictions, average='macro', zero_division=0)
    f1_macro = f1_score(targets, predictions, average='macro', zero_division=0)
    
    # Micro-average
    precision_micro = precision_score(targets, predictions, average='micro', zero_division=0)
    recall_micro = recall_score(targets, predictions, average='micro', zero_division=0)
    f1_micro = f1_score(targets, predictions, average='micro', zero_division=0)
    
    # Accuracy and Hamming loss
    accuracy = accuracy_score(targets, predictions)
    hamming = hamming_loss(targets, predictions)
    
    metrics = {
        "precision_macro": precision_macro,
        "recall_macro": recall_macro,
        "f1_macro": f1_macro,
        "precision_micro": precision_micro,
        "recall_micro": recall_micro,
        "f1_micro": f1_micro,
        "accuracy": accuracy,
        "hamming_loss": hamming,
        **per_class_metrics
    }
    
    return metrics


def get_predictions_from_logits(
    logits: torch.Tensor,
    threshold: float = 0.5
) -> np.ndarray:
    """
    Convert logits to binary predictions.
    
    Args:
        logits: Tensor of shape (N, C)
        threshold: Classification threshold
    
    Returns:
        Binary predictions array (N, C)
    """
    probabilities = torch.sigmoid(logits).detach().cpu().numpy()
    return (probabilities > threshold).astype(int)
