"""
Metric computation for multi-label classification.
"""

import numpy as np
import torch
from sklearn.metrics import (
    precision_score, recall_score, f1_score,
    hamming_loss, accuracy_score, multilabel_confusion_matrix
)
from typing import Dict, Tuple, Optional
import logging

logger = logging.getLogger(__name__)


def find_optimal_thresholds(
    logits: np.ndarray,
    targets: np.ndarray,
    class_names: list = None,
    verbose: bool = False
) -> Dict[str, float]:
    """
    Find optimal threshold per class to maximize F1 score.
    
    Args:
        logits: Predicted logits (N, C)
        targets: Ground truth labels (N, C)
        class_names: Optional class names
        verbose: If True, log per-class threshold details
    
    Returns:
        Dictionary mapping class name to optimal threshold
    """
    probabilities = 1 / (1 + np.exp(-logits))  # Sigmoid
    n_classes = probabilities.shape[1]
    
    optimal_thresholds = {}
    
    for c in range(n_classes):
        class_name = class_names[c] if class_names else f"class_{c}"
        
        # Try different thresholds
        best_f1 = 0
        best_threshold = 0.5
        
        # Search from 0.1 to 0.9 in steps of 0.05
        for threshold in np.arange(0.1, 0.91, 0.05):
            preds = (probabilities[:, c] > threshold).astype(int)
            f1 = f1_score(targets[:, c], preds, zero_division=0)
            
            if f1 > best_f1:
                best_f1 = f1
                best_threshold = threshold
        
        optimal_thresholds[class_name] = best_threshold
        if verbose:
            logger.info(f"  {class_name}: threshold={best_threshold:.2f}, F1={best_f1:.4f}")
    
    return optimal_thresholds


def compute_metrics(
    predictions: np.ndarray,
    targets: np.ndarray,
    threshold: float = 0.5,
    class_names: list = None,
    per_class_thresholds: Optional[Dict[str, float]] = None
) -> Dict:
    """
    Compute multi-label classification metrics.
    
    Args:
        predictions: Binary predictions (shape: N x C) or probabilities
        targets: Binary targets (shape: N x C)
        threshold: Default threshold for converting probabilities to binary predictions
        class_names: Optional names for classes
        per_class_thresholds: Optional dict of per-class thresholds
    
    Returns:
        Dictionary with metrics
    """
    
    # Convert probabilities to binary using per-class thresholds if provided
    if per_class_thresholds is not None:
        binary_preds = np.zeros_like(predictions)
        for c in range(predictions.shape[1]):
            class_name = class_names[c] if class_names else f"class_{c}"
            thresh = per_class_thresholds.get(class_name, threshold)
            binary_preds[:, c] = (predictions[:, c] > thresh).astype(int)
        predictions = binary_preds
    else:
        # Check if predictions are probabilities (0-1 range) or already binary
        if predictions.max() <= 1.0 and predictions.min() >= 0.0:
            # Probabilities - apply threshold
            predictions = (predictions > threshold).astype(int)
        # If already binary (only 0s and 1s), keep as is
    
    # Ensure binary format
    predictions = predictions.astype(int)
    targets = targets.astype(int)
    
    n_classes = predictions.shape[1]
    
    # Confusion matrix per class
    cm_per_class = multilabel_confusion_matrix(targets, predictions)
    
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
        
        # Extract confusion matrix values
        tn, fp, fn, tp = cm_per_class[c].ravel()
        
        per_class_metrics[f"{class_name}_precision"] = precision
        per_class_metrics[f"{class_name}_recall"] = recall
        per_class_metrics[f"{class_name}_f1"] = f1
        per_class_metrics[f"{class_name}_tp"] = int(tp)
        per_class_metrics[f"{class_name}_fp"] = int(fp)
        per_class_metrics[f"{class_name}_tn"] = int(tn)
        per_class_metrics[f"{class_name}_fn"] = int(fn)
        per_class_metrics[f"{class_name}_support"] = int(tp + fn)  # True positives
    
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
    threshold: float = 0.5,
    per_class_thresholds: Optional[Dict[str, float]] = None,
    class_names: list = None
) -> np.ndarray:
    """
    Convert logits to binary predictions.
    
    Args:
        logits: Tensor of shape (N, C)
        threshold: Default classification threshold
        per_class_thresholds: Optional dict of per-class thresholds
        class_names: Optional class names
    
    Returns:
        Binary predictions array (N, C)
    """
    probabilities = torch.sigmoid(logits).detach().cpu().numpy()
    
    if per_class_thresholds is not None:
        predictions = np.zeros_like(probabilities)
        for c in range(probabilities.shape[1]):
            class_name = class_names[c] if class_names else f"class_{c}"
            thresh = per_class_thresholds.get(class_name, threshold)
            predictions[:, c] = (probabilities[:, c] > thresh).astype(int)
        return predictions
    else:
        return (probabilities > threshold).astype(int)


def log_per_class_metrics(metrics: Dict, class_names: list, logger_obj=None):
    """
    Pretty print per-class metrics.
    
    Args:
        metrics: Dictionary with metrics
        class_names: List of class names
        logger_obj: Logger to use (defaults to module logger)
    """
    if logger_obj is None:
        logger_obj = logger
    
    logger_obj.info("\n" + "="*80)
    logger_obj.info("PER-CLASS METRICS")
    logger_obj.info("="*80)
    
    # Header
    logger_obj.info(f"{'Class':<15} {'Precision':>10} {'Recall':>10} {'F1':>10} {'Support':>10} {'TP':>6} {'FP':>6} {'TN':>6} {'FN':>6}")
    logger_obj.info("-"*80)
    
    for class_name in class_names:
        precision = metrics.get(f"{class_name}_precision", 0)
        recall = metrics.get(f"{class_name}_recall", 0)
        f1 = metrics.get(f"{class_name}_f1", 0)
        support = metrics.get(f"{class_name}_support", 0)
        tp = metrics.get(f"{class_name}_tp", 0)
        fp = metrics.get(f"{class_name}_fp", 0)
        tn = metrics.get(f"{class_name}_tn", 0)
        fn = metrics.get(f"{class_name}_fn", 0)
        
        logger_obj.info(
            f"{class_name:<15} {precision:>10.4f} {recall:>10.4f} {f1:>10.4f} "
            f"{support:>10} {tp:>6} {fp:>6} {tn:>6} {fn:>6}"
        )
    
    logger_obj.info("-"*80)
    logger_obj.info(f"{'MACRO AVG':<15} {metrics['precision_macro']:>10.4f} {metrics['recall_macro']:>10.4f} {metrics['f1_macro']:>10.4f}")
    logger_obj.info(f"{'MICRO AVG':<15} {metrics['precision_micro']:>10.4f} {metrics['recall_micro']:>10.4f} {metrics['f1_micro']:>10.4f}")
    logger_obj.info("="*80 + "\n")
