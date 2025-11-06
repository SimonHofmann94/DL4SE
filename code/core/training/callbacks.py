"""
Callbacks for training (early stopping, checkpointing, etc.)
"""

import torch
import os
import logging
from typing import Optional

logger = logging.getLogger(__name__)


class EarlyStoppingCallback:
    """
    Early stopping callback to stop training when validation metric plateaus.
    
    Args:
        patience: Number of epochs with no improvement to wait before stopping
        min_delta: Minimum change in metric to count as improvement
        metric_name: Name of metric to monitor (for logging)
        save_best_only: If True, save model when metric improves
        checkpoint_dir: Directory to save best model
    """
    
    def __init__(
        self,
        patience: int = 10,
        min_delta: float = 0.0,
        metric_name: str = "val_f1_macro",
        save_best_only: bool = True,
        checkpoint_dir: Optional[str] = None,
        mode: str = "max"  # "max" for metrics like F1, "min" for loss
    ):
        self.patience = patience
        self.min_delta = min_delta
        self.metric_name = metric_name
        self.save_best_only = save_best_only
        self.checkpoint_dir = checkpoint_dir
        self.mode = mode
        
        self.best_metric = float('-inf') if mode == "max" else float('inf')
        self.patience_counter = 0
        self.best_epoch = 0
    
    def __call__(
        self,
        current_metric: float,
        epoch: int,
        model: torch.nn.Module = None
    ) -> bool:
        """
        Check if training should stop.
        
        Args:
            current_metric: Current value of the monitored metric
            epoch: Current epoch number
            model: Model to save if checkpoint_dir is set
        
        Returns:
            True if training should stop, False otherwise
        """
        
        # Check if metric improved
        if self.mode == "max":
            improved = current_metric > self.best_metric + self.min_delta
        else:
            improved = current_metric < self.best_metric - self.min_delta
        
        if improved:
            self.best_metric = current_metric
            self.patience_counter = 0
            self.best_epoch = epoch
            
            # Save checkpoint if requested
            if self.save_best_only and model is not None and self.checkpoint_dir:
                self._save_checkpoint(model, epoch, current_metric)
            
            logger.info(
                f"Epoch {epoch}: {self.metric_name} improved to {current_metric:.4f}"
            )
            return False
        
        else:
            self.patience_counter += 1
            logger.info(
                f"Epoch {epoch}: No improvement ({current_metric:.4f}). "
                f"Patience: {self.patience_counter}/{self.patience}"
            )
            
            if self.patience_counter >= self.patience:
                logger.info(
                    f"Early stopping triggered! "
                    f"Best metric: {self.best_metric:.4f} (epoch {self.best_epoch})"
                )
                return True
        
        return False
    
    def _save_checkpoint(
        self,
        model: torch.nn.Module,
        epoch: int,
        metric_value: float
    ) -> None:
        """Save model checkpoint."""
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        
        checkpoint_path = os.path.join(
            self.checkpoint_dir,
            f"best_model_epoch{epoch}_{self.metric_name}_{metric_value:.4f}.pt"
        )
        
        torch.save(model.state_dict(), checkpoint_path)
        logger.info(f"Saved checkpoint: {checkpoint_path}")
    
    def reset(self) -> None:
        """Reset callback state."""
        self.best_metric = float('-inf') if self.mode == "max" else float('inf')
        self.patience_counter = 0
        self.best_epoch = 0
