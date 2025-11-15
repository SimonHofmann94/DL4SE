"""
Main trainer class for coordinating the training pipeline.

This module orchestrates model training, validation, testing, checkpointing,
logging, and experiment tracking.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR
import numpy as np
import logging
import json
import os
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Tuple, Optional, List

from .metrics import compute_metrics, get_predictions_from_logits, find_optimal_thresholds, log_per_class_metrics
from .callbacks import EarlyStoppingCallback

logger = logging.getLogger(__name__)


class Trainer:
    """
    Trainer class for managing the complete training pipeline.
    
    Responsibilities:
    - Load data and model
    - Execute training loop
    - Validate on validation set
    - Evaluate on test set
    - Track experiments and metrics
    - Save checkpoints and results
    
    Args:
        model: The neural network model
        loss_fn: Loss function
        train_loader: Training DataLoader
        val_loader: Validation DataLoader
        test_loader: Test DataLoader
        optimizer: Optimizer (e.g., Adam, SGD)
        scheduler: Learning rate scheduler
        device: torch device
        experiment_dir: Directory to save experiment results
        class_names: Names of classes for logging
    """
    
    def __init__(
        self,
        model: nn.Module,
        loss_fn: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        test_loader: DataLoader,
        optimizer: optim.Optimizer,
        scheduler: object = None,
        device: torch.device = None,
        experiment_dir: str = "experiments/results",
        class_names: List[str] = None,
        config: Optional[Dict[str, Any]] = None,
        split_info: Optional[Dict[str, Any]] = None,
    ):
        self.model = model
        self.loss_fn = loss_fn
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.class_names = class_names or [f"class_{i}" for i in range(4)]
        self.config = config  # Store config for later saving
        self.split_info = split_info  # Store split statistics
        
        # Setup experiment directory
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.experiment_name = f"experiment_{timestamp}"
        self.experiment_dir = Path(experiment_dir) / self.experiment_name
        self.experiment_dir.mkdir(parents=True, exist_ok=True)
        
        # Checkpoint directory
        self.checkpoint_dir = self.experiment_dir / "checkpoints"
        self.checkpoint_dir.mkdir(exist_ok=True)
        
        # Move model to device
        self.model.to(self.device)
        
        # Training state
        self.best_val_metric = float('-inf')
        self.best_epoch = 0
        self.train_history = {
            "epoch": [],
            "train_loss": [],
            "val_loss": [],
        }
        self.metrics_history = {
            "epoch": [],
            "train_metrics": [],
            "val_metrics": [],
            "test_metrics": [],
        }
        
        logger.info(f"Trainer initialized. Experiment: {self.experiment_name}")
        logger.info(f"Model: {type(model).__name__}")
        logger.info(f"Loss: {loss_fn.get_name()}")
        logger.info(f"Device: {self.device}")
    
    def train(
        self,
        num_epochs: int = 50,
        early_stopping_patience: int = 15,
        warmup_epochs: int = 5,
        log_interval: int = 10,
        threshold: float = 0.5
    ) -> Dict[str, Any]:
        """
        Main training loop.
        
        Args:
            num_epochs: Total epochs to train
            early_stopping_patience: Patience for early stopping
            warmup_epochs: Linear warmup epochs
            log_interval: Log metrics every N batches
            threshold: Classification threshold for evaluation
        
        Returns:
            Dictionary with training results and metadata
        """
        
        logger.info(f"Starting training for {num_epochs} epochs...")
        logger.info(f"Warmup: {warmup_epochs}, Early stopping patience: {early_stopping_patience}")
        
        # Setup learning rate scheduler with warmup
        if self.scheduler is None:
            base_scheduler = CosineAnnealingLR(self.optimizer, T_max=num_epochs - warmup_epochs)
            scheduler = torch.optim.lr_scheduler.SequentialLR(
                self.optimizer,
                schedulers=[
                    LinearLR(self.optimizer, start_factor=0.1, total_iters=warmup_epochs),
                    base_scheduler
                ],
                milestones=[warmup_epochs]
            )
        else:
            scheduler = self.scheduler
        
        # Early stopping
        early_stopping = EarlyStoppingCallback(
            patience=early_stopping_patience,
            metric_name="val_f1_macro",
            save_best_only=True,
            checkpoint_dir=str(self.checkpoint_dir),
            mode="max"
        )
        
        # Training loop
        for epoch in range(num_epochs):
            logger.info(f"\n{'='*60}")
            logger.info(f"Epoch {epoch + 1}/{num_epochs}")
            logger.info(f"{'='*60}")
            
            # Training phase
            train_loss = self._train_epoch(
                epoch=epoch,
                log_interval=log_interval
            )
            
            # Validation phase - using fixed threshold (no optimization)
            val_loss, val_metrics = self._validate_epoch(
                threshold=threshold,
                learn_thresholds=False  # Use fixed threshold (0.5) - no optimization
            )
            
            # Record history
            self.train_history["epoch"].append(epoch + 1)
            self.train_history["train_loss"].append(train_loss)
            self.train_history["val_loss"].append(val_loss)
            
            self.metrics_history["epoch"].append(epoch + 1)
            self.metrics_history["train_metrics"].append(None)  # Computed on test data
            self.metrics_history["val_metrics"].append(val_metrics)
            
            # Log metrics
            self._log_epoch_metrics(epoch + 1, train_loss, val_loss, val_metrics)
            
            # Early stopping
            should_stop = early_stopping(
                current_metric=val_metrics["f1_macro"],
                epoch=epoch + 1,
                model=self.model
            )
            
            # Learning rate step
            if scheduler is not None:
                scheduler.step()
            
            if should_stop:
                logger.info(f"\nEarly stopping at epoch {epoch + 1}")
                break
        
        logger.info("\n" + "="*60)
        logger.info("Training completed!")
        logger.info("="*60)
        
        # Test phase - Standard evaluation with fixed threshold
        logger.info("\n" + "="*60)
        logger.info("TEST EVALUATION")
        logger.info("="*60)
        logger.info(f"Using fixed threshold: {threshold}")
        
        test_metrics = self._test(threshold=threshold)
        
        # Store results
        self.metrics_history["test_metrics"] = [test_metrics]
        
        # Save results (including learned thresholds)
        results = self._compile_results(early_stopping.best_epoch)
        self._save_results(results)
        
        return results
    
    def _train_epoch(self, epoch: int, log_interval: int = 10) -> float:
        """
        Single training epoch.
        
        Args:
            epoch: Current epoch number
            log_interval: Log every N batches
        
        Returns:
            Average training loss
        """
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        
        for batch_idx, (images, targets) in enumerate(self.train_loader):
            images = images.to(self.device)
            targets = targets.to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            outputs = self.model(images)
            loss = self.loss_fn(outputs, targets)
            
            # Backward pass
            loss.backward()
            self.optimizer.step()
            
            # Accumulate loss
            total_loss += loss.item()
            num_batches += 1
            
            # Logging
            if (batch_idx + 1) % log_interval == 0:
                avg_loss = total_loss / num_batches
                logger.debug(
                    f"Batch {batch_idx + 1}/{len(self.train_loader)}: "
                    f"Loss = {avg_loss:.4f}"
                )
        
        avg_loss = total_loss / num_batches
        logger.info(f"Train Loss: {avg_loss:.4f}")
        
        return avg_loss
    
    def _validate_epoch(
        self,
        threshold: float = 0.5,
        learn_thresholds: bool = False
    ) -> Tuple[float, Dict[str, float]]:
        """
        Validation epoch.
        
        Args:
            threshold: Default classification threshold
            learn_thresholds: Whether to learn optimal per-class thresholds
        
        Returns:
            Tuple of (average_loss, metrics_dict)
        """
        self.model.eval()
        total_loss = 0.0
        num_batches = 0
        
        all_logits = []
        all_targets = []
        
        with torch.no_grad():
            for images, targets in self.val_loader:
                images = images.to(self.device)
                targets = targets.to(self.device)
                
                # Forward pass
                outputs = self.model(images)
                loss = self.loss_fn(outputs, targets)
                
                # Accumulate
                total_loss += loss.item()
                num_batches += 1
                
                # Store logits and targets
                all_logits.append(outputs.detach().cpu().numpy())
                all_targets.append(targets.detach().cpu().numpy())
        
        # Concatenate all
        all_logits = np.vstack(all_logits)
        all_targets = np.vstack(all_targets)
        
        # Convert to probabilities and compute metrics with fixed threshold
        probabilities = 1 / (1 + np.exp(-all_logits))  # Sigmoid
        metrics = compute_metrics(
            probabilities,
            all_targets,
            threshold=threshold,
            class_names=self.class_names,
            per_class_thresholds=None  # No threshold optimization
        )
        
        avg_loss = total_loss / num_batches
        
        logger.info(f"Val Loss: {avg_loss:.4f}")
        logger.info(f"Val F1 (macro): {metrics['f1_macro']:.4f} [threshold={threshold}]")
        logger.info(
            f"Precision (macro): {metrics['precision_macro']:.4f}, "
            f"Recall (macro): {metrics['recall_macro']:.4f}"
        )
        
        # Log detailed per-class metrics for validation
        log_per_class_metrics(metrics, self.class_names, logger)
        
        return avg_loss, metrics
    
    def _test(self, threshold: float = 0.5) -> Dict[str, float]:
        """
        Test phase with fixed threshold.
        
        Args:
            threshold: Classification threshold (default 0.5)
        
        Returns:
            Test metrics dictionary
        """
        self.model.eval()
        
        all_logits = []
        all_targets = []
        
        with torch.no_grad():
            for images, targets in self.test_loader:
                images = images.to(self.device)
                targets = targets.to(self.device)
                
                # Forward pass
                outputs = self.model(images)
                
                # Store logits
                all_logits.append(outputs.detach().cpu().numpy())
                all_targets.append(targets.detach().cpu().numpy())
        
        # Concatenate all
        all_logits = np.vstack(all_logits)
        all_targets = np.vstack(all_targets)
        
        # Convert to probabilities
        probabilities = 1 / (1 + np.exp(-all_logits))
        
        # Compute metrics with fixed threshold
        metrics = compute_metrics(
            probabilities,
            all_targets,
            threshold=threshold,
            class_names=self.class_names,
            per_class_thresholds=None  # No per-class optimization
        )
        
        logger.info(
            f"Test F1 (macro): {metrics['f1_macro']:.4f}, "
            f"Precision (macro): {metrics['precision_macro']:.4f}, "
            f"Recall (macro): {metrics['recall_macro']:.4f}"
        )
        
        # Log detailed per-class metrics
        log_per_class_metrics(metrics, self.class_names, logger)
        
        return metrics
    
    def _log_epoch_metrics(
        self,
        epoch: int,
        train_loss: float,
        val_loss: float,
        val_metrics: Dict[str, float]
    ) -> None:
        """Log epoch metrics."""
        logger.info(
            f"\nEpoch {epoch} Summary:"
            f" Train Loss: {train_loss:.4f},"
            f" Val Loss: {val_loss:.4f},"
            f" F1 (macro): {val_metrics['f1_macro']:.4f}"
        )
    
    def _compile_results(self, best_epoch: int) -> Dict[str, Any]:
        """Compile final results."""
        results = {
            "experiment_name": self.experiment_name,
            "experiment_dir": str(self.experiment_dir),
            "timestamp": datetime.now().isoformat(),
            "model": type(self.model).__name__,
            "loss": self.loss_fn.get_name(),
            "loss_params": self.loss_fn.get_params(),
            "best_epoch": best_epoch,
            "split_info": self.split_info,  # Detailed train/val/test split statistics
            "config": self.config,  # Save full training configuration
            "training_history": self.train_history,
            "metrics_history": self.metrics_history,
            "device": str(self.device),
            "num_model_parameters": sum(p.numel() for p in self.model.parameters()),
            "num_trainable_parameters": sum(
                p.numel() for p in self.model.parameters() if p.requires_grad
            ),
        }
        
        # Use metrics from best epoch (1-indexed, so subtract 1 for list index)
        if self.metrics_history["val_metrics"] and best_epoch > 0:
            best_idx = best_epoch - 1
            if best_idx < len(self.metrics_history["val_metrics"]):
                results["best_val_metrics"] = self.metrics_history["val_metrics"][best_idx]
            else:
                # Fallback if index out of range
                logger.warning(f"Best epoch {best_epoch} out of range, using last validation metrics")
                results["best_val_metrics"] = self.metrics_history["val_metrics"][-1]
        
        # Add test metrics
        if self.metrics_history["test_metrics"]:
            results["test_metrics"] = self.metrics_history["test_metrics"][0]
        
        return results
    
    def _save_results(self, results: Dict[str, Any]) -> None:
        """Save experiment results to JSON."""
        results_file = self.experiment_dir / "results.json"
        
        # Make results JSON serializable
        serializable_results = {
            k: v for k, v in results.items()
            if not isinstance(v, (dict, list)) or k not in ["training_history", "metrics_history"]
        }
        serializable_results["training_history"] = results["training_history"]
        serializable_results["metrics_history"] = {
            k: (v if not isinstance(v, list) or not v or not isinstance(v[0], np.ndarray)
                else [v_item.tolist() if isinstance(v_item, np.ndarray) else v_item for v_item in v])
            for k, v in results["metrics_history"].items()
        }
        
        with open(results_file, 'w') as f:
            json.dump(serializable_results, f, indent=2)
        
        logger.info(f"Results saved to {results_file}")
        
        # Save model checkpoint
        model_file = self.experiment_dir / "model_final.pt"
        torch.save(self.model.state_dict(), model_file)
        logger.info(f"Model saved to {model_file}")


if __name__ == "__main__":
    print("Trainer module ready for use")
