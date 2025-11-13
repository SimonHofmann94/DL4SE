#!/usr/bin/env python3
"""
Re-evaluate existing experiments with both standard and optimized thresholds.
This script provides transparent reporting of model performance.

Usage:
    python code/analysis_scripts/dual_evaluation.py --experiment experiment_20251113_083139
"""

import os
import sys
import json
import argparse
from pathlib import Path
from datetime import datetime
import torch
import numpy as np
from typing import Dict, Any

# Add code directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.models.registry import get_registry as get_model_registry
from core.data import SeverstalFullImageDataset, StratifiedSplitter, create_dataloaders
from core.training.metrics import compute_metrics, log_per_class_metrics
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def load_experiment_config(experiment_dir: Path) -> Dict[str, Any]:
    """Load experiment configuration and results."""
    results_file = experiment_dir / "results.json"
    
    if not results_file.exists():
        raise FileNotFoundError(f"Results file not found: {results_file}")
    
    with open(results_file, 'r') as f:
        return json.load(f)


def load_model_checkpoint(experiment_dir: Path, config: Dict[str, Any], device: torch.device):
    """Load the best model checkpoint."""
    
    # Find checkpoint file
    checkpoint_files = list(experiment_dir.glob("*.pt"))
    if not checkpoint_files:
        raise FileNotFoundError(f"No checkpoint files found in {experiment_dir}")
    
    # Use the first .pt file found (should be the best model)
    checkpoint_path = checkpoint_files[0]
    logger.info(f"Loading model from: {checkpoint_path}")
    
    # Create model
    model_registry = get_model_registry()
    model_config = config["config"]["model"]
    
    model = model_registry.get(
        model_config["name"],
        num_classes=model_config["num_classes"],
        pretrained=model_config["pretrained"],
        cbam_stages=model_config.get("cbam_stages", [])
    )
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Handle different checkpoint formats
    if isinstance(checkpoint, dict):
        # If checkpoint is a dict, try common keys
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        elif 'state_dict' in checkpoint:
            model.load_state_dict(checkpoint['state_dict'])
        elif 'model' in checkpoint:
            model.load_state_dict(checkpoint['model'])
        else:
            # If none of the above, the dict itself might be the state_dict
            model.load_state_dict(checkpoint)
    else:
        # If checkpoint is directly the state_dict (not wrapped in a dict)
        model.load_state_dict(checkpoint)
    
    model.to(device)
    model.eval()
    
    logger.info(f"Model loaded successfully. Best epoch: {config.get('best_epoch', 'Unknown')}")
    
    return model


def recreate_test_dataset(config: Dict[str, Any]):
    """Recreate the test dataset using the same split strategy."""
    
    # Extract data configuration
    data_config = config["config"]["data"]
    experiment_config = config["config"]["experiment"]
    
    # Setup paths
    project_root = Path(__file__).parent.parent.parent
    img_dir = project_root / data_config["img_dir"]
    ann_dir = project_root / data_config["ann_dir"]
    
    logger.info(f"Image directory: {img_dir}")
    logger.info(f"Annotation directory: {ann_dir}")
    
    # Get all image files (same as in training)
    all_image_files = sorted([
        f for f in os.listdir(img_dir)
        if f.lower().endswith(('.jpg', '.jpeg', '.png'))
    ])
    
    # Create full dataset to extract labels
    dataset_full = SeverstalFullImageDataset(
        img_dir=str(img_dir),
        ann_dir=str(ann_dir),
        image_names=all_image_files,
        transform=None,
        num_classes=data_config["num_classes"]
    )
    
    # Extract labels and names
    all_labels = np.array([sample["label"] for sample in dataset_full.samples])
    all_image_names = np.array([sample["image_name"] for sample in dataset_full.samples])
    
    # Recreate the exact same split
    split_strategy = data_config["split_strategy"]
    if split_strategy == "stratified_70_15_15":
        split_ratios = (0.7, 0.15, 0.15)
    elif split_strategy == "stratified_80_10_10":
        split_ratios = (0.8, 0.1, 0.1)
    else:
        raise ValueError(f"Unknown split strategy: {split_strategy}")
    
    splitter = StratifiedSplitter(random_state=experiment_config["seed"])
    train_idx, val_idx, test_idx = splitter.split(all_labels, split_ratios=split_ratios)
    
    # Create test dataset
    test_image_names = all_image_names[test_idx].tolist()
    
    test_dataset = SeverstalFullImageDataset(
        img_dir=str(img_dir),
        ann_dir=str(ann_dir),
        image_names=test_image_names,
        transform=None,  # No augmentation for evaluation
        num_classes=data_config["num_classes"]
    )
    
    logger.info(f"Recreated test dataset: {len(test_dataset)} samples")
    
    return test_dataset


def get_model_predictions(model: torch.nn.Module, test_dataset, device: torch.device, batch_size: int = 32):
    """Get model predictions on test dataset."""
    
    # Create dataloader
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,  # Use 0 workers on Windows to avoid multiprocessing issues
        pin_memory=False  # Disable on CPU
    )
    
    all_logits = []
    all_targets = []
    
    logger.info("Getting model predictions...")
    logger.info(f"Test dataset size: {len(test_dataset)}")
    logger.info(f"Number of batches: {len(test_loader)}")
    
    with torch.no_grad():
        for batch_idx, (images, targets) in enumerate(test_loader):
            if batch_idx % 10 == 0:
                logger.info(f"Processing batch {batch_idx + 1}/{len(test_loader)}")
            
            # Debug first batch
            if batch_idx == 0:
                logger.info(f"First batch - Images shape: {images.shape}, Targets shape: {targets.shape}")
                logger.info(f"First batch - Targets sample: {targets[0]}")
                
            images = images.to(device)
            targets = targets.to(device)
            
            # Forward pass
            outputs = model(images)
            
            # Debug first batch outputs
            if batch_idx == 0:
                logger.info(f"First batch - Outputs shape: {outputs.shape}")
                logger.info(f"First batch - Outputs sample (raw logits): {outputs[0]}")
                probs_sample = torch.sigmoid(outputs[0])
                logger.info(f"First batch - Probabilities sample: {probs_sample}")
            
            # Store results
            all_logits.append(outputs.detach().cpu().numpy())
            all_targets.append(targets.detach().cpu().numpy())
    
    # Concatenate all
    all_logits = np.vstack(all_logits)
    all_targets = np.vstack(all_targets)
    
    # Convert to probabilities
    probabilities = 1 / (1 + np.exp(-all_logits))
    
    logger.info(f"Predictions completed. Logits shape: {all_logits.shape}")
    logger.info(f"Probabilities range: [{probabilities.min():.4f}, {probabilities.max():.4f}]")
    logger.info(f"Targets shape: {all_targets.shape}")
    logger.info(f"Targets sum per class: {all_targets.sum(axis=0)}")
    
    return probabilities, all_targets


def dual_evaluation(
    probabilities: np.ndarray, 
    targets: np.ndarray, 
    class_names: list,
    optimal_thresholds: list = None
) -> Dict[str, Dict[str, float]]:
    """Perform dual evaluation with both standard and optimized thresholds."""
    
    logger.info("\n" + "="*80)
    logger.info("DUAL EVALUATION RESULTS")
    logger.info("="*80)
    
    # 1. Standard evaluation (fair comparison)
    logger.info("\n📊 STANDARD THRESHOLD EVALUATION (Fair Comparison)")
    logger.info("   Using uniform threshold: 0.5 for all classes")
    
    metrics_standard = compute_metrics(
        probabilities,
        targets,
        threshold=0.5,
        class_names=class_names,
        per_class_thresholds=None
    )
    
    logger.info(f"   F1 Macro: {metrics_standard['f1_macro']:.4f}")
    logger.info(f"   F1 Micro: {metrics_standard['f1_micro']:.4f}")
    logger.info(f"   Accuracy: {metrics_standard['accuracy']:.4f}")
    logger.info(f"   Precision Macro: {metrics_standard['precision_macro']:.4f}")
    logger.info(f"   Recall Macro: {metrics_standard['recall_macro']:.4f}")
    
    # 2. Optimized evaluation (production performance)
    logger.info("\n🎯 OPTIMIZED THRESHOLD EVALUATION (Production Performance)")
    
    if optimal_thresholds:
        logger.info("   Using learned per-class thresholds:")
        for class_name in class_names:
            thresh = optimal_thresholds.get(class_name, 0.5)
            logger.info(f"     {class_name}: {thresh:.3f}")
    else:
        logger.info("   No learned thresholds available, using 0.5")
    
    metrics_optimized = compute_metrics(
        probabilities,
        targets,
        threshold=0.5,  # Fallback
        class_names=class_names,
        per_class_thresholds=optimal_thresholds
    )
    
    logger.info(f"   F1 Macro: {metrics_optimized['f1_macro']:.4f}")
    logger.info(f"   F1 Micro: {metrics_optimized['f1_micro']:.4f}")
    logger.info(f"   Accuracy: {metrics_optimized['accuracy']:.4f}")
    logger.info(f"   Precision Macro: {metrics_optimized['precision_macro']:.4f}")
    logger.info(f"   Recall Macro: {metrics_optimized['recall_macro']:.4f}")
    
    # 3. Comparison and benefit analysis
    logger.info("\n💡 THRESHOLD OPTIMIZATION BENEFIT")
    f1_improvement = metrics_optimized['f1_macro'] - metrics_standard['f1_macro']
    f1_improvement_pct = (f1_improvement / metrics_standard['f1_macro'] * 100) if metrics_standard['f1_macro'] > 0 else 0
    
    acc_improvement = metrics_optimized['accuracy'] - metrics_standard['accuracy']
    acc_improvement_pct = (acc_improvement / metrics_standard['accuracy'] * 100) if metrics_standard['accuracy'] > 0 else 0
    
    logger.info(f"   F1 Macro Improvement: {f1_improvement:+.4f} ({f1_improvement_pct:+.2f}%)")
    logger.info(f"   Accuracy Improvement: {acc_improvement:+.4f} ({acc_improvement_pct:+.2f}%)")
    
    # Status assessment
    if f1_improvement_pct > 5.0:
        status = "🔴 Significant benefit - consider if this is realistic"
    elif f1_improvement_pct > 2.0:
        status = "🟡 Notable benefit - good optimization"
    elif f1_improvement_pct > 0.5:
        status = "🟢 Moderate benefit - reasonable optimization"
    else:
        status = "🟢 Minimal benefit - thresholds close to optimal"
    
    logger.info(f"   Assessment: {status}")
    
    return {
        'standard': metrics_standard,
        'optimized': metrics_optimized,
        'benefit': {
            'f1_macro_improvement': f1_improvement,
            'f1_macro_improvement_percent': f1_improvement_pct,
            'accuracy_improvement': acc_improvement,
            'accuracy_improvement_percent': acc_improvement_pct
        }
    }


def save_dual_evaluation_results(experiment_dir: Path, results: Dict[str, Any]):
    """Save dual evaluation results to a new file."""
    
    output_file = experiment_dir / "dual_evaluation_results.json"
    
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    logger.info(f"\n💾 Dual evaluation results saved to: {output_file}")


def main():
    """Main evaluation function."""
    
    parser = argparse.ArgumentParser(description='Dual evaluation of experiments')
    parser.add_argument('--experiment', type=str, required=True, 
                       help='Experiment directory name (e.g., experiment_20251113_083139)')
    parser.add_argument('--batch_size', type=int, default=32,
                       help='Batch size for evaluation (default: 32)')
    
    args = parser.parse_args()
    
    # Setup paths
    project_root = Path(__file__).parent.parent.parent
    experiment_dir = project_root / "code" / "experiments" / "results" / args.experiment
    
    if not experiment_dir.exists():
        logger.error(f"Experiment directory not found: {experiment_dir}")
        return
    
    logger.info(f"🔍 Dual Evaluation: {args.experiment}")
    logger.info(f"📂 Experiment directory: {experiment_dir}")
    
    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"💻 Using device: {device}")
    
    try:
        # Load experiment configuration
        logger.info("\n📋 Loading experiment configuration...")
        config = load_experiment_config(experiment_dir)
        
        # Load model
        logger.info("\n🧠 Loading model...")
        model = load_model_checkpoint(experiment_dir, config, device)
        
        # Recreate test dataset
        logger.info("\n📊 Recreating test dataset...")
        test_dataset = recreate_test_dataset(config)
        
        # Get predictions
        logger.info("\n🔮 Getting model predictions...")
        probabilities, targets = get_model_predictions(model, test_dataset, device, args.batch_size)
        
        # Perform dual evaluation
        logger.info("\n⚖️ Performing dual evaluation...")
        class_names = config["config"]["data"]["class_names"]
        optimal_thresholds = config.get("optimal_thresholds")
        
        # Convert optimal_thresholds to dict if it's a list
        if optimal_thresholds:
            if isinstance(optimal_thresholds, list):
                # Convert list to dict using class_names
                optimal_thresholds = {class_names[i]: optimal_thresholds[i] for i in range(len(class_names))}
            # If already a dict, use as is
        
        eval_results = dual_evaluation(probabilities, targets, class_names, optimal_thresholds)
        
        # Save results
        logger.info("\n💾 Saving results...")
        full_results = {
            "experiment_name": config.get("experiment_name", args.experiment),
            "evaluation_timestamp": datetime.now().isoformat(),
            "original_results": {
                "best_val_metrics": config.get("best_val_metrics"),
                "test_metrics": config.get("test_metrics")  # Original (likely optimized)
            },
            "dual_evaluation": eval_results,
            "optimal_thresholds_used": optimal_thresholds,
            "class_names": class_names
        }
        
        save_dual_evaluation_results(experiment_dir, full_results)
        
        logger.info("\n✅ Dual evaluation completed successfully!")
        
    except Exception as e:
        logger.error(f"❌ Error during evaluation: {e}")
        raise


if __name__ == "__main__":
    main()