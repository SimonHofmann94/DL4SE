"""
Main training entry point.

This is the clean, minimal script that users interact with.
All complexity is abstracted into modular components.

Usage:
    python code/train.py
    python code/train.py model.name=convnext_tiny_cbam loss.type=focal_loss
    python code/train.py data.batch_size=8 training.num_epochs=50
"""

import os
import sys
import logging
from pathlib import Path

import torch
import numpy as np
from hydra import compose, initialize_config_dir
from omegaconf import DictConfig, OmegaConf

# Add code directory to path
sys.path.insert(0, str(Path(__file__).parent))

from core.models.registry import get_registry as get_model_registry
from core.losses.registry import get_registry as get_loss_registry
from core.augmentation import get_augmentation_pipeline
from core.data import SeverstalFullImageDataset, StratifiedSplitter, create_dataloaders
from core.training import Trainer
import torch.optim as optim

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def set_seed(seed: int) -> None:
    """Set random seeds for reproducibility."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    logger.info(f"Random seed set to {seed}")


def setup_device() -> torch.device:
    """Setup compute device."""
    if torch.cuda.is_available():
        device = torch.device("cuda")
        logger.info(f"Using GPU: {torch.cuda.get_device_name(0)}")
        logger.info(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f}GB")
    else:
        device = torch.device("cpu")
        logger.info("Using CPU")
    
    return device


def load_data(cfg: DictConfig, device: torch.device) -> tuple:
    """
    Load and prepare data.
    
    Args:
        cfg: Configuration
        device: Compute device
    
    Returns:
        Tuple of (train_loader, val_loader, test_loader)
    """
    
    logger.info("\n" + "="*60)
    logger.info("Loading Data")
    logger.info("="*60)
    
    # Get image and annotation directories
    project_root = Path(__file__).parent.parent
    img_dir = project_root / cfg.data.img_dir
    ann_dir = project_root / cfg.data.ann_dir
    
    logger.info(f"Image directory: {img_dir}")
    logger.info(f"Annotation directory: {ann_dir}")
    
    # Get list of image files
    all_image_files = sorted([
        f for f in os.listdir(img_dir)
        if f.lower().endswith(('.jpg', '.jpeg', '.png'))
    ])
    
    logger.info(f"Found {len(all_image_files)} images")
    
    # Get all labels
    dataset_full = SeverstalFullImageDataset(
        img_dir=str(img_dir),
        ann_dir=str(ann_dir),
        image_names=all_image_files,
        transform=None,
        num_classes=cfg.data.num_classes
    )
    
    # Extract labels
    all_labels = np.array([
        sample["label"] for sample in dataset_full.samples
    ])
    all_image_names = np.array([
        sample["image_name"] for sample in dataset_full.samples
    ])
    
    logger.info(f"Label matrix shape: {all_labels.shape}")
    
    # Stratified split
    split_strategy = cfg.data.split_strategy
    if split_strategy == "stratified_70_15_15":
        split_ratios = (0.7, 0.15, 0.15)
    elif split_strategy == "stratified_80_10_10":
        split_ratios = (0.8, 0.1, 0.1)
    else:
        raise ValueError(f"Unknown split strategy: {split_strategy}")
    
    logger.info(f"Using split strategy: {split_strategy} {split_ratios}")
    
    splitter = StratifiedSplitter(random_state=cfg.experiment.seed)
    train_idx, val_idx, test_idx = splitter.split(
        all_labels,
        split_ratios=split_ratios
    )
    
    # Create subsets
    train_image_names = all_image_names[train_idx].tolist()
    val_image_names = all_image_names[val_idx].tolist()
    test_image_names = all_image_names[test_idx].tolist()
    
    # Setup augmentations
    transform = get_augmentation_pipeline(
        image_size=tuple(cfg.data.image_size),
        augmentation_config=OmegaConf.to_container(cfg.augmentation)
    )
    
    # Create datasets
    train_dataset = SeverstalFullImageDataset(
        img_dir=str(img_dir),
        ann_dir=str(ann_dir),
        image_names=train_image_names,
        transform=transform,
        num_classes=cfg.data.num_classes
    )
    
    val_dataset = SeverstalFullImageDataset(
        img_dir=str(img_dir),
        ann_dir=str(ann_dir),
        image_names=val_image_names,
        transform=transform,
        num_classes=cfg.data.num_classes
    )
    
    test_dataset = SeverstalFullImageDataset(
        img_dir=str(img_dir),
        ann_dir=str(ann_dir),
        image_names=test_image_names,
        transform=transform,
        num_classes=cfg.data.num_classes
    )
    
    logger.info(f"Train set: {len(train_dataset)} samples")
    logger.info(f"Val set: {len(val_dataset)} samples")
    logger.info(f"Test set: {len(test_dataset)} samples")
    
    # Create dataloaders
    train_loader, val_loader, test_loader = create_dataloaders(
        train_dataset,
        val_dataset,
        test_dataset,
        batch_size=cfg.data.batch_size,
        num_workers=cfg.data.num_workers,
        pin_memory=cfg.data.pin_memory
    )
    
    return train_loader, val_loader, test_loader


def build_model(cfg: DictConfig, device: torch.device) -> torch.nn.Module:
    """Build model from configuration."""
    
    logger.info("\n" + "="*60)
    logger.info("Building Model")
    logger.info("="*60)
    
    model_registry = get_model_registry()
    
    logger.info(f"Model: {cfg.model.name}")
    logger.info(f"Available models: {list(model_registry.list_models().keys())}")
    
    model = model_registry.get(
        cfg.model.name,
        num_classes=cfg.model.num_classes,
        pretrained=cfg.model.pretrained,
        cbam_stages=cfg.model.cbam_stages
    )
    
    # Freeze backbone if requested
    if cfg.training.freeze_backbone:
        logger.info("Freezing backbone weights")
        model.freeze_backbone(freeze=True)
    
    if cfg.training.freeze_early_stages is not None:
        logger.info(f"Freezing first {cfg.training.freeze_early_stages} stages")
        model.freeze_early_stages(num_stages=cfg.training.freeze_early_stages)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Total parameters: {total_params:,}")
    logger.info(f"Trainable parameters: {trainable_params:,}")
    
    return model


def build_loss(cfg: DictConfig) -> torch.nn.Module:
    """Build loss function from configuration."""
    
    logger.info("\n" + "="*60)
    logger.info("Building Loss Function")
    logger.info("="*60)
    
    loss_registry = get_loss_registry()
    
    logger.info(f"Loss: {cfg.loss.type}")
    logger.info(f"Available losses: {list(loss_registry.list_losses().keys())}")
    
    loss_fn = loss_registry.get(
        cfg.loss.type,
        num_classes=cfg.data.num_classes,
        alpha=cfg.loss.alpha,
        gamma=cfg.loss.gamma,
        reduction=cfg.loss.reduction
    )
    
    logger.info(f"Loss configuration: {loss_fn.log_info()}")
    
    return loss_fn


def build_optimizer(cfg: DictConfig, model: torch.nn.Module) -> torch.optim.Optimizer:
    """Build optimizer from configuration."""
    
    logger.info("\n" + "="*60)
    logger.info("Building Optimizer")
    logger.info("="*60)
    
    if cfg.optimizer.type.lower() == "adamw":
        optimizer = optim.AdamW(
            model.parameters(),
            lr=cfg.optimizer.lr,
            weight_decay=cfg.optimizer.weight_decay,
            betas=cfg.optimizer.betas
        )
    elif cfg.optimizer.type.lower() == "adam":
        optimizer = optim.Adam(
            model.parameters(),
            lr=cfg.optimizer.lr,
            weight_decay=cfg.optimizer.weight_decay
        )
    elif cfg.optimizer.type.lower() == "sgd":
        optimizer = optim.SGD(
            model.parameters(),
            lr=cfg.optimizer.lr,
            weight_decay=cfg.optimizer.weight_decay,
            momentum=0.9
        )
    else:
        raise ValueError(f"Unknown optimizer: {cfg.optimizer.type}")
    
    logger.info(f"Optimizer: {cfg.optimizer.type}")
    logger.info(f"Learning rate: {cfg.optimizer.lr}")
    logger.info(f"Weight decay: {cfg.optimizer.weight_decay}")
    
    return optimizer


def main(cfg: DictConfig) -> None:
    """Main training pipeline."""
    
    logger.info("\n" + "="*70)
    logger.info("SEVERSTAL DEFECT CLASSIFICATION - TRAINING PIPELINE")
    logger.info("="*70)
    
    # Print configuration
    logger.info("\nConfiguration:")
    logger.info(OmegaConf.to_yaml(cfg))
    
    # Set seed
    set_seed(cfg.experiment.seed)
    
    # Setup device
    device = setup_device()
    
    # Load data
    train_loader, val_loader, test_loader = load_data(cfg, device)
    
    # Build model
    model = build_model(cfg, device)
    
    # Build loss
    loss_fn = build_loss(cfg)
    
    # Build optimizer
    optimizer = build_optimizer(cfg, model)
    
    # Create trainer
    logger.info("\n" + "="*60)
    logger.info("Initializing Trainer")
    logger.info("="*60)
    
    project_root = Path(__file__).parent.parent
    experiment_dir = project_root / cfg.experiment.save_dir
    
    trainer = Trainer(
        model=model,
        loss_fn=loss_fn,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        optimizer=optimizer,
        scheduler=None,  # Will be created in train()
        device=device,
        experiment_dir=str(experiment_dir),
        class_names=cfg.data.class_names
    )
    
    # Train
    results = trainer.train(
        num_epochs=cfg.training.num_epochs,
        early_stopping_patience=cfg.training.early_stopping_patience,
        warmup_epochs=cfg.training.warmup_epochs,
        log_interval=cfg.training.log_interval,
        threshold=cfg.training.threshold
    )
    
    logger.info("\n" + "="*70)
    logger.info("TRAINING COMPLETED SUCCESSFULLY")
    logger.info("="*70)
    logger.info(f"Experiment directory: {trainer.experiment_dir}")


if __name__ == "__main__":
    # Load config using Hydra
    config_dir = Path(__file__).parent.parent / "config"
    
    with initialize_config_dir(version_base=None, config_dir=str(config_dir)):
        cfg = compose(config_name="train_config")
        main(cfg)
