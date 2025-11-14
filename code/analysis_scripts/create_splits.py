#!/usr/bin/env python3
"""
Create train/val/test splits and save them to disk.

This script:
1. Loads all images and annotations
2. Creates stratified splits
3. Saves splits to data/splits/
4. Verifies no overlap

Run this ONCE to create the splits, then use them for all training runs.
"""

import os
import sys
import logging
from pathlib import Path
import numpy as np

# Add code directory to path
sys.path.insert(0, str(Path(__file__).parent))

from core.data import SeverstalFullImageDataset, StratifiedSplitter

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def create_splits(
    img_dir: str,
    ann_dir: str,
    splits_dir: str,
    split_strategy: str = "stratified_70_15_15",
    random_state: int = 42
):
    """
    Create and save train/val/test splits.
    
    Args:
        img_dir: Directory containing images
        ann_dir: Directory containing annotations
        splits_dir: Directory to save splits
        split_strategy: Split strategy name
        random_state: Random seed
    """
    logger.info("="*60)
    logger.info("CREATING TRAIN/VAL/TEST SPLITS")
    logger.info("="*60)
    
    # Get all image files
    all_image_files = sorted([
        f for f in os.listdir(img_dir)
        if f.lower().endswith(('.jpg', '.jpeg', '.png'))
    ])
    
    logger.info(f"Found {len(all_image_files)} images in {img_dir}")
    
    # Load dataset to get labels
    logger.info("Loading annotations...")
    dataset_full = SeverstalFullImageDataset(
        img_dir=img_dir,
        ann_dir=ann_dir,
        image_names=all_image_files,
        transform=None,
        num_classes=5
    )
    
    # Extract labels and image names
    all_labels = np.array([
        sample["label"] for sample in dataset_full.samples
    ])
    all_image_names = np.array([
        sample["image_name"] for sample in dataset_full.samples
    ])
    
    logger.info(f"Label matrix shape: {all_labels.shape}")
    
    # Determine split ratios
    if split_strategy == "stratified_70_15_15":
        split_ratios = (0.7, 0.15, 0.15)
    elif split_strategy == "stratified_80_10_10":
        split_ratios = (0.8, 0.1, 0.1)
    else:
        raise ValueError(f"Unknown split strategy: {split_strategy}")
    
    logger.info(f"Split strategy: {split_strategy}")
    logger.info(f"Split ratios: train={split_ratios[0]}, val={split_ratios[1]}, test={split_ratios[2]}")
    logger.info(f"Random seed: {random_state}")
    
    # Create splits
    logger.info("\nCreating stratified splits...")
    splitter = StratifiedSplitter(random_state=random_state)
    train_idx, val_idx, test_idx = splitter.split(
        all_labels,
        split_ratios=split_ratios
    )
    
    # Verify no overlap
    logger.info("\nVerifying no overlap...")
    train_set = set(train_idx)
    val_set = set(val_idx)
    test_set = set(test_idx)
    
    assert len(train_set & val_set) == 0, "Overlap between train and val!"
    assert len(train_set & test_set) == 0, "Overlap between train and test!"
    assert len(val_set & test_set) == 0, "Overlap between val and test!"
    logger.info("✅ No overlap detected")
    
    # Save splits
    logger.info(f"\nSaving splits to {splits_dir}...")
    splitter.save_splits(
        train_idx, val_idx, test_idx,
        all_image_names,
        save_dir=splits_dir,
        split_name=split_strategy
    )
    
    logger.info("\n" + "="*60)
    logger.info("✅ SPLITS CREATED SUCCESSFULLY!")
    logger.info("="*60)
    logger.info(f"Location: {splits_dir}")
    logger.info(f"Files created:")
    logger.info(f"  - train.txt ({len(train_idx)} images)")
    logger.info(f"  - val.txt ({len(val_idx)} images)")
    logger.info(f"  - test.txt ({len(test_idx)} images)")
    logger.info(f"  - split_metadata.json")
    logger.info("\nYou can now run training with these fixed splits!")


if __name__ == "__main__":
    # Paths
    project_root = Path(__file__).parent.parent
    img_dir = project_root / "data" / "images"
    ann_dir = project_root / "data" / "annotations"
    splits_dir = project_root / "data" / "splits"
    
    # Configuration
    SPLIT_STRATEGY = "stratified_70_15_15"  # or "stratified_80_10_10"
    RANDOM_SEED = 42
    
    # Create splits
    create_splits(
        img_dir=str(img_dir),
        ann_dir=str(ann_dir),
        splits_dir=str(splits_dir),
        split_strategy=SPLIT_STRATEGY,
        random_state=RANDOM_SEED
    )
