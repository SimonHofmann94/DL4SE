#!/usr/bin/env python3
"""
Verify that train/val/test splits have no overlap and are correctly saved.

This script checks:
1. No image appears in multiple splits
2. All images are accounted for
3. Split files are correctly formatted
4. Class distribution is preserved
"""

import os
import sys
from pathlib import Path
import numpy as np

# Add code directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.data import SeverstalFullImageDataset


def load_split_file(filepath):
    """Load image names from a split file."""
    with open(filepath, 'r') as f:
        return set(line.strip() for line in f if line.strip())


def verify_splits(splits_dir, img_dir, ann_dir):
    """
    Verify splits have no overlap and cover all images.
    
    Args:
        splits_dir: Directory containing split files
        img_dir: Directory with images
        ann_dir: Directory with annotations
    """
    print("="*60)
    print("SPLIT VERIFICATION")
    print("="*60)
    
    splits_path = Path(splits_dir)
    
    # Check if split files exist
    train_file = splits_path / "train.txt"
    val_file = splits_path / "val.txt"
    test_file = splits_path / "test.txt"
    
    if not all(f.exists() for f in [train_file, val_file, test_file]):
        print("❌ Split files not found!")
        print(f"   Looking in: {splits_path}")
        return False
    
    print(f"✅ Split files found in: {splits_path}\n")
    
    # Load splits
    train_set = load_split_file(train_file)
    val_set = load_split_file(val_file)
    test_set = load_split_file(test_file)
    
    print(f"Train set: {len(train_set)} images")
    print(f"Val set:   {len(val_set)} images")
    print(f"Test set:  {len(test_set)} images")
    print(f"Total:     {len(train_set) + len(val_set) + len(test_set)} images\n")
    
    # Check for overlaps
    print("Checking for overlaps...")
    
    train_val_overlap = train_set & val_set
    train_test_overlap = train_set & test_set
    val_test_overlap = val_set & test_set
    
    all_good = True
    
    if train_val_overlap:
        print(f"❌ OVERLAP between train and val: {len(train_val_overlap)} images")
        print(f"   Examples: {list(train_val_overlap)[:5]}")
        all_good = False
    else:
        print("✅ No overlap between train and val")
    
    if train_test_overlap:
        print(f"❌ OVERLAP between train and test: {len(train_test_overlap)} images")
        print(f"   Examples: {list(train_test_overlap)[:5]}")
        all_good = False
    else:
        print("✅ No overlap between train and test")
    
    if val_test_overlap:
        print(f"❌ OVERLAP between val and test: {len(val_test_overlap)} images")
        print(f"   Examples: {list(val_test_overlap)[:5]}")
        all_good = False
    else:
        print("✅ No overlap between val and test")
    
    print()
    
    # Check against actual image directory
    print("Checking coverage of actual images...")
    
    all_image_files = sorted([
        f for f in os.listdir(img_dir)
        if f.lower().endswith(('.jpg', '.jpeg', '.png'))
    ])
    
    print(f"Images in directory: {len(all_image_files)}")
    
    all_split_images = train_set | val_set | test_set
    
    missing_in_splits = set(all_image_files) - all_split_images
    extra_in_splits = all_split_images - set(all_image_files)
    
    if missing_in_splits:
        print(f"⚠️  Images in directory but NOT in splits: {len(missing_in_splits)}")
        print(f"   Examples: {list(missing_in_splits)[:5]}")
    else:
        print("✅ All images from directory are in splits")
    
    if extra_in_splits:
        print(f"❌ Images in splits but NOT in directory: {len(extra_in_splits)}")
        print(f"   Examples: {list(extra_in_splits)[:5]}")
        all_good = False
    else:
        print("✅ No extra images in splits")
    
    print()
    
    # Verify class distribution
    print("Verifying class distribution...")
    
    dataset_full = SeverstalFullImageDataset(
        img_dir=str(img_dir),
        ann_dir=str(ann_dir),
        image_names=all_image_files,
        transform=None,
        num_classes=5
    )
    
    # Get labels for each split
    name_to_label = {
        sample["image_name"]: sample["label"]
        for sample in dataset_full.samples
    }
    
    def get_distribution(image_set):
        labels = np.array([name_to_label[name] for name in image_set if name in name_to_label])
        if len(labels) == 0:
            return np.zeros(5)
        return labels.sum(axis=0) / len(labels)
    
    train_dist = get_distribution(train_set)
    val_dist = get_distribution(val_set)
    test_dist = get_distribution(test_set)
    overall_dist = get_distribution(all_split_images)
    
    class_names = ["no_defect", "defect_1", "defect_2", "defect_3", "defect_4"]
    
    print(f"\n{'Class':<12} {'Overall':<10} {'Train':<10} {'Val':<10} {'Test':<10}")
    print("-" * 52)
    
    for i, class_name in enumerate(class_names):
        print(f"{class_name:<12} {overall_dist[i]:<10.3f} {train_dist[i]:<10.3f} "
              f"{val_dist[i]:<10.3f} {test_dist[i]:<10.3f}")
    
    print()
    
    # Final verdict
    print("="*60)
    if all_good and not missing_in_splits:
        print("✅ ALL CHECKS PASSED - Splits are valid!")
    else:
        print("❌ SOME CHECKS FAILED - Please review above")
    print("="*60)
    
    return all_good and not missing_in_splits


if __name__ == "__main__":
    # Paths
    project_root = Path(__file__).parent.parent.parent
    splits_dir = project_root / "data" / "splits"
    img_dir = project_root / "data" / "images"
    ann_dir = project_root / "data" / "annotations"
    
    # Run verification
    success = verify_splits(splits_dir, img_dir, ann_dir)
    
    sys.exit(0 if success else 1)
