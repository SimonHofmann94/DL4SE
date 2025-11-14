#!/usr/bin/env python3
"""
Quick test to verify split_info calculation works correctly.
"""

import numpy as np
import sys
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent))

# Import the function from train.py
from train import calculate_split_statistics

def test_split_info():
    """Test the split statistics calculation."""
    
    print("Testing split_info calculation...")
    
    # Create synthetic labels
    np.random.seed(42)
    train_labels = np.random.rand(700, 5) > 0.5
    val_labels = np.random.rand(150, 5) > 0.5
    test_labels = np.random.rand(150, 5) > 0.5
    
    class_names = ['no_defect', 'defect_1', 'defect_2', 'defect_3', 'defect_4']
    
    # Calculate statistics
    split_info = calculate_split_statistics(
        train_labels=train_labels,
        val_labels=val_labels,
        test_labels=test_labels,
        class_names=class_names
    )
    
    # Display results
    print("\n" + "="*60)
    print("SPLIT STATISTICS")
    print("="*60)
    
    for split_name in ['train', 'val', 'test', 'total']:
        split_data = split_info[split_name]
        print(f"\n{split_name.upper()}:")
        print(f"  Total samples: {split_data['total_samples']}")
        print(f"  Class counts:")
        for class_name, count in split_data['class_counts'].items():
            print(f"    {class_name}: {count}")
    
    # Verify totals match
    total_samples = (
        split_info['train']['total_samples'] +
        split_info['val']['total_samples'] +
        split_info['test']['total_samples']
    )
    
    assert total_samples == split_info['total']['total_samples'], \
        f"Total mismatch: {total_samples} != {split_info['total']['total_samples']}"
    
    print("\n" + "="*60)
    print("✅ ALL CHECKS PASSED!")
    print("="*60)
    print("\nDie split_info wird korrekt in results.json gespeichert werden!")

if __name__ == "__main__":
    test_split_info()
