"""
Script to count images with actual defect annotations in the Severstal dataset.
Excludes annotations with empty objects arrays.
"""

import json
import os
from pathlib import Path


def count_annotations_with_defects(ann_folder):
    """
    Count annotation files that contain at least one defect object.
    
    Args:
        ann_folder: Path to the annotation folder
        
    Returns:
        tuple: (total_annotations, annotations_with_defects, annotations_without_defects)
    """
    ann_folder = Path(ann_folder)
    
    if not ann_folder.exists():
        print(f"Warning: Folder {ann_folder} does not exist!")
        return 0, 0, 0
    
    total_annotations = 0
    annotations_with_defects = 0
    annotations_without_defects = 0
    
    # Get all JSON files
    json_files = list(ann_folder.glob("*.json"))
    
    for json_file in json_files:
        total_annotations += 1
        
        try:
            with open(json_file, 'r') as f:
                data = json.load(f)
            
            # Check if objects array exists and has at least one object
            objects = data.get("objects", [])
            
            if objects and len(objects) > 0:
                annotations_with_defects += 1
            else:
                annotations_without_defects += 1
                
        except Exception as e:
            print(f"Error reading {json_file.name}: {e}")
    
    return total_annotations, annotations_with_defects, annotations_without_defects


def main():
    # Define base path
    base_path = Path(__file__).parent.parent.parent / "data" / "Severstal"
    
    print("=" * 70)
    print("SEVERSTAL DATASET - DEFECT ANNOTATION ANALYSIS")
    print("=" * 70)
    print()
    
    # Analyze train set
    train_ann_folder = base_path / "train" / "ann"
    print(f"Analyzing TRAIN set: {train_ann_folder}")
    print("-" * 70)
    
    train_total, train_with_defects, train_without_defects = count_annotations_with_defects(train_ann_folder)
    
    print(f"Total annotation files:              {train_total}")
    print(f"Annotations WITH defects:            {train_with_defects}")
    print(f"Annotations WITHOUT defects (empty): {train_without_defects}")
    
    if train_total > 0:
        percentage = (train_with_defects / train_total) * 100
        print(f"Percentage with defects:             {percentage:.2f}%")
    
    print()
    print("=" * 70)
    print()
    
    # Analyze test set
    test_ann_folder = base_path / "test" / "ann"
    print(f"Analyzing TEST set: {test_ann_folder}")
    print("-" * 70)
    
    test_total, test_with_defects, test_without_defects = count_annotations_with_defects(test_ann_folder)
    
    print(f"Total annotation files:              {test_total}")
    print(f"Annotations WITH defects:            {test_with_defects}")
    print(f"Annotations WITHOUT defects (empty): {test_without_defects}")
    
    if test_total > 0:
        percentage = (test_with_defects / test_total) * 100
        print(f"Percentage with defects:             {percentage:.2f}%")
    
    print()
    print("=" * 70)
    print()
    
    # Summary
    print("SUMMARY")
    print("-" * 70)
    combined_total = train_total + test_total
    combined_with_defects = train_with_defects + test_with_defects
    combined_without_defects = train_without_defects + test_without_defects
    
    print(f"Total images across train+test:      {combined_total}")
    print(f"Total WITH defects:                  {combined_with_defects}")
    print(f"Total WITHOUT defects:               {combined_without_defects}")
    
    if combined_total > 0:
        percentage = (combined_with_defects / combined_total) * 100
        print(f"Overall percentage with defects:     {percentage:.2f}%")
    
    print("=" * 70)


if __name__ == "__main__":
    main()
