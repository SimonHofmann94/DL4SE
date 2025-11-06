"""
Test script for analyzing class distribution in the full dataset.
"""

import sys
import os

# Add current directory to path
sys.path.append(os.path.dirname(__file__))

from severstal_dataset import SeverstalFullImageDataset, get_class_distribution
import torchvision.transforms as transforms

def analyze_full_dataset():
    """Analyze class distribution in the complete dataset."""
    
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    img_dir = os.path.join(project_root, "data", "images")
    ann_dir = os.path.join(project_root, "data", "annotations")
    
    print("Creating full dataset for analysis...")
    
    # Simple transforms for analysis
    simple_transforms = transforms.Compose([
        transforms.Resize((256, 256)),  # Smaller size for faster loading
        transforms.ToTensor()
    ])
    
    # Create dataset with sample for analysis
    dataset = SeverstalFullImageDataset(
        img_dir=img_dir,
        ann_dir=ann_dir,
        transform=simple_transforms,
        debug_limit=6658  # Analyze first 500 images
    )
    
    if len(dataset) == 0:
        print("No images found in dataset!")
        return
    
    print(f"Analyzing {len(dataset)} images...")
    
    # Get class distribution
    distribution = get_class_distribution(dataset)
    
    print("\n" + "="*50)
    print("CLASS DISTRIBUTION ANALYSIS")
    print("="*50)
    
    total_images = len(dataset)
    
    for class_name in ["defect_1", "defect_2", "defect_3", "defect_4"]:
        count = distribution[class_name]
        percentage = (count / total_images) * 100
        print(f"{class_name}: {count:4d} images ({percentage:5.2f}%)")
    
    print(f"\nImages with defects:    {distribution['images_with_defects']:4d} ({distribution['images_with_defects']/total_images*100:5.2f}%)")
    print(f"Images without defects: {distribution['images_without_defects']:4d} ({distribution['images_without_defects']/total_images*100:5.2f}%)")
    
    # Calculate class weights for BCEWithLogitsLoss
    print(f"\nRecommended pos_weights for BCEWithLogitsLoss:")
    pos_weights = []
    for class_name in ["defect_1", "defect_2", "defect_3", "defect_4"]:
        count = distribution[class_name]
        if count > 0:
            # Weight = (total_images - positive_samples) / positive_samples
            weight = (total_images - count) / count
            pos_weights.append(weight)
            print(f"{class_name}: {weight:.2f}")
        else:
            pos_weights.append(1.0)
            print(f"{class_name}: 1.00 (no samples)")
    
    print(f"\npos_weights tensor: {pos_weights}")

if __name__ == "__main__":
    analyze_full_dataset()