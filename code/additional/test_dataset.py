"""
Simple test script for Severstal Dataset without Hydra dependency.
"""

import os
import sys
import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

# Add code directory to path
sys.path.append(os.path.dirname(__file__))
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from severstal_dataset import SeverstalFullImageDataset

def test_dataset():
    """Simple test of the Severstal dataset."""
    
    # Paths based on our project structure - corrected paths
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    img_dir = os.path.join(project_root, "data", "images")
    ann_dir = os.path.join(project_root, "data", "annotations")
    
    print(f"Testing Severstal Full Image Dataset")
    print(f"Project root: {project_root}")
    print(f"Image directory: {img_dir}")
    print(f"Annotation directory: {ann_dir}")
    
    # Check if directories exist
    if not os.path.exists(img_dir):
        print(f"ERROR: Image directory not found: {img_dir}")
        return
    if not os.path.exists(ann_dir):
        print(f"ERROR: Annotation directory not found: {ann_dir}")
        return
    
    # Simple transforms for testing - keep original Severstal dimensions
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Create dataset with limited images for testing
    print("\nCreating dataset...")
    dataset = SeverstalFullImageDataset(
        img_dir=img_dir,
        ann_dir=ann_dir,
        transform=test_transform,
        debug_limit=10  # Only process first 10 images for testing
    )
    
    print(f"Dataset created with {len(dataset)} images")
    
    if len(dataset) == 0:
        print("No images found in dataset!")
        return
    
    # Create dataloader
    dataloader = DataLoader(
        dataset, 
        batch_size=2,  # Smaller batch for full images
        shuffle=True, 
        num_workers=0  # Use 0 for Windows compatibility
    )
    
    print("\nTesting dataloader...")
    
    # Test first batch
    for i, (images, labels, image_names) in enumerate(dataloader):
        print(f"\nBatch {i+1}:")
        print(f"  Images shape: {images.shape}")
        print(f"  Labels shape: {labels.shape}")
        print(f"  Sample labels: {labels[0].numpy().astype(int)}")
        print(f"  Image names: {image_names[:2]}")  # Show first 2 names
        
        # Check for defects
        num_positive_samples = (labels.sum(dim=1) > 0).sum().item()
        print(f"  Images with defects: {num_positive_samples}/{len(labels)}")
        
        # Verify correct Severstal dimensions
        expected_shape = (images.shape[0], 3, 256, 1600)  # batch, channels, height, width
        if images.shape == expected_shape:
            print(f"  ✅ Correct Severstal dimensions: {images.shape}")
        else:
            print(f"  ❌ Wrong dimensions: got {images.shape}, expected {expected_shape}")
        
        if i >= 1:  # Only test first 2 batches
            break
    
    print("\nDataset test completed successfully!")

if __name__ == "__main__":
    test_dataset()