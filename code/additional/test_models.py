"""
Test script for model     # Test transforms for testing - use correct Severstal dimensions
    test_transforms = transforms.Compose([
        transforms.Resize((256, 1600)),  # Correct Severstal dimensions: height=256, width=1600
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Create small dataset for testing
    dataset = SeverstalFullImageDataset(
        img_dir=img_dir,
        ann_dir=ann_dir,
        transform=test_transforms,
        target_size=(256, 1600),  # Correct dimensions
        debug_limit=5  # Just 5 images for testing
    )sic functionality.
"""

import sys
import os
import torch
import torch.nn as nn

# Add current directory and parent code directory to path
sys.path.append(os.path.dirname(__file__))
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from models import ClassifierModel, FocalLoss, create_model, count_parameters
from severstal_dataset import SeverstalFullImageDataset
import torchvision.transforms as transforms


def test_model_with_real_data():
    """Test model with actual dataset images."""
    
    print("Testing model with real Severstal data...")
    
    # Paths - go up two levels from additional folder to project root
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    img_dir = os.path.join(project_root, "data", "images")
    ann_dir = os.path.join(project_root, "data", "annotations")
    
    # Transforms for testing - keep original Severstal dimensions
    test_transforms = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Create small dataset for testing
    dataset = SeverstalFullImageDataset(
        img_dir=img_dir,
        ann_dir=ann_dir,
        transform=test_transforms,
        debug_limit=5  # Just 5 images for testing
    )
    
    if len(dataset) == 0:
        print("No images found for testing!")
        return
    
    # Create dataloader
    dataloader = torch.utils.data.DataLoader(
        dataset, 
        batch_size=2, 
        shuffle=False, 
        num_workers=0
    )
    
    # Test different models
    model_configs = [
        {"backbone": "densenet121", "name": "DenseNet121"},
        {"backbone": "efficientnet_b0", "name": "EfficientNet-B0"},
        {"backbone": "resnet50", "name": "ResNet50"}
    ]
    
    for config in model_configs:
        print(f"\n--- Testing {config['name']} with real data ---")
        
        # Create model
        model = ClassifierModel(
            backbone=config['backbone'],
            num_classes=4,
            pretrained=True,  # Use pretrained weights for better performance
            dropout_rate=0.5
        )
        
        # Count parameters
        total_params, trainable_params = count_parameters(model)
        print(f"Parameters: {total_params:,} total, {trainable_params:,} trainable")
        
        # Test with real data
        model.eval()
        with torch.no_grad():
            for batch_idx, (images, labels, names) in enumerate(dataloader):
                print(f"  Batch {batch_idx + 1}:")
                print(f"    Input shape: {images.shape}")
                print(f"    Labels shape: {labels.shape}")
                print(f"    Sample labels: {labels[0].numpy().astype(int)}")
                
                # Forward pass
                outputs = model(images)
                print(f"    Output shape: {outputs.shape}")
                print(f"    Output range: [{outputs.min().item():.3f}, {outputs.max().item():.3f}]")
                
                # Apply sigmoid to get probabilities
                probs = torch.sigmoid(outputs)
                print(f"    Probabilities: {probs[0].numpy()}")
                
                break  # Only test first batch
        
        print(f"✅ {config['name']} test with real data passed!")


def test_loss_functions():
    """Test different loss functions."""
    
    print("\n--- Testing Loss Functions ---")
    
    # Create dummy data
    batch_size, num_classes = 4, 4
    predictions = torch.randn(batch_size, num_classes)
    targets = torch.randint(0, 2, (batch_size, num_classes)).float()
    
    print(f"Predictions shape: {predictions.shape}")
    print(f"Targets shape: {targets.shape}")
    print(f"Sample targets: {targets[0].numpy().astype(int)}")
    
    # Test BCEWithLogitsLoss
    pos_weights = torch.tensor([6.25, 20.74, 0.29, 9.42])
    bce_loss = nn.BCEWithLogitsLoss(pos_weight=pos_weights)
    bce_value = bce_loss(predictions, targets)
    print(f"BCEWithLogitsLoss (with pos_weight): {bce_value.item():.4f}")
    
    # Test Focal Loss
    focal_loss = FocalLoss(alpha=1.0, gamma=2.0)
    focal_value = focal_loss(predictions, targets)
    print(f"Focal Loss (α=1.0, γ=2.0): {focal_value.item():.4f}")
    
    # Test regular BCE for comparison
    regular_bce = nn.BCEWithLogitsLoss()
    regular_value = regular_bce(predictions, targets)
    print(f"Regular BCEWithLogitsLoss: {regular_value.item():.4f}")
    
    print("✅ Loss function tests passed!")


def main():
    """Main test function."""
    
    print("🔬 SEVERSTAL MODEL TESTING")
    print("=" * 50)
    
    # Test models with real data
    test_model_with_real_data()
    
    # Test loss functions
    test_loss_functions()
    
    print("\n🎉 All model tests completed successfully!")


if __name__ == "__main__":
    main()