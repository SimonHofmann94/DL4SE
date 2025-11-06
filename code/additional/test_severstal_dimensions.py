"""
Test EfficientNet with correct Severstal dimensions (256x1600).
"""

import torch
import torch.nn as nn
import timm

def test_with_severstal_dimensions():
    """Test EfficientNet with actual Severstal image dimensions."""
    
    print("🔍 TESTING WITH SEVERSTAL DIMENSIONS (256x1600)")
    print("="*60)
    
    # Create test input with Severstal dimensions
    batch_size = 2
    height, width = 256, 1600  # Correct Severstal dimensions
    x = torch.randn(batch_size, 3, height, width)
    print(f"Input shape: {x.shape} (height={height}, width={width})")
    
    for model_name in ["efficientnet_b0", "efficientnet_b1"]:
        print(f"\n--- Testing {model_name} ---")
        
        # Test with pretrained weights and feature extraction
        print("1. Feature extraction with pretrained weights:")
        model_features = timm.create_model(model_name, pretrained=True, num_classes=0)
        
        with torch.no_grad():
            features = model_features(x)
            print(f"   Features shape: {features.shape}")
            print(f"   Features range: [{features.min().item():.6f}, {features.max().item():.6f}]")
            print(f"   Features mean: {features.mean().item():.6f}")
            print(f"   Features std: {features.std().item():.6f}")
            
            # Add custom classifier
            classifier = nn.Linear(features.shape[1], 4)
            nn.init.kaiming_normal_(classifier.weight, mode='fan_out', nonlinearity='relu')
            nn.init.constant_(classifier.bias, 0.01)
            
            logits = classifier(features)
            print(f"   Custom classifier output shape: {logits.shape}")
            print(f"   Custom classifier range: [{logits.min().item():.3f}, {logits.max().item():.3f}]")
            
            # Apply sigmoid to see probabilities
            probs = torch.sigmoid(logits)
            print(f"   Probabilities (first sample): {probs[0].numpy()}")
        
        # Test without pretrained weights for comparison
        print("2. Without pretrained weights:")
        model_no_pretrained = timm.create_model(model_name, pretrained=False, num_classes=0)
        
        with torch.no_grad():
            features_no_pretrained = model_no_pretrained(x)
            print(f"   Features range: [{features_no_pretrained.min().item():.6f}, {features_no_pretrained.max().item():.6f}]")
            
            classifier_no_pretrained = nn.Linear(features_no_pretrained.shape[1], 4)
            nn.init.kaiming_normal_(classifier_no_pretrained.weight, mode='fan_out', nonlinearity='relu')
            nn.init.constant_(classifier_no_pretrained.bias, 0.01)
            
            logits_no_pretrained = classifier_no_pretrained(features_no_pretrained)
            print(f"   Classifier range: [{logits_no_pretrained.min().item():.3f}, {logits_no_pretrained.max().item():.3f}]")

    print(f"\n✅ Test completed with Severstal dimensions!")

if __name__ == "__main__":
    test_with_severstal_dimensions()