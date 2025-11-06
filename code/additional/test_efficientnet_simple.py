"""
Simplified EfficientNet test to identify the exact issue.
"""

import torch
import torch.nn as nn
import timm

def test_efficientnet_simple():
    """Test EfficientNet with a simple approach."""
    
    print("🔍 SIMPLE EFFICIENTNET TEST")
    print("="*40)
    
    # Create test input
    x = torch.randn(1, 3, 224, 224)  # Smaller image for testing
    print(f"Input shape: {x.shape}")
    
    for model_name in ["efficientnet_b0", "efficientnet_b1"]:
        print(f"\n--- Testing {model_name} ---")
        
        # Test 1: Original model with pretrained weights
        print("1. With pretrained weights:")
        model_pretrained = timm.create_model(model_name, pretrained=True, num_classes=4)
        with torch.no_grad():
            out = model_pretrained(x)
            print(f"   Output shape: {out.shape}")
            print(f"   Output range: [{out.min().item():.3f}, {out.max().item():.3f}]")
        
        # Test 2: Model without pretrained weights
        print("2. Without pretrained weights:")
        model_no_pretrained = timm.create_model(model_name, pretrained=False, num_classes=4)
        with torch.no_grad():
            out = model_no_pretrained(x)
            print(f"   Output shape: {out.shape}")
            print(f"   Output range: [{out.min().item():.3f}, {out.max().item():.3f}]")
        
        # Test 3: Feature extraction
        print("3. Feature extraction (num_classes=0):")
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
            print(f"   Custom classifier output: [{logits.min().item():.3f}, {logits.max().item():.3f}]")

if __name__ == "__main__":
    test_efficientnet_simple()