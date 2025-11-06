"""
Debug script to understand EfficientNet feature dimensions and behavior.
"""

import torch
import timm

def debug_efficientnet():
    """Debug EfficientNet models to understand the issue."""
    
    print("🔍 DEBUGGING EFFICIENTNET MODELS")
    print("=" * 50)
    
    # Test input
    batch_size = 2
    input_tensor = torch.randn(batch_size, 3, 512, 1600)
    print(f"Input shape: {input_tensor.shape}")
    
    for model_name in ["efficientnet_b0", "efficientnet_b1"]:
        print(f"\n--- Debugging {model_name} ---")
        
        # Method 1: features_only=True
        print("Method 1: features_only=True")
        model_features = timm.create_model(model_name, pretrained=False, features_only=True)
        with torch.no_grad():
            features_list = model_features(input_tensor)
            print(f"Number of feature maps: {len(features_list)}")
            for i, feat in enumerate(features_list):
                print(f"  Feature {i}: {feat.shape}")
        
        # Method 2: num_classes=0 (global pooling applied)
        print("Method 2: num_classes=0")
        model_no_classifier = timm.create_model(model_name, pretrained=False, num_classes=0)
        with torch.no_grad():
            global_features = model_no_classifier(input_tensor)
            print(f"Global features shape: {global_features.shape}")
            print(f"Global features range: [{global_features.min().item():.6f}, {global_features.max().item():.6f}]")
            print(f"Global features mean: {global_features.mean().item():.6f}")
            print(f"Global features std: {global_features.std().item():.6f}")
        
        # Method 3: Get model info
        print("Method 3: Model info")
        model_info = timm.create_model(model_name, pretrained=False, num_classes=1000)
        print(f"Model info: {model_info.classifier}")
        print(f"Number of features: {model_info.classifier.in_features}")
        
        # Check if model is properly initialized
        print("Method 4: Check parameter initialization")
        total_params = sum(p.numel() for p in model_no_classifier.parameters())
        zero_params = sum((p == 0).sum().item() for p in model_no_classifier.parameters())
        print(f"Total parameters: {total_params:,}")
        print(f"Zero parameters: {zero_params:,} ({zero_params/total_params*100:.2f}%)")

if __name__ == "__main__":
    debug_efficientnet()