"""
Model architectures for Severstal Steel Defect Multi-Label Classification.
Supports DenseNet121, EfficientNetB0/B1, and ResNet backbones.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from typing import Optional, Dict, Any
import timm


class ClassifierModel(nn.Module):
    """
    Multi-label classifier for steel defect detection.
    Supports multiple backbone architectures.
    """
    
    def __init__(self, 
                 backbone: str = "densenet121",
                 num_classes: int = 4,
                 pretrained: bool = True,
                 dropout_rate: float = 0.5,
                 use_global_avg_pool: bool = True):
        """
        Initialize classifier model.
        
        Args:
            backbone: Backbone architecture ('densenet121', 'efficientnet_b0', 'efficientnet_b1', 'resnet50')
            num_classes: Number of output classes (4 for defects)
            pretrained: Use ImageNet pretrained weights
            dropout_rate: Dropout rate before final layer
            use_global_avg_pool: Use global average pooling instead of adaptive
        """
        super(ClassifierModel, self).__init__()
        
        self.backbone_name = backbone
        self.num_classes = num_classes
        self.dropout_rate = dropout_rate
        
        # Initialize backbone and get correct feature dimensions
        if backbone == "densenet121":
            self.backbone = self._create_densenet121(pretrained)
            self.feature_dim = 1024
        elif backbone == "efficientnet_b0":
            self.backbone = self._create_efficientnet_b0(pretrained)
            self.feature_dim = 1280  # EfficientNet-B0 always outputs 1280 features
        elif backbone == "efficientnet_b1":
            self.backbone = self._create_efficientnet_b1(pretrained)
            self.feature_dim = 1280  # EfficientNet-B1 also outputs 1280 features
        elif backbone == "resnet50":
            self.backbone = self._create_resnet50(pretrained)
            self.feature_dim = 2048
        else:
            raise ValueError(f"Unsupported backbone: {backbone}")
        
        # Global pooling
        if use_global_avg_pool:
            self.global_pool = nn.AdaptiveAvgPool2d(1)
        else:
            self.global_pool = nn.AdaptiveMaxPool2d(1)
        
        # Classifier head
        self.dropout = nn.Dropout(dropout_rate)
        self.classifier = nn.Linear(self.feature_dim, num_classes)
        
        # Initialize classifier weights
        self._initialize_classifier()
    
    def _create_densenet121(self, pretrained: bool) -> nn.Module:
        """Create DenseNet121 backbone."""
        model = models.densenet121(pretrained=pretrained)
        # Remove classifier layer, keep features
        return model.features
    
    def _create_efficientnet_b0(self, pretrained: bool) -> nn.Module:
        """Create EfficientNet-B0 backbone using timm."""
        # Create model without classifier head (num_classes=0 removes the final fc layer)
        model = timm.create_model('efficientnet_b0', pretrained=pretrained, num_classes=0)
        return model
    
    def _create_efficientnet_b1(self, pretrained: bool) -> nn.Module:
        """Create EfficientNet-B1 backbone using timm."""
        # Create model without classifier head (num_classes=0 removes the final fc layer)
        model = timm.create_model('efficientnet_b1', pretrained=pretrained, num_classes=0)
        return model
    

    
    def _create_resnet50(self, pretrained: bool) -> nn.Module:
        """Create ResNet50 backbone."""
        model = models.resnet50(pretrained=pretrained)
        # Remove avgpool and fc layers
        return nn.Sequential(*list(model.children())[:-2])
    
    def _initialize_classifier(self):
        """Initialize classifier layer weights."""
        # Use stronger initialization for better initial gradients
        nn.init.kaiming_normal_(self.classifier.weight, mode='fan_out', nonlinearity='relu')
        nn.init.constant_(self.classifier.bias, 0.01)  # Small positive bias
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input tensor of shape (batch_size, 3, height, width)
            
        Returns:
            Logits tensor of shape (batch_size, num_classes)
        """
        batch_size = x.size(0)
        
        # Backbone feature extraction
        if self.backbone_name.startswith("efficientnet"):
            # EfficientNet with num_classes=0 returns global pooled features
            features = self.backbone(x)  # Shape: (batch_size, feature_dim)
            # EfficientNet already applies global pooling, so we can skip manual pooling
            if len(features.shape) == 2:  # Already flattened
                dropped = self.dropout(features)
                logits = self.classifier(dropped)
                return logits
            else:
                # If not flattened, apply our pooling
                pooled = self.global_pool(features)
                flattened = pooled.view(batch_size, -1)
                dropped = self.dropout(flattened)
                logits = self.classifier(dropped)
                return logits
        else:
            # DenseNet and ResNet
            features = self.backbone(x)
        
        # Global pooling
        pooled = self.global_pool(features)
        
        # Flatten
        flattened = pooled.view(batch_size, -1)
        
        # Dropout and classification
        dropped = self.dropout(flattened)
        logits = self.classifier(dropped)
        
        return logits
    
    def get_feature_maps(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extract feature maps before global pooling.
        Useful for visualization and analysis.
        
        Args:
            x: Input tensor
            
        Returns:
            Feature maps tensor
        """
        if self.backbone_name.startswith("efficientnet"):
            # For EfficientNet, we need to use features_only=True model
            features_model = timm.create_model(
                self.backbone_name, 
                pretrained=False, 
                features_only=True
            )
            features = features_model(x)[-1]  # Last feature map
        else:
            features = self.backbone(x)
        
        return features


class FocalLoss(nn.Module):
    """
    Focal Loss for addressing class imbalance in multi-label classification.
    """
    
    def __init__(self, alpha: float = 1.0, gamma: float = 2.0, reduction: str = 'mean'):
        """
        Initialize Focal Loss.
        
        Args:
            alpha: Weighting factor for rare class
            gamma: Focusing parameter
            reduction: Reduction method ('mean', 'sum', 'none')
        """
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
    
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            inputs: Predicted logits of shape (batch_size, num_classes)
            targets: Ground truth labels of shape (batch_size, num_classes)
            
        Returns:
            Focal loss value
        """
        # Convert logits to probabilities
        probs = torch.sigmoid(inputs)
        
        # Calculate BCE loss
        bce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        
        # Calculate focal weight
        p_t = probs * targets + (1 - probs) * (1 - targets)
        focal_weight = self.alpha * (1 - p_t) ** self.gamma
        
        # Apply focal weight
        focal_loss = focal_weight * bce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


def create_model(config: Dict[str, Any]) -> ClassifierModel:
    """
    Create model from configuration.
    
    Args:
        config: Model configuration dictionary
        
    Returns:
        Initialized model
    """
    return ClassifierModel(
        backbone=config.get('backbone', 'densenet121'),
        num_classes=config.get('num_classes', 4),
        pretrained=config.get('pretrained', True),
        dropout_rate=config.get('dropout_rate', 0.5),
        use_global_avg_pool=config.get('use_global_avg_pool', True)
    )


def count_parameters(model: nn.Module) -> tuple:
    """Count total and trainable parameters."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


if __name__ == "__main__":
    """Test model architectures."""
    
    print("Testing Severstal Classifier Models...")
    
    # Test different backbones
    backbones = ["densenet121", "efficientnet_b0", "efficientnet_b1", "resnet50"]
    
    for backbone in backbones:
        print(f"\n--- Testing {backbone} ---")
        
        try:
            # Create model
            model = ClassifierModel(
                backbone=backbone,
                num_classes=4,
                pretrained=False,  # Faster for testing
                dropout_rate=0.5
            )
            
            # Count parameters
            total_params, trainable_params = count_parameters(model)
            print(f"Total parameters: {total_params:,}")
            print(f"Trainable parameters: {trainable_params:,}")
            
            # Test forward pass
            batch_size = 2
            height, width = 512, 1600
            dummy_input = torch.randn(batch_size, 3, height, width)
            
            model.eval()
            with torch.no_grad():
                output = model(dummy_input)
                print(f"Input shape: {dummy_input.shape}")
                print(f"Output shape: {output.shape}")
                print(f"Output range: [{output.min().item():.3f}, {output.max().item():.3f}]")
                
                # Test feature maps
                feature_maps = model.get_feature_maps(dummy_input)
                print(f"Feature maps shape: {feature_maps.shape}")
            
            print(f"✅ {backbone} test passed!")
            
        except Exception as e:
            print(f"❌ {backbone} test failed: {e}")
    
    # Test Focal Loss
    print(f"\n--- Testing Focal Loss ---")
    focal_loss = FocalLoss(alpha=1.0, gamma=2.0)
    
    # Dummy predictions and targets
    predictions = torch.randn(4, 4)  # 4 samples, 4 classes
    targets = torch.randint(0, 2, (4, 4)).float()  # Binary targets
    
    loss = focal_loss(predictions, targets)
    print(f"Focal loss: {loss.item():.4f}")
    print("✅ Focal Loss test passed!")
    
    print("\n🎉 All model tests completed!")