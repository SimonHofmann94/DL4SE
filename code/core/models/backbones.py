"""
Model backbones for Severstal Steel Defect Classification.

This module contains various CNN architectures optimized for multi-label defect detection:
- ConvNextTinyCBAM: ConvNext-Tiny with CBAM attention modules
- ConvNextTiny: Standard ConvNext-Tiny without attention
- ResNet50: Classic ResNet-50 architecture

All models are adapted for multi-label classification with consistent interfaces.
"""

import torch
import torch.nn as nn
from typing import Optional
from torchvision.models import (
    convnext_tiny, ConvNeXt_Tiny_Weights,
    resnet50, ResNet50_Weights
)

# Handle both relative and absolute imports
try:
    from .attention.cbam import CBAM
except ImportError:
    from attention.cbam import CBAM


class ConvNextTinyCBAM(nn.Module):
    """
    ConvNext-Tiny with CBAM attention modules integrated at deeper stages.
    
    Architecture:
    - Stages 0-2: Early feature extraction (no CBAM)
    - Stages 3-4: Semantic features (with CBAM for defect detection)
    
    The CBAM modules are placed in stages 3-4 where the receptive fields
    are large enough to capture small defects while still maintaining spatial information.
    
    Args:
        num_classes: Number of output classes (default: 4 for Severstal)
        pretrained: Whether to use ImageNet pretrained weights (default: True)
        cbam_stages: List of stages to add CBAM to (default: [3, 4])
        drop_path_rate: Stochastic depth rate (default: 0.0)
    """
    
    def __init__(
        self,
        num_classes: int = 4,
        pretrained: bool = True,
        cbam_stages: list = None,
        drop_path_rate: float = 0.0
    ):
        super().__init__()
        
        self.num_classes = num_classes
        self.cbam_stages = cbam_stages or [3, 4]
        
        # Load pretrained ConvNext-Tiny
        if pretrained:
            self.backbone = convnext_tiny(weights=ConvNeXt_Tiny_Weights.IMAGENET1K_V1)
        else:
            self.backbone = convnext_tiny(weights=None)
        
        # Get the features module (everything except the classifier)
        self.features = self.backbone.features
        self.avgpool = self.backbone.avgpool
        
        # Store original layers structure
        # ConvNext-Tiny has 4 stages, each stage has sequential blocks
        self._add_cbam_modules()
        
        # Replace classifier for multi-label classification
        # ConvNext uses (num_classes,) output for classification
        num_features = self.backbone.classifier[2].in_features
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(num_features, num_classes, bias=True)
        )
    
    def _add_cbam_modules(self) -> None:
        """
        Add CBAM modules to specified stages.
        
        For ConvNext-Tiny, we modify the features sequentially by
        wrapping blocks in each target stage with CBAM.
        """
        for stage_idx in self.cbam_stages:
            if 0 <= stage_idx < len(self.features):
                stage_block = self.features[stage_idx]
                
                # Get the number of channels from the stage
                # Each stage in ConvNext is a sequential container
                if isinstance(stage_block, nn.Sequential) and len(stage_block) > 0:
                    # Get channel info from first conv in the stage
                    for module in stage_block.modules():
                        if isinstance(module, nn.Conv2d):
                            in_channels = module.out_channels
                            break
                    
                    # Wrap this stage with CBAM
                    # Create a wrapper that applies CBAM after stage processing
                    original_stage = self.features[stage_idx]
                    
                    class StageWithCBAM(nn.Module):
                        def __init__(self, stage, cbam):
                            super().__init__()
                            self.stage = stage
                            self.cbam = cbam
                        
                        def forward(self, x):
                            x = self.stage(x)
                            x = self.cbam(x)
                            return x
                    
                    # Get channels from the last conv layer in stage
                    last_channels = None
                    for module in reversed(list(original_stage.modules())):
                        if isinstance(module, nn.Conv2d):
                            last_channels = module.out_channels
                            break
                    
                    if last_channels is not None:
                        cbam = CBAM(in_channels=last_channels)
                        self.features[stage_idx] = StageWithCBAM(original_stage, cbam)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through ConvNext-Tiny with CBAM.
        
        Args:
            x: Input tensor of shape (B, 3, H, W)
        
        Returns:
            Logits tensor of shape (B, num_classes) for multi-label classification
        """
        # Features
        x = self.features(x)
        
        # Average pooling
        x = self.avgpool(x)
        
        # Flatten
        x = torch.flatten(x, 1)
        
        # Classification head
        x = self.classifier(x)
        
        return x
    
    def get_attention_modules(self) -> list:
        """
        Get all CBAM modules in the model for inspection or visualization.
        
        Returns:
            List of CBAM modules
        """
        cbam_modules = []
        for module in self.modules():
            if isinstance(module, CBAM):
                cbam_modules.append(module)
        return cbam_modules
    
    def freeze_backbone(self, freeze: bool = True) -> None:
        """
        Freeze or unfreeze backbone weights (for transfer learning).
        
        Args:
            freeze: If True, freeze backbone; if False, unfreeze
        """
        for param in self.features.parameters():
            param.requires_grad = not freeze
    
    def freeze_early_stages(self, num_stages: int = 2) -> None:
        """
        Freeze early stages of the backbone.
        
        Args:
            num_stages: Number of stages to freeze from the beginning
        """
        for stage_idx in range(min(num_stages, len(self.features))):
            for param in self.features[stage_idx].parameters():
                param.requires_grad = False


class ConvNextTiny(nn.Module):
    """
    Standard ConvNext-Tiny without attention modules.
    
    This is the baseline ConvNext-Tiny model for comparison with the CBAM variant.
    Uses the same architecture but without additional attention mechanisms.
    
    Args:
        num_classes: Number of output classes (default: 4 for Severstal)
        pretrained: Whether to use ImageNet pretrained weights (default: True)
        drop_path_rate: Stochastic depth rate (default: 0.0)
        **kwargs: Additional arguments (for compatibility with registry)
    """
    
    def __init__(
        self,
        num_classes: int = 4,
        pretrained: bool = True,
        drop_path_rate: float = 0.0,
        **kwargs  # Accept but ignore extra args like cbam_stages
    ):
        super().__init__()
        
        self.num_classes = num_classes
        
        # Load pretrained ConvNext-Tiny
        if pretrained:
            self.backbone = convnext_tiny(weights=ConvNeXt_Tiny_Weights.IMAGENET1K_V1)
        else:
            self.backbone = convnext_tiny(weights=None)
        
        # Get the features module (everything except the classifier)
        self.features = self.backbone.features
        self.avgpool = self.backbone.avgpool
        
        # Replace classifier for multi-label classification
        num_features = self.backbone.classifier[2].in_features
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(num_features, num_classes, bias=True)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through ConvNext-Tiny.
        
        Args:
            x: Input tensor of shape (B, 3, H, W)
        
        Returns:
            Logits tensor of shape (B, num_classes) for multi-label classification
        """
        # Features
        x = self.features(x)
        
        # Average pooling
        x = self.avgpool(x)
        
        # Flatten
        x = torch.flatten(x, 1)
        
        # Classification head
        x = self.classifier(x)
        
        return x
    
    def freeze_backbone(self, freeze: bool = True) -> None:
        """
        Freeze or unfreeze backbone weights (for transfer learning).
        
        Args:
            freeze: If True, freeze backbone; if False, unfreeze
        """
        for param in self.features.parameters():
            param.requires_grad = not freeze
    
    def freeze_early_stages(self, num_stages: int = 2) -> None:
        """
        Freeze early stages of the backbone.
        
        Args:
            num_stages: Number of stages to freeze from the beginning
        """
        for stage_idx in range(min(num_stages, len(self.features))):
            for param in self.features[stage_idx].parameters():
                param.requires_grad = False


class ResNet50(nn.Module):
    """
    ResNet-50 architecture for multi-label defect classification.
    
    Classic residual network with 50 layers. Uses ImageNet pretrained weights
    and adapts the final layer for multi-label classification.
    
    Args:
        num_classes: Number of output classes (default: 4 for Severstal)
        pretrained: Whether to use ImageNet pretrained weights (default: True)
        **kwargs: Additional arguments (for compatibility with registry)
    """
    
    def __init__(
        self,
        num_classes: int = 4,
        pretrained: bool = True,
        **kwargs  # Accept but ignore extra args like cbam_stages
    ):
        super().__init__()
        
        self.num_classes = num_classes
        
        # Load pretrained ResNet-50
        if pretrained:
            self.backbone = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
        else:
            self.backbone = resnet50(weights=None)
        
        # Get number of features from the last layer
        num_features = self.backbone.fc.in_features
        
        # Replace final fully connected layer for multi-label classification
        self.backbone.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(num_features, num_classes, bias=True)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through ResNet-50.
        
        Args:
            x: Input tensor of shape (B, 3, H, W)
        
        Returns:
            Logits tensor of shape (B, num_classes) for multi-label classification
        """
        return self.backbone(x)
    
    def freeze_backbone(self, freeze: bool = True) -> None:
        """
        Freeze or unfreeze backbone weights (for transfer learning).
        
        Args:
            freeze: If True, freeze all layers except fc; if False, unfreeze all
        """
        # Freeze all layers except the final fc layer
        for name, param in self.backbone.named_parameters():
            if not name.startswith('fc'):
                param.requires_grad = not freeze
    
    def freeze_early_stages(self, num_stages: int = 2) -> None:
        """
        Freeze early stages of ResNet (conv1, bn1, layer1, layer2, etc.).
        
        Args:
            num_stages: Number of stages to freeze (0-4)
                       0: conv1 + bn1
                       1: + layer1
                       2: + layer2
                       3: + layer3
                       4: + layer4
        """
        # Always freeze conv1 and bn1 if num_stages > 0
        if num_stages >= 1:
            for param in self.backbone.conv1.parameters():
                param.requires_grad = False
            for param in self.backbone.bn1.parameters():
                param.requires_grad = False
        
        # Freeze layer1, layer2, layer3, layer4 based on num_stages
        layer_names = ['layer1', 'layer2', 'layer3', 'layer4']
        for i in range(min(num_stages, len(layer_names))):
            layer = getattr(self.backbone, layer_names[i])
            for param in layer.parameters():
                param.requires_grad = False


if __name__ == "__main__":
    # Example usage - Test all models
    print("="*70)
    print("Testing ConvNextTinyCBAM")
    print("="*70)
    model = ConvNextTinyCBAM(num_classes=4, pretrained=True)
    
    # Test forward pass
    x = torch.randn(2, 3, 256, 1600)  # Severstal image size
    output = model(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Expected shape: (2, 4)")
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nTotal parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # Show CBAM modules
    cbam_modules = model.get_attention_modules()
    print(f"Number of CBAM modules: {len(cbam_modules)}")
    
    print("\n" + "="*70)
    print("Testing ConvNextTiny (without CBAM)")
    print("="*70)
    model2 = ConvNextTiny(num_classes=4, pretrained=True)
    output2 = model2(x)
    print(f"Output shape: {output2.shape}")
    total_params2 = sum(p.numel() for p in model2.parameters())
    print(f"Total parameters: {total_params2:,}")
    
    print("\n" + "="*70)
    print("Testing ResNet50")
    print("="*70)
    model3 = ResNet50(num_classes=4, pretrained=True)
    output3 = model3(x)
    print(f"Output shape: {output3.shape}")
    total_params3 = sum(p.numel() for p in model3.parameters())
    print(f"Total parameters: {total_params3:,}")

