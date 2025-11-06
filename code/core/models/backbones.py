"""
ConvNext-Tiny model with integrated CBAM attention modules.

ConvNext is a modern CNN architecture that adopts design principles from Vision Transformers.
This implementation adds CBAM modules at strategic deeper layers to improve defect detection.
"""

import torch
import torch.nn as nn
from typing import Optional
from torchvision.models import convnext_tiny, ConvNeXt_Tiny_Weights

from .attention.cbam import CBAM


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


if __name__ == "__main__":
    # Example usage
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
