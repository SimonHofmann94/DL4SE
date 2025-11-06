"""
CBAM (Convolutional Block Attention Module) implementation.

This module provides spatial and channel attention mechanisms that can be
integrated into convolutional neural networks to improve feature discrimination.
"""

import torch
import torch.nn as nn


class ChannelAttention(nn.Module):
    """
    Channel Attention Module.
    
    Learns to reweight channels by squeezing spatial information and applying
    a shared MLP to capture channel relationships.
    
    Args:
        in_channels: Number of input channels
        reduction_ratio: Ratio for reducing dimensions in the MLP (default: 16)
    """
    
    def __init__(self, in_channels: int, reduction_ratio: int = 16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        
        # Shared MLP
        reduced_channels = max(1, in_channels // reduction_ratio)
        self.fc = nn.Sequential(
            nn.Linear(in_channels, reduced_channels, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(reduced_channels, in_channels, bias=False)
        )
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply channel attention.
        
        Args:
            x: Input tensor of shape (B, C, H, W)
        
        Returns:
            Attention-weighted tensor of same shape as input
        """
        # Squeeze: (B, C, H, W) -> (B, C)
        avg_out = self.fc(self.avg_pool(x).view(x.size(0), -1))
        max_out = self.fc(self.max_pool(x).view(x.size(0), -1))
        
        # Add and apply sigmoid
        channel_attention = self.sigmoid(avg_out + max_out)
        
        # Reshape for broadcasting: (B, C) -> (B, C, 1, 1)
        return x * channel_attention.unsqueeze(2).unsqueeze(3)


class SpatialAttention(nn.Module):
    """
    Spatial Attention Module.
    
    Learns to generate spatial attention maps by analyzing channel statistics
    (mean and max pooling along the channel dimension).
    
    Args:
        kernel_size: Kernel size for the convolutional layer (default: 7)
    """
    
    def __init__(self, kernel_size: int = 7):
        super().__init__()
        padding = kernel_size // 2
        self.conv = nn.Conv2d(
            in_channels=2,
            out_channels=1,
            kernel_size=kernel_size,
            padding=padding,
            bias=False
        )
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply spatial attention.
        
        Args:
            x: Input tensor of shape (B, C, H, W)
        
        Returns:
            Attention-weighted tensor of same shape as input
        """
        # Channel statistics: (B, C, H, W) -> (B, 1, H, W) each
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        
        # Concatenate: (B, 2, H, W)
        x_concat = torch.cat([avg_out, max_out], dim=1)
        
        # Apply convolution and sigmoid
        spatial_attention = self.sigmoid(self.conv(x_concat))
        
        # Apply attention
        return x * spatial_attention


class CBAM(nn.Module):
    """
    Convolutional Block Attention Module (CBAM).
    
    Sequentially applies channel attention followed by spatial attention.
    Designed to be easily integrated into any convolutional block.
    
    Args:
        in_channels: Number of input channels
        reduction_ratio: Reduction ratio for channel attention MLP (default: 16)
        spatial_kernel_size: Kernel size for spatial attention conv (default: 7)
    
    Example:
        >>> x = torch.randn(2, 64, 32, 32)
        >>> cbam = CBAM(in_channels=64)
        >>> out = cbam(x)
        >>> print(out.shape)
        torch.Size([2, 64, 32, 32])
    """
    
    def __init__(
        self,
        in_channels: int,
        reduction_ratio: int = 16,
        spatial_kernel_size: int = 7
    ):
        super().__init__()
        self.channel_attention = ChannelAttention(in_channels, reduction_ratio)
        self.spatial_attention = SpatialAttention(spatial_kernel_size)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply CBAM sequentially: channel attention -> spatial attention.
        
        Args:
            x: Input tensor of shape (B, C, H, W)
        
        Returns:
            Attention-weighted tensor of same shape as input
        """
        # Channel attention first
        x = self.channel_attention(x)
        # Then spatial attention
        x = self.spatial_attention(x)
        return x
