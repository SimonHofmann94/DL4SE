"""
Augmentation pipeline with reusable, toggleable components.

Each augmentation can be independently enabled/disabled via configuration,
making it easy to test different combinations.
"""

import torchvision.transforms as transforms
from typing import Dict, List, Optional
import logging

logger = logging.getLogger(__name__)


class AugmentationPipeline:
    """
    Composable augmentation pipeline with individual toggles.
    
    Each augmentation component can be independently enabled/disabled.
    New augmentations can be easily added to the registry.
    
    Example:
        >>> pipeline = AugmentationPipeline(
        ...     image_size=(256, 1600),
        ...     augmentations={
        ...         "horizontal_flip": 0.5,
        ...         "vertical_flip": 0.3,
        ...         "rotation": {"probability": 0.5, "degrees": 15},
        ...         "brightness_contrast": {"brightness": 0.2, "contrast": 0.2},
        ...     }
        ... )
        >>> transform = pipeline.build()
    """
    
    def __init__(
        self,
        image_size: tuple = (256, 1600),
        augmentations: Optional[Dict] = None
    ):
        self.image_size = image_size
        self.augmentations = augmentations or {}
        self.transform_list = []
    
    def build(self) -> transforms.Compose:
        """
        Build the augmentation pipeline from configuration.
        
        Returns:
            torchvision.transforms.Compose object
        """
        self.transform_list = []
        
        logger.info("Building augmentation pipeline...")
        
        # Add augmentations based on config
        for aug_name, aug_config in self.augmentations.items():
            if aug_config is None or (isinstance(aug_config, bool) and not aug_config):
                continue
            
            if aug_name == "horizontal_flip" and aug_config:
                prob = aug_config if isinstance(aug_config, (int, float)) else 0.5
                self.transform_list.append(
                    transforms.RandomHorizontalFlip(p=prob)
                )
                logger.debug(f"Added: RandomHorizontalFlip(p={prob})")
            
            elif aug_name == "vertical_flip" and aug_config:
                prob = aug_config if isinstance(aug_config, (int, float)) else 0.3
                self.transform_list.append(
                    transforms.RandomVerticalFlip(p=prob)
                )
                logger.debug(f"Added: RandomVerticalFlip(p={prob})")
            
            elif aug_name == "rotation" and aug_config:
                if isinstance(aug_config, dict):
                    degrees = aug_config.get("degrees", 15)
                    prob = aug_config.get("probability", 1.0)
                else:
                    degrees = 15
                    prob = 1.0
                
                self.transform_list.append(
                    transforms.RandomRotation(degrees=degrees)
                )
                logger.debug(f"Added: RandomRotation(degrees={degrees})")
            
            elif aug_name == "brightness_contrast" and aug_config:
                if isinstance(aug_config, dict):
                    brightness = aug_config.get("brightness", 0.2)
                    contrast = aug_config.get("contrast", 0.2)
                else:
                    brightness = 0.2
                    contrast = 0.2
                
                self.transform_list.append(
                    transforms.ColorJitter(
                        brightness=brightness,
                        contrast=contrast,
                        saturation=0.0,
                        hue=0.0
                    )
                )
                logger.debug(
                    f"Added: ColorJitter(brightness={brightness}, contrast={contrast})"
                )
            
            elif aug_name == "color_jitter" and aug_config:
                if isinstance(aug_config, dict):
                    brightness = aug_config.get("brightness", 0.1)
                    contrast = aug_config.get("contrast", 0.1)
                    saturation = aug_config.get("saturation", 0.1)
                    hue = aug_config.get("hue", 0.05)
                else:
                    brightness = contrast = saturation = 0.1
                    hue = 0.05
                
                self.transform_list.append(
                    transforms.ColorJitter(
                        brightness=brightness,
                        contrast=contrast,
                        saturation=saturation,
                        hue=hue
                    )
                )
                logger.debug(
                    f"Added: ColorJitter(b={brightness}, c={contrast}, s={saturation}, h={hue})"
                )
            
            elif aug_name == "gaussian_blur" and aug_config:
                if isinstance(aug_config, dict):
                    kernel_size = aug_config.get("kernel_size", 3)
                    sigma = aug_config.get("sigma", (0.1, 2.0))
                else:
                    kernel_size = 3
                    sigma = (0.1, 2.0)
                
                self.transform_list.append(
                    transforms.GaussianBlur(kernel_size=kernel_size, sigma=sigma)
                )
                logger.debug(f"Added: GaussianBlur(kernel_size={kernel_size})")
        
        # Always add: Convert to tensor and normalize
        self.transform_list.extend([
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
        logger.debug("Added: ToTensor and Normalize (ImageNet stats)")
        
        logger.info(f"Built augmentation pipeline with {len(self.transform_list)} transforms")
        
        return transforms.Compose(self.transform_list)


def get_augmentation_pipeline(
    image_size: tuple = (256, 1600),
    augmentation_config: Optional[Dict] = None,
    preset: str = "standard"
) -> transforms.Compose:
    """
    Get augmentation pipeline from configuration or preset.
    
    Args:
        image_size: Target image size (H, W)
        augmentation_config: Dictionary of augmentation configs
        preset: Preset name ('standard', 'light', 'heavy') if augmentation_config is None
    
    Returns:
        torchvision.transforms.Compose object
    """
    
    if augmentation_config is None:
        # Use preset
        presets = {
            "standard": {
                "horizontal_flip": 0.5,
                "vertical_flip": 0.3,
                "rotation": {"degrees": 15, "probability": 0.5},
                "brightness_contrast": {"brightness": 0.2, "contrast": 0.2},
            },
            "light": {
                "horizontal_flip": 0.3,
                "brightness_contrast": {"brightness": 0.1, "contrast": 0.1},
            },
            "heavy": {
                "horizontal_flip": 0.7,
                "vertical_flip": 0.5,
                "rotation": {"degrees": 30, "probability": 0.7},
                "brightness_contrast": {"brightness": 0.3, "contrast": 0.3},
                "color_jitter": {"saturation": 0.2, "hue": 0.1},
                "gaussian_blur": {"kernel_size": 3},
            },
            "none": {}
        }
        
        if preset not in presets:
            logger.warning(f"Unknown preset '{preset}', using 'standard'")
            preset = "standard"
        
        augmentation_config = presets[preset]
        logger.info(f"Using preset augmentations: {preset}")
    
    pipeline = AugmentationPipeline(
        image_size=image_size,
        augmentations=augmentation_config
    )
    
    return pipeline.build()


if __name__ == "__main__":
    # Test augmentation pipeline
    from PIL import Image
    import numpy as np
    
    # Create dummy image
    dummy_img = Image.new("RGB", (1600, 256), color="red")
    
    # Test standard preset
    transform = get_augmentation_pipeline(
        image_size=(256, 1600),
        preset="standard"
    )
    
    print("Testing standard preset augmentations:")
    try:
        transformed = transform(dummy_img)
        print(f"Success! Output shape: {transformed.shape}")
    except Exception as e:
        print(f"Error: {e}")
    
    # Test custom augmentations
    custom_aug = {
        "horizontal_flip": 0.5,
        "rotation": {"degrees": 20},
    }
    
    transform2 = get_augmentation_pipeline(
        image_size=(256, 1600),
        augmentation_config=custom_aug
    )
    
    print("\nTesting custom augmentations:")
    try:
        transformed2 = transform2(dummy_img)
        print(f"Success! Output shape: {transformed2.shape}")
    except Exception as e:
        print(f"Error: {e}")
