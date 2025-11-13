"""
Defect Blackout Augmentation for Steel Defect Detection.

This augmentation randomly masks out (blackouts) defect regions in training images
to improve model robustness and prevent over-reliance on specific defect patterns.

The technique helps the model learn contextual features around defects rather than
just the defect appearance itself.

Can be applied selectively to specific defect types (e.g., only defect_2).
"""

import numpy as np
from PIL import Image, ImageDraw
from typing import List, Tuple, Optional, Dict
import torch
import logging

try:
    from scipy import ndimage
except ImportError:
    ndimage = None
    logging.warning(
        "scipy.ndimage not available. DefectBlackoutTransform will not work. "
        "Install with: pip install scipy"
    )

logger = logging.getLogger(__name__)


class DefectBlackoutTransform:
    """
    Randomly blackout (mask) defect regions in images.
    
    Two modes:
    1. Selective: Randomly blackout individual defect instances
    2. Complete: Blackout all defects (creates synthetic clean samples)
    
    Args:
        enabled: Whether to apply blackout (default: False)
        instance_blackout_prob: Probability of blacking out each defect instance (0.0-1.0)
        defect_indices_to_blackout: List of defect class indices to apply blackout to.
                                    If None, applies to all defects (1-4).
                                    Example: [2] to only blackout defect_2
        min_pixels_to_blackout: Minimum defect size (in pixels) to be considered for blackout
        fill_value: Pixel value to fill blacked-out regions (default: 0 = black)
        complete_blackout_prob: Probability of blacking out ALL defects instead of selective (0.0-1.0)
        verbose: Enable debug logging
    
    Example:
        >>> # Blackout only defect_2 with 50% probability per instance
        >>> blackout = DefectBlackoutTransform(
        ...     enabled=True,
        ...     instance_blackout_prob=0.5,
        ...     defect_indices_to_blackout=[2],  # Only defect_2
        ...     min_pixels_to_blackout=15
        ... )
        >>> 
        >>> # Apply to image and mask
        >>> img_aug, mask_aug, was_modified = blackout(image_pil, gt_mask_np)
    """
    
    def __init__(
        self,
        enabled: bool = False,
        instance_blackout_prob: float = 0.5,
        defect_indices_to_blackout: Optional[List[int]] = None,
        min_pixels_to_blackout: int = 10,
        fill_value: int = 0,
        complete_blackout_prob: float = 0.0,
        verbose: bool = False
    ):
        self.enabled = enabled
        self.instance_blackout_prob = instance_blackout_prob
        self.defect_indices_to_blackout = defect_indices_to_blackout or [1, 2, 3, 4]
        self.min_pixels_to_blackout = min_pixels_to_blackout
        self.fill_value = fill_value
        self.complete_blackout_prob = complete_blackout_prob
        self.verbose = verbose
        
        if self.enabled and ndimage is None:
            logger.error(
                "DefectBlackoutTransform is enabled but scipy.ndimage is not available! "
                "Install scipy or disable blackout augmentation."
            )
            self.enabled = False
        
        if self.enabled and self.verbose:
            logger.info(
                f"DefectBlackoutTransform initialized:\n"
                f"  Enabled: {self.enabled}\n"
                f"  Instance blackout probability: {self.instance_blackout_prob}\n"
                f"  Defect indices to blackout: {self.defect_indices_to_blackout}\n"
                f"  Min pixels: {self.min_pixels_to_blackout}\n"
                f"  Complete blackout probability: {self.complete_blackout_prob}"
            )
    
    def _blackout_selected_instances(
        self,
        image_pil: Image.Image,
        gt_mask_np: np.ndarray,
        instances_to_blackout: List[np.ndarray]
    ) -> Tuple[Image.Image, np.ndarray]:
        """
        Blackout specified defect instances in image and mask.
        
        Args:
            image_pil: PIL Image
            gt_mask_np: Ground truth mask (H, W) with class indices
            instances_to_blackout: List of binary masks for instances to blackout
        
        Returns:
            Tuple of (modified_image, modified_mask)
        """
        img_out_pil = image_pil.copy()
        mask_out_np = gt_mask_np.copy()
        draw = ImageDraw.Draw(img_out_pil)
        
        for instance_mask_pixels in instances_to_blackout:
            # Get pixel coordinates
            ys, xs = np.where(instance_mask_pixels)
            
            # Blackout in image
            for y_coord, x_coord in zip(ys, xs):
                draw.point((x_coord, y_coord), fill=self.fill_value)
            
            # Remove from mask (set to no_defect index = 0)
            mask_out_np[instance_mask_pixels] = 0
        
        return img_out_pil, mask_out_np
    
    def _blackout_all_instances(
        self,
        image_pil: Image.Image,
        gt_mask_np: np.ndarray
    ) -> Tuple[Image.Image, np.ndarray, bool]:
        """
        Blackout ALL defect instances (creates synthetic clean sample).
        
        Args:
            image_pil: PIL Image
            gt_mask_np: Ground truth mask
        
        Returns:
            Tuple of (modified_image, modified_mask, success_flag)
        """
        if self.verbose:
            logger.debug(f"[Blackout - ALL] Starting complete blackout")
        
        all_instances_to_blackout_masks = []
        
        # Check if there are any defects
        original_defect_pixels = np.sum(np.isin(gt_mask_np, self.defect_indices_to_blackout))
        if original_defect_pixels == 0:
            if self.verbose:
                logger.debug("[Blackout - ALL] No defects found, skipping")
            return image_pil, gt_mask_np, False
        
        # Find all defect instances
        for defect_idx in self.defect_indices_to_blackout:
            binary_mask_for_class = (gt_mask_np == defect_idx).astype(np.int32)
            labeled_array, num_features = ndimage.label(binary_mask_for_class)
            
            if num_features == 0:
                continue
            
            for i in range(1, num_features + 1):
                instance_mask_pixels = (labeled_array == i)
                if np.sum(instance_mask_pixels) >= self.min_pixels_to_blackout:
                    all_instances_to_blackout_masks.append(instance_mask_pixels)
        
        if not all_instances_to_blackout_masks:
            if self.verbose:
                logger.debug("[Blackout - ALL] No qualifying instances found")
            return image_pil, gt_mask_np, False
        
        img_out, mask_out = self._blackout_selected_instances(
            image_pil, gt_mask_np, all_instances_to_blackout_masks
        )
        
        if self.verbose:
            final_defect_pixels = np.sum(np.isin(mask_out, self.defect_indices_to_blackout))
            logger.debug(
                f"[Blackout - ALL] Blacked out {len(all_instances_to_blackout_masks)} instances. "
                f"Remaining defect pixels: {final_defect_pixels}"
            )
        
        return img_out, mask_out, True
    
    def __call__(
        self,
        image_pil: Image.Image,
        gt_mask_np: np.ndarray
    ) -> Tuple[Image.Image, np.ndarray, bool]:
        """
        Apply blackout augmentation.
        
        Args:
            image_pil: PIL Image to augment
            gt_mask_np: Ground truth mask (H, W) with class indices (0=no_defect, 1-4=defects)
        
        Returns:
            Tuple of:
                - Augmented image (PIL Image)
                - Augmented mask (numpy array)
                - Boolean flag indicating if modification was applied
        """
        # If not enabled, return unchanged
        if not self.enabled:
            return image_pil, gt_mask_np, False
        
        # If scipy not available, return unchanged
        if ndimage is None:
            return image_pil, gt_mask_np, False
        
        # Check if there are any defects to blackout
        original_defect_pixels = np.sum(np.isin(gt_mask_np, self.defect_indices_to_blackout))
        if original_defect_pixels == 0:
            if self.verbose:
                logger.debug("[Blackout] No target defects found, skipping")
            return image_pil, gt_mask_np, False
        
        # Decide whether to do complete or selective blackout
        if torch.rand(1).item() < self.complete_blackout_prob:
            # Complete blackout mode
            return self._blackout_all_instances(image_pil, gt_mask_np)
        
        # Selective blackout mode
        if self.verbose:
            logger.debug(
                f"[Blackout - SELECTIVE] Starting selective blackout. "
                f"Instance probability: {self.instance_blackout_prob}"
            )
        
        instances_randomly_selected_for_blackout = []
        num_total_qualified_instances = 0
        
        # Process each defect type
        for defect_idx in self.defect_indices_to_blackout:
            # Extract binary mask for this defect class
            binary_mask_for_class = (gt_mask_np == defect_idx).astype(np.int32)
            
            # Label connected components (instances)
            labeled_array, num_features = ndimage.label(binary_mask_for_class)
            
            if num_features == 0:
                continue
            
            # Process each instance
            for i in range(1, num_features + 1):
                instance_mask_pixels = (labeled_array == i)
                instance_size = np.sum(instance_mask_pixels)
                
                # Only consider instances above minimum size
                if instance_size >= self.min_pixels_to_blackout:
                    num_total_qualified_instances += 1
                    
                    # Randomly decide whether to blackout this instance
                    if torch.rand(1).item() < self.instance_blackout_prob:
                        instances_randomly_selected_for_blackout.append(instance_mask_pixels)
                        if self.verbose:
                            logger.debug(
                                f"[Blackout - SELECTIVE] Instance (defect_{defect_idx}, "
                                f"#{i}, {instance_size}px) will be blacked out"
                            )
        
        # If no instances were selected, return unchanged
        if not instances_randomly_selected_for_blackout:
            if self.verbose:
                logger.debug(
                    f"[Blackout - SELECTIVE] No instances selected for blackout "
                    f"(out of {num_total_qualified_instances} qualifying instances)"
                )
            return image_pil, gt_mask_np, False
        
        # Apply blackout
        img_out, mask_out = self._blackout_selected_instances(
            image_pil, gt_mask_np, instances_randomly_selected_for_blackout
        )
        
        if self.verbose:
            final_defect_pixels = np.sum(np.isin(mask_out, self.defect_indices_to_blackout))
            logger.debug(
                f"[Blackout - SELECTIVE] Blacked out {len(instances_randomly_selected_for_blackout)} "
                f"instances. Remaining defect pixels: {final_defect_pixels}"
            )
        
        return img_out, mask_out, True
    
    def get_config(self) -> Dict:
        """Get augmentation configuration for logging."""
        return {
            "enabled": self.enabled,
            "instance_blackout_prob": self.instance_blackout_prob,
            "defect_indices_to_blackout": self.defect_indices_to_blackout,
            "min_pixels_to_blackout": self.min_pixels_to_blackout,
            "complete_blackout_prob": self.complete_blackout_prob,
            "fill_value": self.fill_value
        }


if __name__ == "__main__":
    # Test the blackout transform
    import matplotlib.pyplot as plt
    
    # Create dummy image and mask
    img = Image.new("RGB", (1600, 256), color=(100, 100, 100))
    mask = np.zeros((256, 1600), dtype=np.uint8)
    
    # Add some synthetic defects
    mask[50:100, 200:300] = 2  # defect_2
    mask[150:180, 500:600] = 2  # defect_2
    mask[100:150, 800:900] = 1  # defect_1 (won't be blacked out if only targeting defect_2)
    
    print(f"Original mask - Defect pixels: defect_1={np.sum(mask==1)}, defect_2={np.sum(mask==2)}")
    
    # Test 1: Blackout only defect_2
    blackout = DefectBlackoutTransform(
        enabled=True,
        instance_blackout_prob=1.0,  # Always blackout for testing
        defect_indices_to_blackout=[2],  # Only defect_2
        verbose=True
    )
    
    img_aug, mask_aug, modified = blackout(img, mask)
    print(f"After blackout - Modified: {modified}")
    print(f"After blackout - Defect pixels: defect_1={np.sum(mask_aug==1)}, defect_2={np.sum(mask_aug==2)}")
    
    # Test 2: Complete blackout
    blackout_complete = DefectBlackoutTransform(
        enabled=True,
        complete_blackout_prob=1.0,  # Always do complete blackout
        defect_indices_to_blackout=[1, 2],
        verbose=True
    )
    
    img_aug2, mask_aug2, modified2 = blackout_complete(img, mask)
    print(f"After complete blackout - Modified: {modified2}")
    print(f"After complete blackout - All defects: {np.sum(mask_aug2>0)}")
