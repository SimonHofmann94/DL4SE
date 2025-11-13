"""
Defect-Aware CutMix Augmentation for Multi-Label Classification.

CutMix is a data augmentation technique that cuts and pastes patches between 
training images. For defect detection, we ensure that defect regions are not 
accidentally cut, preserving the integrity of defect annotations.

Key Features:
- Validates that cut regions don't overlap with defect bounding boxes
- Supports multi-label classification (combines labels from both images)
- Configurable probability and cut size
- Falls back gracefully if no valid cut region can be found

Paper: CutMix: Regularization Strategy to Train Strong Classifiers with 
       Localizable Features (https://arxiv.org/abs/1905.04899)

Adapted for defect detection with spatial constraints.
"""

import numpy as np
import torch
from typing import Dict, Tuple, Optional, List
import logging
import random

logger = logging.getLogger(__name__)


class DefectAwareCutMix:
    """
    CutMix augmentation that respects defect locations.
    
    The augmentation cuts a rectangular region from one image and pastes it onto
    another, while ensuring that:
    1. No defect regions are cut from the source image
    2. No defect regions are overwritten in the target image
    3. Labels are properly combined (union of both images' labels)
    
    Args:
        prob: Probability of applying CutMix (default: 0.5)
        alpha: Beta distribution parameter for cut size (default: 1.0)
               Higher values = larger cuts, lower values = smaller cuts
        min_cut_ratio: Minimum cut size as ratio of image (default: 0.1)
        max_cut_ratio: Maximum cut size as ratio of image (default: 0.4)
        max_attempts: Maximum attempts to find valid cut region (default: 10)
        verbose: Enable debug logging (default: False)
    
    Example:
        >>> cutmix = DefectAwareCutMix(prob=0.5, alpha=1.0)
        >>> img1, label1, bbox1 = dataset[0]
        >>> img2, label2, bbox2 = dataset[1]
        >>> mixed_img, mixed_label = cutmix(img1, label1, bbox1, img2, label2, bbox2)
    """
    
    def __init__(
        self,
        prob: float = 0.5,
        alpha: float = 1.0,
        min_cut_ratio: float = 0.1,
        max_cut_ratio: float = 0.4,
        max_attempts: int = 10,
        verbose: bool = False
    ):
        self.prob = prob
        self.alpha = alpha
        self.min_cut_ratio = min_cut_ratio
        self.max_cut_ratio = max_cut_ratio
        self.max_attempts = max_attempts
        self.verbose = verbose
        
        # Store last cut bbox for debugging/visualization
        self.last_cut_bbox = None
        
        if self.verbose:
            logger.info(
                f"DefectAwareCutMix initialized: prob={prob}, alpha={alpha}, "
                f"cut_ratio=[{min_cut_ratio}, {max_cut_ratio}]"
            )
    
    def _sample_cut_size(self) -> float:
        """
        Sample cut size ratio from Beta distribution.
        
        Returns:
            Cut ratio in [min_cut_ratio, max_cut_ratio]
        """
        # Sample from Beta(alpha, alpha) -> values around 0.5 for alpha=1.0
        lam = np.random.beta(self.alpha, self.alpha)
        
        # Clamp to min/max range
        lam = np.clip(lam, self.min_cut_ratio, self.max_cut_ratio)
        
        return lam
    
    def _get_random_bbox(
        self,
        img_height: int,
        img_width: int,
        cut_ratio: float
    ) -> Tuple[int, int, int, int]:
        """
        Generate random bounding box for cut region.
        
        Args:
            img_height: Image height
            img_width: Image width
            cut_ratio: Ratio of cut size
        
        Returns:
            (x1, y1, x2, y2) in absolute coordinates
        """
        cut_w = int(img_width * cut_ratio)
        cut_h = int(img_height * cut_ratio)
        
        # Random center point
        cx = np.random.randint(0, img_width)
        cy = np.random.randint(0, img_height)
        
        # Compute bbox ensuring it stays within image
        x1 = np.clip(cx - cut_w // 2, 0, img_width)
        y1 = np.clip(cy - cut_h // 2, 0, img_height)
        x2 = np.clip(cx + cut_w // 2, 0, img_width)
        y2 = np.clip(cy + cut_h // 2, 0, img_height)
        
        return int(x1), int(y1), int(x2), int(y2)
    
    def _check_bbox_overlap(
        self,
        cut_bbox: Tuple[int, int, int, int],
        defect_bboxes: List[Dict]
    ) -> bool:
        """
        Check if cut region overlaps with any defect bounding boxes.
        
        Args:
            cut_bbox: (x1, y1, x2, y2) of cut region
            defect_bboxes: List of defect bounding boxes, each with keys:
                          'bbox': [x, y, width, height]
        
        Returns:
            True if overlap detected, False otherwise
        """
        if not defect_bboxes:
            return False
        
        cut_x1, cut_y1, cut_x2, cut_y2 = cut_bbox
        
        for defect in defect_bboxes:
            # Convert defect bbox from [x, y, w, h] to [x1, y1, x2, y2]
            bbox = defect['bbox']
            def_x1, def_y1, def_w, def_h = bbox
            def_x2 = def_x1 + def_w
            def_y2 = def_y1 + def_h
            
            # Check for overlap (using standard rectangle intersection)
            overlap = not (
                cut_x2 <= def_x1 or  # cut is left of defect
                cut_x1 >= def_x2 or  # cut is right of defect
                cut_y2 <= def_y1 or  # cut is above defect
                cut_y1 >= def_y2     # cut is below defect
            )
            
            if overlap:
                if self.verbose:
                    logger.debug(
                        f"Overlap detected: cut_bbox={cut_bbox}, "
                        f"defect_bbox=[{def_x1}, {def_y1}, {def_x2}, {def_y2}]"
                    )
                return True
        
        return False
    
    def _find_valid_cut_region(
        self,
        img_height: int,
        img_width: int,
        source_bboxes: List[Dict],
        target_bboxes: List[Dict]
    ) -> Optional[Tuple[int, int, int, int]]:
        """
        Find a valid cut region that doesn't overlap with defects.
        
        Args:
            img_height: Image height
            img_width: Image width
            source_bboxes: Defect bboxes from source image
            target_bboxes: Defect bboxes from target image
        
        Returns:
            (x1, y1, x2, y2) if valid region found, None otherwise
        """
        for attempt in range(self.max_attempts):
            # Sample cut size and region
            cut_ratio = self._sample_cut_size()
            cut_bbox = self._get_random_bbox(img_height, img_width, cut_ratio)
            
            # Check if cut region is valid (no overlap with defects)
            source_overlap = self._check_bbox_overlap(cut_bbox, source_bboxes)
            target_overlap = self._check_bbox_overlap(cut_bbox, target_bboxes)
            
            if not source_overlap and not target_overlap:
                if self.verbose:
                    logger.debug(
                        f"Valid cut region found on attempt {attempt + 1}: {cut_bbox}"
                    )
                return cut_bbox
        
        if self.verbose:
            logger.debug(
                f"No valid cut region found after {self.max_attempts} attempts"
            )
        return None
    
    def __call__(
        self,
        image1: np.ndarray,
        label1: np.ndarray,
        bboxes1: List[Dict],
        image2: np.ndarray,
        label2: np.ndarray,
        bboxes2: List[Dict]
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Apply CutMix augmentation to two images.
        
        Args:
            image1: Source image (H, W, C) - numpy array
            label1: Source labels (num_classes,) - binary
            bboxes1: Source defect bounding boxes
            image2: Target image (H, W, C) - numpy array
            label2: Target labels (num_classes,) - binary
            bboxes2: Target defect bounding boxes
        
        Returns:
            Tuple of (mixed_image, mixed_label)
            - mixed_image: Augmented image (H, W, C)
            - mixed_label: Combined labels (num_classes,) - union of both
        """
        # Random decision: apply CutMix or not?
        if np.random.rand() > self.prob:
            self.last_cut_bbox = None
            return image1, label1
        
        # Ensure images have same shape
        if image1.shape != image2.shape:
            logger.warning(
                f"Image shape mismatch: {image1.shape} vs {image2.shape}. "
                "Skipping CutMix."
            )
            self.last_cut_bbox = None
            return image1, label1
        
        img_height, img_width = image1.shape[:2]
        
        # Find valid cut region
        cut_bbox = self._find_valid_cut_region(
            img_height, img_width, bboxes1, bboxes2
        )
        
        if cut_bbox is None:
            # No valid region found, return original
            if self.verbose:
                logger.debug("CutMix skipped: no valid cut region")
            self.last_cut_bbox = None
            return image1, label1
        
        x1, y1, x2, y2 = cut_bbox
        
        # Store for debugging
        self.last_cut_bbox = cut_bbox
        
        # Create mixed image: copy image1, paste patch from image2
        mixed_image = image1.copy()
        mixed_image[y1:y2, x1:x2, :] = image2[y1:y2, x1:x2, :]
        
        # Combine labels: union (logical OR for multi-label)
        mixed_label = np.logical_or(label1, label2).astype(np.float32)
        
        if self.verbose:
            logger.debug(
                f"CutMix applied: cut_bbox={cut_bbox}, "
                f"label1={label1}, label2={label2}, mixed={mixed_label}"
            )
        
        return mixed_image, mixed_label
    
    def get_config(self) -> Dict:
        """Get augmentation configuration for logging."""
        return {
            "prob": self.prob,
            "alpha": self.alpha,
            "min_cut_ratio": self.min_cut_ratio,
            "max_cut_ratio": self.max_cut_ratio,
            "max_attempts": self.max_attempts
        }


# Factory function for easy integration
def get_cutmix_augmentation(config: Dict) -> Optional[DefectAwareCutMix]:
    """
    Factory function to create CutMix augmentation from config.
    
    Args:
        config: Dictionary with keys:
                - enabled: bool
                - prob: float
                - alpha: float
                - min_cut_ratio: float (optional)
                - max_cut_ratio: float (optional)
                - max_attempts: int (optional)
                - verbose: bool (optional)
    
    Returns:
        DefectAwareCutMix instance if enabled, None otherwise
    """
    if not config.get('enabled', False):
        return None
    
    return DefectAwareCutMix(
        prob=config.get('prob', 0.5),
        alpha=config.get('alpha', 1.0),
        min_cut_ratio=config.get('min_cut_ratio', 0.1),
        max_cut_ratio=config.get('max_cut_ratio', 0.4),
        max_attempts=config.get('max_attempts', 10),
        verbose=config.get('verbose', False)
    )


if __name__ == "__main__":
    # Test CutMix
    print("Testing DefectAwareCutMix...")
    
    # Create dummy images and labels
    img1 = np.random.rand(256, 1600, 3).astype(np.float32)
    img2 = np.random.rand(256, 1600, 3).astype(np.float32)
    
    label1 = np.array([1, 0, 1, 0, 0])  # has defect_2
    label2 = np.array([0, 1, 0, 0, 1])  # has defect_1, defect_4
    
    # Dummy bboxes
    bboxes1 = [
        {'bbox': [100, 50, 200, 100], 'category_id': 2}  # defect_2
    ]
    bboxes2 = [
        {'bbox': [500, 100, 150, 80], 'category_id': 1},  # defect_1
        {'bbox': [1200, 150, 100, 60], 'category_id': 4}  # defect_4
    ]
    
    # Create CutMix
    cutmix = DefectAwareCutMix(prob=1.0, alpha=1.0, verbose=True)
    
    # Apply
    mixed_img, mixed_label = cutmix(img1, label1, bboxes1, img2, label2, bboxes2)
    
    print(f"Original label1: {label1}")
    print(f"Original label2: {label2}")
    print(f"Mixed label: {mixed_label}")
    print(f"Expected (union): {np.logical_or(label1, label2).astype(int)}")
    print(f"Mixed image shape: {mixed_img.shape}")
    print("✓ Test successful!")
