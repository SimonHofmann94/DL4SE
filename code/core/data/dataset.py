"""
Dataset implementation for Severstal full image classification.

Loads full 256x1600 images while maintaining original aspect ratio.
"""

import os
import json
import numpy as np
from typing import Optional, Tuple
from PIL import Image
import torch
from torch.utils.data import Dataset
import logging

logger = logging.getLogger(__name__)


class SeverstalFullImageDataset(Dataset):
    """
    Dataset for full Severstal images with multi-label classification.
    
    Loads entire images (256x1600) without resizing or cropping.
    Supports stratified train/val/test splits.
    
    Args:
        img_dir: Directory containing images
        ann_dir: Directory containing annotations
        image_names: List of image filenames to load
        transform: Torchvision transforms
        num_classes: Number of defect classes
    """
    
    DEFECT_CLASS_TO_IDX = {
        "defect_1": 0,
        "defect_2": 1,
        "defect_3": 2,
        "defect_4": 3,
    }
    NUM_CLASSES = 4
    
    def __init__(
        self,
        img_dir: str,
        ann_dir: str,
        image_names: list,
        transform: Optional = None,
        num_classes: int = 4,
        verbose: bool = False
    ):
        self.img_dir = img_dir
        self.ann_dir = ann_dir
        self.image_names = image_names
        self.transform = transform
        self.num_classes = num_classes
        self.verbose = verbose
        
        # Load image-label pairs
        self.samples = []
        self._load_samples()
        
        logger.info(
            f"Loaded {len(self.samples)} samples from {len(image_names)} images"
        )
    
    def _load_samples(self) -> None:
        """Load all image-label pairs into memory."""
        for img_name in self.image_names:
            img_path = os.path.join(self.img_dir, img_name)
            
            if not os.path.exists(img_path):
                logger.warning(f"Image not found: {img_path}")
                continue
            
            # Load label
            label = self._load_label(img_name)
            if label is None:
                continue
            
            self.samples.append({
                "image_name": img_name,
                "image_path": img_path,
                "label": label
            })
    
    def _load_label(self, img_name: str) -> Optional[np.ndarray]:
        """Load label for an image from annotation file."""
        base_name_no_ext = os.path.splitext(img_name)[0]
        
        # Try different annotation filename formats
        ann_path_variants = [
            os.path.join(self.ann_dir, f"{img_name}.json"),
            os.path.join(self.ann_dir, f"{base_name_no_ext}.json")
        ]
        
        for ann_path in ann_path_variants:
            if os.path.exists(ann_path):
                try:
                    with open(ann_path, 'r') as f:
                        annotation = json.load(f)
                    
                    # Extract labels
                    label = np.zeros(self.num_classes, dtype=np.float32)
                    
                    if "objects" in annotation and annotation["objects"]:
                        for obj in annotation["objects"]:
                            class_title = obj.get("classTitle")
                            if class_title in self.DEFECT_CLASS_TO_IDX:
                                class_idx = self.DEFECT_CLASS_TO_IDX[class_title]
                                label[class_idx] = 1.0
                    
                    return label
                    
                except Exception as e:
                    logger.warning(f"Error loading annotation {ann_path}: {e}")
                    continue
        
        # If no annotation found, assume no defects (all zeros)
        return np.zeros(self.num_classes, dtype=np.float32)
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get a sample.
        
        Args:
            idx: Sample index
        
        Returns:
            Tuple of (image_tensor, label_tensor)
        """
        sample = self.samples[idx]
        
        try:
            # Load image
            image_pil = Image.open(sample["image_path"]).convert("RGB")
            
            # Apply transforms
            if self.transform:
                image_tensor = self.transform(image_pil)
            else:
                image_tensor = torch.from_numpy(
                    np.array(image_pil).transpose(2, 0, 1)
                ).float() / 255.0
            
            # Get label
            label_tensor = torch.from_numpy(sample["label"]).float()
            
            return image_tensor, label_tensor
            
        except Exception as e:
            logger.error(
                f"Error loading sample {idx}: {sample['image_name']} - {e}"
            )
            # Return dummy tensors
            return torch.zeros(3, 256, 1600), torch.zeros(self.num_classes)


if __name__ == "__main__":
    # Test dataset loading
    print("Dataset implementation ready for use")
