"""
Severstal Full Image Dataset for Multi-Label Classification
Processes complete images instead of patches for steel defect detection.
"""

import os
import json
from typing import Optional, Tuple, List, Dict
import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms
import numpy as np
import base64
import zlib
from io import BytesIO


class SeverstalFullImageDataset(Dataset):
    """
    Dataset for full image multi-label classification of steel defects.
    Processes complete images with 4 defect classes.
    """
    
    DEFECT_CLASS_TO_IDX = {
        "defect_1": 0, "defect_2": 1, "defect_3": 2, "defect_4": 3
    }
    NUM_CLASSES = 4
    
    def __init__(self,
                 img_dir: str,
                 ann_dir: str,
                 transform: Optional[transforms.Compose] = None,
                 target_size: Optional[Tuple[int, int]] = None,
                 debug_limit: Optional[int] = None,
                 image_filenames: Optional[List[str]] = None):
        """
        Initialize Severstal Full Image Dataset.
        
        Args:
            img_dir: Directory containing images
            ann_dir: Directory containing JSON annotations
            transform: Image transformations
            target_size: (height, width) for resizing, None to keep original
            debug_limit: Limit number of images for debugging
            image_filenames: Specific list of image filenames to use
        """
        self.img_dir = img_dir
        self.ann_dir = ann_dir
        self.transform = transform
        self.target_size = target_size
        
        # Get image files
        if image_filenames is not None:
            self.image_files = sorted(list(set(image_filenames)))
            print(f"SeverstalFullImageDataset: Using provided list of {len(self.image_files)} images")
        else:
            all_images = sorted([f for f in os.listdir(img_dir) 
                               if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
            
            if debug_limit is not None and debug_limit > 0:
                self.image_files = all_images[:debug_limit]
                print(f"SeverstalFullImageDataset: Debug mode - using {len(self.image_files)} images")
            else:
                self.image_files = all_images
                print(f"SeverstalFullImageDataset: Using all {len(self.image_files)} images")
        
        # Verify directories exist
        if not os.path.exists(img_dir):
            raise ValueError(f"Image directory not found: {img_dir}")
        if not os.path.exists(ann_dir):
            raise ValueError(f"Annotation directory not found: {ann_dir}")
            
        print(f"Dataset initialized with {len(self.image_files)} images")
    
    def __len__(self) -> int:
        return len(self.image_files)
    
    def _load_annotation(self, img_name: str) -> torch.Tensor:
        """
        Load annotation and create multi-label target vector.
        
        Args:
            img_name: Image filename
            
        Returns:
            Binary tensor of shape (num_classes,) indicating presence of each defect
        """
        # Create label vector - all zeros initially
        label_vector = torch.zeros(self.NUM_CLASSES, dtype=torch.float32)
        
        # Find annotation file
        base_name = os.path.splitext(img_name)[0]
        ann_path_variants = [
            os.path.join(self.ann_dir, f"{img_name}.json"),
            os.path.join(self.ann_dir, f"{base_name}.json"),
            os.path.join(self.ann_dir, f"{base_name}.jpg.json")
        ]
        
        ann_path = None
        for variant in ann_path_variants:
            if os.path.exists(variant):
                ann_path = variant
                break
        
        if ann_path is None:
            # No annotation found - return all zeros (no defects)
            return label_vector
        
        try:
            with open(ann_path, 'r', encoding='utf-8') as f:
                annotation = json.load(f)
            
            # Extract defect classes from annotation
            if "objects" in annotation:
                for obj in annotation["objects"]:
                    class_title = obj.get("classTitle", "")
                    if class_title in self.DEFECT_CLASS_TO_IDX:
                        class_idx = self.DEFECT_CLASS_TO_IDX[class_title]
                        label_vector[class_idx] = 1.0
                        
        except Exception as e:
            print(f"Warning: Error loading annotation {ann_path}: {e}")
            # Return all zeros on error
            return label_vector
        
        return label_vector
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, str]:
        """
        Get item from dataset.
        
        Args:
            idx: Index
            
        Returns:
            Tuple of (image_tensor, label_vector, image_name)
        """
        img_name = self.image_files[idx]
        img_path = os.path.join(self.img_dir, img_name)
        
        try:
            # Load image
            image = Image.open(img_path).convert("RGB")
            
            # Resize if target size specified
            if self.target_size is not None:
                image = image.resize((self.target_size[1], self.target_size[0]), Image.LANCZOS)
            
            # Apply transforms
            if self.transform:
                image = self.transform(image)
            else:
                # Default: convert to tensor
                image = transforms.ToTensor()(image)
            
            # Load labels
            labels = self._load_annotation(img_name)
            
            return image, labels, img_name
            
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
            # Return dummy data on error
            if self.target_size:
                dummy_image = torch.zeros(3, self.target_size[0], self.target_size[1])
            else:
                dummy_image = torch.zeros(3, 256, 256)  # Default size
            dummy_labels = torch.zeros(self.NUM_CLASSES, dtype=torch.float32)
            return dummy_image, dummy_labels, f"ERROR_{img_name}"


def create_train_val_test_split(dataset: SeverstalFullImageDataset, 
                                train_ratio: float = 0.7,
                                val_ratio: float = 0.2,
                                test_ratio: float = 0.1,
                                seed: int = 42) -> Tuple[Dataset, Dataset, Dataset]:
    """
    Split dataset into train/validation/test sets.
    
    Args:
        dataset: Full dataset
        train_ratio: Fraction for training
        val_ratio: Fraction for validation  
        test_ratio: Fraction for testing
        seed: Random seed
        
    Returns:
        Tuple of (train_dataset, val_dataset, test_dataset)
    """
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, "Ratios must sum to 1.0"
    
    # Set seed for reproducible splits
    torch.manual_seed(seed)
    
    # Get total length
    total_length = len(dataset)
    train_size = int(train_ratio * total_length)
    val_size = int(val_ratio * total_length)
    test_size = total_length - train_size - val_size
    
    # Random split
    train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size, test_size]
    )
    
    print(f"Dataset split: Train={len(train_dataset)}, Val={len(val_dataset)}, Test={len(test_dataset)}")
    
    return train_dataset, val_dataset, test_dataset


def get_class_distribution(dataset: SeverstalFullImageDataset) -> Dict[str, int]:
    """
    Analyze class distribution in dataset.
    
    Args:
        dataset: Dataset to analyze
        
    Returns:
        Dictionary with class counts
    """
    class_counts = {f"defect_{i+1}": 0 for i in range(4)}
    total_images = len(dataset)
    images_with_defects = 0
    
    print("Analyzing class distribution...")
    
    for i in range(total_images):
        _, labels, _ = dataset[i]
        
        if labels.sum() > 0:
            images_with_defects += 1
            
        for j, class_name in enumerate([f"defect_{i+1}" for i in range(4)]):
            if labels[j] > 0:
                class_counts[class_name] += 1
        
        if (i + 1) % 1000 == 0:
            print(f"  Processed {i+1}/{total_images} images...")
    
    class_counts["images_with_defects"] = images_with_defects
    class_counts["images_without_defects"] = total_images - images_with_defects
    
    return class_counts


if __name__ == "__main__":
    # Simple test
    print("Testing SeverstalFullImageDataset...")
    
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    img_dir = os.path.join(project_root, "data", "images")
    ann_dir = os.path.join(project_root, "data", "annotations")
    
    # Test transforms - use correct Severstal dimensions
    test_transforms = transforms.Compose([
        transforms.Resize((256, 1600)),  # Correct Severstal dimensions
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Create dataset with limited images for testing
    dataset = SeverstalFullImageDataset(
        img_dir=img_dir,
        ann_dir=ann_dir,
        transform=test_transforms,
        target_size=(256, 1600),  # Correct dimensions
        debug_limit=10
    )
    
    if len(dataset) > 0:
        # Test first item
        image, labels, name = dataset[0]
        print(f"First item: {name}")
        print(f"Image shape: {image.shape}")
        print(f"Labels: {labels.numpy()}")
        print(f"Has defects: {labels.sum().item() > 0}")
        
        print("Dataset test completed successfully!")
    else:
        print("Dataset is empty!")