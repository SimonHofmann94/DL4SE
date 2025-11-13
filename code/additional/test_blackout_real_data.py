"""
Test Defect Blackout Augmentation with REAL data from data/images and data/annotations.

Loads actual Severstal images and annotations, applies blackout augmentation,
and visualizes the results exactly like test_blackout_augmentation.py but with REAL data.

Run with: python code/test_blackout_real_data.py
"""

import os
import sys
import json
import base64
import zlib
import random
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from typing import Tuple, Optional, List
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from core.augmentation.defect_blackout import DefectBlackoutTransform

# Try to import scipy
try:
    from scipy import ndimage
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    logger.error("scipy not available. Install with: pip install scipy")


def create_full_mask_from_png_object(b64_data: str, origin: List[int], full_h: int, full_w: int) -> Optional[np.ndarray]:
    """
    Create full-size mask from base64 encoded PNG object.
    
    Args:
        b64_data: Base64 encoded bitmap data
        origin: [x, y] coordinates of the top-left corner
        full_h: Full image height
        full_w: Full image width
    
    Returns:
        Binary mask array (0s and 1s) of size (full_h, full_w)
    """
    try:
        # Decode the base64 bitmap
        decoded_bytes = base64.b64decode(b64_data)
        decompressed_bytes = zlib.decompress(decoded_bytes)
        
        # Convert to PIL Image
        from io import BytesIO
        mask_pil = Image.open(BytesIO(decompressed_bytes)).convert('L')
        mask_np = np.array(mask_pil)
        
        # Create full-size mask
        full_mask = np.zeros((full_h, full_w), dtype=np.uint8)
        
        # Get mask dimensions and origin
        mask_h, mask_w = mask_np.shape
        orig_x, orig_y = origin
        
        # Ensure we don't go out of bounds
        end_y = min(orig_y + mask_h, full_h)
        end_x = min(orig_x + mask_w, full_w)
        
        # Place the mask in the full image
        # Convert non-zero pixels to 1 (binary mask)
        mask_binary = (mask_np > 0).astype(np.uint8)
        full_mask[orig_y:end_y, orig_x:end_x] = mask_binary[:end_y-orig_y, :end_x-orig_x]
        
        return full_mask
        
    except Exception as e:
        logger.error(f"Error creating full mask from PNG object: {e}")
        return None


def load_real_image_and_mask(img_dir: str, ann_dir: str, img_name: str) -> Tuple[Optional[Image.Image], Optional[np.ndarray]]:
    """
    Load a real image and create its ground-truth mask from annotation.
    
    Returns:
        Tuple of (PIL Image, mask array with class indices 0-4)
    """
    DEFECT_CLASS_TO_IDX = {
        "defect_1": 1,
        "defect_2": 2,
        "defect_3": 3,
        "defect_4": 4,
    }
    
    # Load image
    img_path = os.path.join(img_dir, img_name)
    if not os.path.exists(img_path):
        logger.error(f"Image not found: {img_path}")
        return None, None
    
    img = Image.open(img_path).convert("RGB")
    img_width, img_height = img.size
    
    # Load annotation
    base_name_no_ext = os.path.splitext(img_name)[0]
    ann_path_variants = [
        os.path.join(ann_dir, f"{img_name}.json"),
        os.path.join(ann_dir, f"{base_name_no_ext}.json")
    ]
    
    annotation = None
    for ann_path in ann_path_variants:
        if os.path.exists(ann_path):
            try:
                with open(ann_path, 'r') as f:
                    annotation = json.load(f)
                break
            except Exception as e:
                logger.warning(f"Error loading {ann_path}: {e}")
    
    if not annotation:
        logger.warning(f"No annotation found for {img_name}")
        # Return image with empty mask (all no_defect)
        mask = np.zeros((img_height, img_width), dtype=np.uint8)
        return img, mask
    
    # Create combined mask
    combined_mask = np.zeros((img_height, img_width), dtype=np.uint8)
    
    if "objects" in annotation and annotation["objects"]:
        for obj in annotation["objects"]:
            class_title = obj.get("classTitle")
            
            if class_title not in DEFECT_CLASS_TO_IDX:
                continue
            
            defect_idx = DEFECT_CLASS_TO_IDX[class_title]
            
            # Extract bitmap
            if "bitmap" not in obj:
                continue
            
            bitmap_data = obj["bitmap"]
            b64_data = bitmap_data.get("data")
            origin = bitmap_data.get("origin")
            
            if not b64_data or not origin:
                continue
            
            # Create mask for this object
            single_obj_mask = create_full_mask_from_png_object(
                b64_data, origin, img_height, img_width
            )
            
            if single_obj_mask is not None:
                # Add to combined mask with class index
                combined_mask[single_obj_mask == 1] = defect_idx
    
    return img, combined_mask


def visualize_blackout_result(
    original_img: Image.Image,
    original_mask: np.ndarray,
    augmented_img: Image.Image,
    augmented_mask: np.ndarray,
    title: str,
    save_path: str = None
):
    """Visualize blackout augmentation results with clear before/after comparison - SAME AS test_blackout_augmentation.py"""
    fig, axes = plt.subplots(2, 3, figsize=(20, 8))
    
    # Row 1: ORIGINAL (Before Blackout)
    # Original image (no overlay)
    axes[0, 0].imshow(original_img)
    axes[0, 0].set_title("ORIGINAL IMAGE\n(Before Blackout)", fontweight='bold')
    axes[0, 0].axis('off')
    
    # Original image WITH mask overlay
    axes[0, 1].imshow(original_img)
    # Create colored overlay for defects
    overlay = np.zeros((*original_mask.shape, 4))
    colors_rgba = {
        1: [1.0, 0.0, 0.0, 0.4],  # Red for defect_1
        2: [0.0, 1.0, 0.0, 0.4],  # Green for defect_2
        3: [0.0, 0.0, 1.0, 0.4],  # Blue for defect_3
        4: [1.0, 1.0, 0.0, 0.4]   # Yellow for defect_4
    }
    for defect_idx, color in colors_rgba.items():
        overlay[original_mask == defect_idx] = color
    axes[0, 1].imshow(overlay)
    axes[0, 1].set_title("ORIGINAL + Mask Overlay\n(Defects highlighted)", fontweight='bold')
    axes[0, 1].axis('off')
    
    # Original mask only
    mask_display = axes[0, 2].imshow(original_mask, cmap='tab10', vmin=0, vmax=4)
    axes[0, 2].set_title("ORIGINAL MASK\n(0=no_defect, 1-4=defects)", fontweight='bold')
    axes[0, 2].axis('off')
    cbar1 = plt.colorbar(mask_display, ax=axes[0, 2], fraction=0.046, pad=0.04)
    cbar1.set_label('Defect Class', rotation=270, labelpad=15)
    
    # Row 2: AUGMENTED (After Blackout)
    # Augmented image (no overlay)
    axes[1, 0].imshow(augmented_img)
    axes[1, 0].set_title("AUGMENTED IMAGE\n(After Blackout - Black regions)", fontweight='bold', color='red')
    axes[1, 0].axis('off')
    
    # Augmented image WITH mask overlay
    axes[1, 1].imshow(augmented_img)
    overlay_aug = np.zeros((*augmented_mask.shape, 4))
    for defect_idx, color in colors_rgba.items():
        overlay_aug[augmented_mask == defect_idx] = color
    axes[1, 1].imshow(overlay_aug)
    axes[1, 1].set_title("AUGMENTED + Mask Overlay\n(Remaining defects)", fontweight='bold', color='red')
    axes[1, 1].axis('off')
    
    # Augmented mask only
    mask_display2 = axes[1, 2].imshow(augmented_mask, cmap='tab10', vmin=0, vmax=4)
    axes[1, 2].set_title("AUGMENTED MASK\n(Defects removed from mask)", fontweight='bold', color='red')
    axes[1, 2].axis('off')
    cbar2 = plt.colorbar(mask_display2, ax=axes[1, 2], fraction=0.046, pad=0.04)
    cbar2.set_label('Defect Class', rotation=270, labelpad=15)
    
    # Add detailed statistics
    orig_stats = {i: np.sum(original_mask == i) for i in range(5)}
    aug_stats = {i: np.sum(augmented_mask == i) for i in range(5)}
    
    stats_text = "PIXEL COUNT STATISTICS:\n"
    stats_text += "=" * 50 + "\n"
    stats_text += "Class |  Original | Augmented |   Change   | %Change\n"
    stats_text += "-" * 50 + "\n"
    
    class_names = ["no_defect", "defect_1", "defect_2", "defect_3", "defect_4"]
    for i in range(5):
        diff = aug_stats[i] - orig_stats[i]
        pct_change = (diff / orig_stats[i] * 100) if orig_stats[i] > 0 else 0
        stats_text += f"{class_names[i]:9} | {orig_stats[i]:9,} | {aug_stats[i]:9,} | {diff:+10,} | {pct_change:+6.1f}%\n"
    
    # Count blacked out instances
    orig_defects = np.sum(original_mask > 0)
    aug_defects = np.sum(augmented_mask > 0)
    removed = orig_defects - aug_defects
    
    stats_text += "-" * 50 + "\n"
    stats_text += f"Total defect pixels removed: {removed:,} ({removed/orig_defects*100:.1f}%)\n" if orig_defects > 0 else ""
    
    fig.text(0.02, 0.02, stats_text, fontfamily='monospace', fontsize=9, 
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.suptitle(title, fontsize=16, fontweight='bold', y=0.98)
    plt.tight_layout(rect=[0, 0.08, 1, 0.96])
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        logger.info(f"✅ Saved visualization to: {save_path}")
    
    plt.show()
    plt.close()


def test_with_real_data(img_dir: str, ann_dir: str, num_samples: int = 4):
    """Test blackout augmentation with real Severstal images."""
    
    if not SCIPY_AVAILABLE:
        logger.error("scipy not available - cannot run tests")
        return
    
    # Find images with defects
    available_images = sorted([
        f for f in os.listdir(img_dir)
        if f.lower().endswith(('.jpg', '.jpeg', '.png'))
    ])
    
    logger.info(f"Found {len(available_images)} images with annotations")
    
    # Try to find images WITH defect_2 for better demonstration
    images_with_defect_2 = []
    images_with_any_defect = []
    
    logger.info("Scanning for images with defect_2...")
    for img_name in available_images:
        base_name = os.path.splitext(img_name)[0]
        ann_path = os.path.join(ann_dir, f"{base_name}.json")
        
        if not os.path.exists(ann_path):
            ann_path = os.path.join(ann_dir, f"{img_name}.json")
        
        if not os.path.exists(ann_path):
            continue
        
        try:
            with open(ann_path, 'r') as f:
                annotation = json.load(f)
            
            has_defect_2 = False
            has_any_defect = False
            
            if "objects" in annotation and annotation["objects"]:
                for obj in annotation["objects"]:
                    class_title = obj.get("classTitle")
                    if class_title == "defect_2":
                        has_defect_2 = True
                    if class_title in ["defect_1", "defect_2", "defect_3", "defect_4"]:
                        has_any_defect = True
            
            if has_defect_2:
                images_with_defect_2.append(img_name)
            elif has_any_defect:
                images_with_any_defect.append(img_name)
        except:
            continue
    
    # Select best samples based on availability
    if len(images_with_defect_2) >= num_samples:
        selected_images = random.sample(images_with_defect_2, num_samples)
        target_defects = [2]  # Only blackout defect_2
        logger.info(f"✓ Found {len(images_with_defect_2)} images WITH defect_2")
        logger.info(f"  Using {num_samples} images with defect_2 for testing")
        logger.info(f"  Target: Only defect_2 will be blacked out")
    elif len(images_with_defect_2) > 0:
        # Mix: some with defect_2, some with other defects
        num_d2 = len(images_with_defect_2)
        num_other = min(num_samples - num_d2, len(images_with_any_defect))
        selected_images = images_with_defect_2 + random.sample(images_with_any_defect, num_other)
        target_defects = [1, 2, 3, 4]  # Blackout all defects for mixed samples
        logger.info(f"⚠ Only {num_d2} images with defect_2 found")
        logger.info(f"  Using mixed samples: {num_d2} with defect_2 + {num_other} with other defects")
        logger.info(f"  Target: ALL defect types [1,2,3,4] will be blacked out")
    else:
        # No defect_2 found
        selected_images = random.sample(images_with_any_defect, min(num_samples, len(images_with_any_defect)))
        target_defects = [1, 2, 3, 4]  # Blackout all defects
        logger.warning(f"✗ NO images with defect_2 found in dataset!")
        logger.info(f"  Using {len(selected_images)} images with OTHER defect types")
        logger.info(f"  Target: ALL defect types [1,2,3,4] will be blacked out")
    
    logger.info(f"\n{'='*80}")
    logger.info(f"Testing Blackout Augmentation with {len(selected_images)} REAL images")
    logger.info(f"Target defects for blackout: {target_defects}")
    logger.info(f"{'='*80}\n")
    
    for sample_num, img_name in enumerate(selected_images, 1):
        logger.info(f"\nTest Sample {sample_num}/{num_samples}: {img_name}")
        logger.info(f"{'-'*60}")
        
        # Load real image and mask
        img, mask = load_real_image_and_mask(img_dir, ann_dir, img_name)
        
        if img is None or mask is None:
            logger.warning(f"Skipping {img_name} - could not load")
            continue
        
        # Check if image has defects
        defect_pixels = np.sum(mask > 0)
        if defect_pixels == 0:
            logger.info(f"Skipping {img_name} - no defects found (clean image)")
            continue
        
        # Log defect statistics
        for defect_idx in range(1, 5):
            pixels = np.sum(mask == defect_idx)
            if pixels > 0:
                logger.info(f"  defect_{defect_idx}: {pixels:,} pixels")
        
        # Create blackout transform with dynamic target defects
        blackout = DefectBlackoutTransform(
            enabled=True,
            instance_blackout_prob=0.8,  # 80% chance per instance
            defect_indices_to_blackout=target_defects,  # Dynamically determined defects
            min_pixels_to_blackout=20,
            verbose=True
        )
        
        # Apply blackout
        img_aug, mask_aug, modified = blackout(img, mask)
        
        logger.info(f"  Blackout applied: {modified}")
        
        # Visualize
        defect_str = f"defect_{target_defects[0]}" if len(target_defects) == 1 else f"defects {target_defects}"
        visualize_blackout_result(
            img, mask, img_aug, mask_aug,
            f"REAL DATA Sample {sample_num}: {img_name}\nSelective Blackout ({defect_str}, 80% probability)",
            f"real_data_test_{sample_num}_{img_name.replace('.jpg', '.png')}"
        )
    
    logger.info(f"\n{'='*80}")
    logger.info("✅ Real data testing complete!")
    logger.info(f"{'='*80}\n")


def main():
    """Main function."""
    print("\n" + "="*80)
    print("DEFECT BLACKOUT AUGMENTATION - REAL DATA TEST")
    print("="*80 + "\n")
    
    # Paths
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    img_dir = os.path.join(project_root, "data", "images")
    ann_dir = os.path.join(project_root, "data", "annotations")
    
    # Check if directories exist
    if not os.path.exists(img_dir):
        logger.error(f"Image directory not found: {img_dir}")
        return
    
    if not os.path.exists(ann_dir):
        logger.error(f"Annotation directory not found: {ann_dir}")
        return
    
    # Run tests with real data
    test_with_real_data(img_dir, ann_dir, num_samples=4)
    
    print("\n✅ Testing complete! Check the generated visualization images.")



if __name__ == "__main__":
    main()
