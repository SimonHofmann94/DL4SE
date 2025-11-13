"""
Visual testing and debugging for DefectAwareCutMix augmentation.

This script:
1. Loads real images from the Severstal dataset
2. Applies CutMix with defect-aware constraints
3. Visualizes the results (original images, mixed image, masks, labels)
4. Saves output images for inspection

Usage:
    python code/additional/test_cutmix_visualization.py
"""

import os
import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import json
import cv2
import random

# Add code directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.augmentation import DefectAwareCutMix


def load_image_and_annotation(img_dir, ann_dir, image_name):
    """
    Load image and its annotation.
    
    Args:
        img_dir: Path to images directory
        ann_dir: Path to annotations directory
        image_name: Name of image file
    
    Returns:
        Tuple of (image, label, bboxes)
    """
    # Load image
    img_path = os.path.join(img_dir, image_name)
    image = cv2.imread(img_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Load annotation
    ann_name = image_name + ".json"
    ann_path = os.path.join(ann_dir, ann_name)
    
    with open(ann_path, 'r') as f:
        annotation = json.load(f)
    
    # Extract label (multi-label binary vector)
    # Classes: [no_defect, defect_1, defect_2, defect_3, defect_4]
    label = np.zeros(5, dtype=np.float32)
    
    # Map class titles to indices
    class_map = {
        'no_defect': 0,
        'defect_1': 1,
        'defect_2': 2,
        'defect_3': 3,
        'defect_4': 4
    }
    
    # Check if image has defects (using 'objects' field from Supervisely format)
    if annotation.get('objects') and len(annotation['objects']) > 0:
        for obj in annotation['objects']:
            class_title = obj.get('classTitle', '')
            if class_title in class_map:
                category_id = class_map[class_title]
                label[category_id] = 1.0
    else:
        # No defects = class 0
        label[0] = 1.0
    
    # Extract bounding boxes from bitmap objects
    bboxes = []
    if annotation.get('objects'):
        for obj in annotation['objects']:
            class_title = obj.get('classTitle', '')
            if class_title in class_map and 'bitmap' in obj:
                # Get bitmap origin and data to estimate bbox
                bitmap = obj['bitmap']
                origin = bitmap.get('origin', [0, 0])  # [x, y]
                
                # For simplicity, we'll create a small bbox around the origin
                # In a real implementation, you'd decode the bitmap to get exact bounds
                # For now, use a fixed size as approximation
                x, y = origin
                w, h = 50, 30  # Approximate defect size
                
                category_id = class_map[class_title]
                bboxes.append({
                    'bbox': [x, y, w, h],  # [x, y, width, height]
                    'category_id': category_id
                })
    
    return image, label, bboxes


def visualize_cutmix_result(
    img1, label1, bbox1,
    img2, label2, bbox2,
    mixed_img, mixed_label,
    cut_bbox,
    save_path=None
):
    """
    Visualize CutMix result with original images and mixed output.
    
    Args:
        img1, label1, bbox1: Source image data
        img2, label2, bbox2: Target image data
        mixed_img, mixed_label: Result after CutMix
        cut_bbox: The cut region (x1, y1, x2, y2) or None if no cut
        save_path: Path to save visualization (optional)
    """
    class_names = ['no_defect', 'defect_1', 'defect_2', 'defect_3', 'defect_4']
    
    # Create figure with 3 subplots
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))
    
    # --- Subplot 1: Source Image (img1) ---
    ax1 = axes[0]
    ax1.imshow(img1)
    ax1.set_title(f'Source Image 1\nLabel: {[class_names[i] for i, v in enumerate(label1) if v > 0]}', 
                  fontsize=12, fontweight='bold')
    
    # Draw defect bboxes on img1
    for bbox_dict in bbox1:
        bbox = bbox_dict['bbox']
        category_id = bbox_dict['category_id']
        x, y, w, h = bbox
        rect = patches.Rectangle(
            (x, y), w, h,
            linewidth=2, edgecolor='red', facecolor='none'
        )
        ax1.add_patch(rect)
        ax1.text(x, y - 5, f'defect_{category_id}', 
                color='red', fontsize=10, fontweight='bold',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))
    
    ax1.axis('off')
    
    # --- Subplot 2: Target Image (img2) ---
    ax2 = axes[1]
    ax2.imshow(img2)
    ax2.set_title(f'Source Image 2\nLabel: {[class_names[i] for i, v in enumerate(label2) if v > 0]}',
                  fontsize=12, fontweight='bold')
    
    # Draw defect bboxes on img2
    for bbox_dict in bbox2:
        bbox = bbox_dict['bbox']
        category_id = bbox_dict['category_id']
        x, y, w, h = bbox
        rect = patches.Rectangle(
            (x, y), w, h,
            linewidth=2, edgecolor='blue', facecolor='none'
        )
        ax2.add_patch(rect)
        ax2.text(x, y - 5, f'defect_{category_id}',
                color='blue', fontsize=10, fontweight='bold',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))
    
    # Draw cut region on img2 (if exists)
    if cut_bbox is not None:
        x1, y1, x2, y2 = cut_bbox
        rect = patches.Rectangle(
            (x1, y1), x2 - x1, y2 - y1,
            linewidth=3, edgecolor='green', facecolor='green', alpha=0.3
        )
        ax2.add_patch(rect)
        ax2.text(x1, y1 - 10, 'CUT REGION',
                color='green', fontsize=12, fontweight='bold',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))
    
    ax2.axis('off')
    
    # --- Subplot 3: Mixed Image ---
    ax3 = axes[2]
    ax3.imshow(mixed_img)
    ax3.set_title(f'Mixed Image (CutMix Result)\nLabel: {[class_names[i] for i, v in enumerate(mixed_label) if v > 0]}',
                  fontsize=12, fontweight='bold', color='green')
    
    # Draw all defect bboxes on mixed image
    # From img1 (red)
    for bbox_dict in bbox1:
        bbox = bbox_dict['bbox']
        category_id = bbox_dict['category_id']
        x, y, w, h = bbox
        rect = patches.Rectangle(
            (x, y), w, h,
            linewidth=2, edgecolor='red', facecolor='none', linestyle='--'
        )
        ax3.add_patch(rect)
    
    # From img2 (blue) - only if in cut region
    if cut_bbox is not None:
        x1_cut, y1_cut, x2_cut, y2_cut = cut_bbox
        for bbox_dict in bbox2:
            bbox = bbox_dict['bbox']
            x, y, w, h = bbox
            # Check if bbox is in cut region
            if x >= x1_cut and y >= y1_cut and x + w <= x2_cut and y + h <= y2_cut:
                rect = patches.Rectangle(
                    (x, y), w, h,
                    linewidth=2, edgecolor='blue', facecolor='none', linestyle='--'
                )
                ax3.add_patch(rect)
    
    # Draw cut region outline
    if cut_bbox is not None:
        x1, y1, x2, y2 = cut_bbox
        rect = patches.Rectangle(
            (x1, y1), x2 - x1, y2 - y1,
            linewidth=3, edgecolor='lime', facecolor='none', linestyle='-'
        )
        ax3.add_patch(rect)
    
    ax3.axis('off')
    
    # Add legend
    legend_elements = [
        patches.Patch(facecolor='red', edgecolor='red', label='Defects from Image 1'),
        patches.Patch(facecolor='blue', edgecolor='blue', label='Defects from Image 2'),
        patches.Patch(facecolor='lime', edgecolor='lime', label='Cut Region')
    ]
    fig.legend(handles=legend_elements, loc='upper center', ncol=3, fontsize=11)
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.92)
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved visualization to: {save_path}")
    
    plt.show()


def test_cutmix_with_real_data(
    img_dir,
    ann_dir,
    num_examples=5,
    output_dir=None,
    cutmix_config=None
):
    """
    Test CutMix with real dataset images.
    
    Args:
        img_dir: Path to images directory
        ann_dir: Path to annotations directory
        num_examples: Number of examples to generate
        output_dir: Directory to save visualizations
        cutmix_config: CutMix configuration dict
    """
    print("="*80)
    print("TESTING DEFECT-AWARE CUTMIX WITH REAL DATA")
    print("="*80)
    
    # Get all image files
    all_images = [
        f for f in os.listdir(img_dir)
        if f.lower().endswith(('.jpg', '.jpeg', '.png'))
    ]
    
    print(f"\nFound {len(all_images)} images in dataset")
    
    # Filter images with defects for better visualization
    images_with_defects = []
    for img_name in all_images[:100]:  # Check first 100 for speed
        ann_path = os.path.join(ann_dir, img_name + ".json")
        if os.path.exists(ann_path):
            with open(ann_path, 'r') as f:
                ann = json.load(f)
                # Check for objects (Supervisely format)
                if ann.get('objects') and len(ann['objects']) > 0:
                    images_with_defects.append(img_name)
    
    print(f"Found {len(images_with_defects)} images with defects")
    
    # Create CutMix augmentation
    if cutmix_config is None:
        cutmix_config = {
            'prob': 1.0,  # Always apply for testing
            'alpha': 1.0,
            'min_cut_ratio': 0.1,
            'max_cut_ratio': 0.3,
            'max_attempts': 10,
            'verbose': True
        }
    
    cutmix = DefectAwareCutMix(**cutmix_config)
    
    print(f"\nCutMix config:")
    for key, val in cutmix.get_config().items():
        print(f"  {key}: {val}")
    
    # Create output directory
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        print(f"\nSaving visualizations to: {output_dir}")
    
    # Generate examples
    print(f"\n{'='*80}")
    print(f"GENERATING {num_examples} CUTMIX EXAMPLES")
    print(f"{'='*80}\n")
    
    successful = 0
    failed = 0
    
    for i in range(num_examples):
        print(f"\n--- Example {i+1}/{num_examples} ---")
        
        # Randomly select two images with defects
        img1_name = random.choice(images_with_defects)
        img2_name = random.choice(images_with_defects)
        
        # Ensure different images
        while img1_name == img2_name:
            img2_name = random.choice(images_with_defects)
        
        print(f"Image 1: {img1_name}")
        print(f"Image 2: {img2_name}")
        
        # Load data
        img1, label1, bbox1 = load_image_and_annotation(img_dir, ann_dir, img1_name)
        img2, label2, bbox2 = load_image_and_annotation(img_dir, ann_dir, img2_name)
        
        print(f"Image 1 label: {label1} ({len(bbox1)} defects)")
        print(f"Image 2 label: {label2} ({len(bbox2)} defects)")
        
        # Apply CutMix
        mixed_img, mixed_label = cutmix(img1, label1, bbox1, img2, label2, bbox2)
        
        # Get the cut bbox from the last operation
        cut_bbox = cutmix.last_cut_bbox
        
        # Check if CutMix was applied
        if cut_bbox is None:
            print("⚠️  CutMix NOT applied (no valid cut region found)")
            failed += 1
        else:
            print(f"✓ CutMix applied successfully! Cut region: {cut_bbox}")
            successful += 1
        
        print(f"Mixed label: {mixed_label}")
        
        # Visualize
        save_path = None
        if output_dir:
            save_path = os.path.join(output_dir, f"cutmix_example_{i+1}.png")
        
        visualize_cutmix_result(
            img1, label1, bbox1,
            img2, label2, bbox2,
            mixed_img, mixed_label,
            cut_bbox,
            save_path=save_path
        )
    
    # Summary
    print(f"\n{'='*80}")
    print(f"SUMMARY")
    print(f"{'='*80}")
    print(f"Total examples: {num_examples}")
    print(f"Successful CutMix: {successful}")
    print(f"Failed (no valid region): {failed}")
    print(f"Success rate: {successful/num_examples*100:.1f}%")
    
    if failed > 0:
        print(f"\n⚠️  Some CutMix operations failed.")
        print(f"This is normal if images have many/large defects.")
        print(f"Consider increasing max_cut_ratio or max_attempts.")


def main():
    """Main test function."""
    # Setup paths
    project_root = Path(__file__).parent.parent.parent
    img_dir = project_root / "data" / "images"
    ann_dir = project_root / "data" / "annotations"
    output_dir = project_root / "data" / "reports" / "cutmix_test"
    
    print(f"Project root: {project_root}")
    print(f"Images directory: {img_dir}")
    print(f"Annotations directory: {ann_dir}")
    print(f"Output directory: {output_dir}")
    
    # Check if directories exist
    if not img_dir.exists():
        print(f"\n❌ Error: Images directory not found: {img_dir}")
        return
    
    if not ann_dir.exists():
        print(f"\n❌ Error: Annotations directory not found: {ann_dir}")
        return
    
    # Test with different configurations
    configs_to_test = [
        {
            'name': 'Conservative (small cuts)',
            'config': {
                'prob': 1.0,
                'alpha': 1.0,
                'min_cut_ratio': 0.05,
                'max_cut_ratio': 0.2,
                'max_attempts': 10,
                'verbose': True
            }
        },
        {
            'name': 'Moderate (medium cuts)',
            'config': {
                'prob': 1.0,
                'alpha': 1.0,
                'min_cut_ratio': 0.1,
                'max_cut_ratio': 0.3,
                'max_attempts': 15,
                'verbose': True
            }
        },
        {
            'name': 'Aggressive (large cuts)',
            'config': {
                'prob': 1.0,
                'alpha': 1.0,
                'min_cut_ratio': 0.2,
                'max_cut_ratio': 0.4,
                'max_attempts': 20,
                'verbose': True
            }
        }
    ]
    
    # Ask user which config to test
    print("\n" + "="*80)
    print("CUTMIX CONFIGURATION OPTIONS")
    print("="*80)
    for i, cfg in enumerate(configs_to_test):
        print(f"{i+1}. {cfg['name']}")
    
    choice = input(f"\nSelect configuration (1-{len(configs_to_test)}) or press Enter for default (2): ")
    
    if choice.strip() == "":
        choice = 2
    else:
        choice = int(choice)
    
    selected_config = configs_to_test[choice - 1]
    
    print(f"\nSelected: {selected_config['name']}")
    
    # Ask number of examples
    num_examples = input("\nNumber of examples to generate (default: 5): ")
    num_examples = int(num_examples) if num_examples.strip() else 5
    
    # Run test
    test_cutmix_with_real_data(
        img_dir=str(img_dir),
        ann_dir=str(ann_dir),
        num_examples=num_examples,
        output_dir=str(output_dir),
        cutmix_config=selected_config['config']
    )
    
    print(f"\n{'='*80}")
    print(f"TESTING COMPLETE!")
    print(f"{'='*80}")
    print(f"\nVisualizations saved to: {output_dir}")
    print(f"\nYou can now:")
    print(f"1. Check the visualizations to see if CutMix preserves defects")
    print(f"2. Verify that cut regions don't overlap with defect bounding boxes")
    print(f"3. Confirm that labels are correctly combined (union)")


if __name__ == "__main__":
    main()
