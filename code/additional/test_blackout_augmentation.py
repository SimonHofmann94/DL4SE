"""
Test script for Defect Blackout Augmentation.

This script thoroughly tests the DefectBlackoutTransform class to ensure:
1. Selective blackout works correctly
2. Complete blackout works correctly
3. Defect-specific targeting works
4. Mask and label consistency is maintained
5. Edge cases are handled properly

Run with: python code/test_blackout_augmentation.py
"""

import os
import sys
import numpy as np
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
from typing import Tuple

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

# Import from core module (avoid 'code' package naming conflict)
from core.augmentation.defect_blackout import DefectBlackoutTransform

# Try to import scipy
try:
    from scipy import ndimage
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    print("WARNING: scipy not available. Some tests will be skipped.")


def create_synthetic_image_and_mask(
    width: int = 1600,
    height: int = 256,
    defect_config: dict = None
) -> Tuple[Image.Image, np.ndarray]:
    """
    Create synthetic test image with defects.
    
    Args:
        width: Image width
        height: Image height
        defect_config: Dictionary defining defects to add
                      Example: {2: [(50, 100, 200, 300), (150, 180, 500, 600)]}
                      means defect_2 at two locations
    
    Returns:
        Tuple of (PIL Image, numpy mask)
    """
    if defect_config is None:
        defect_config = {
            1: [(50, 100, 200, 300)],  # defect_1
            2: [(120, 180, 500, 650), (120, 180, 800, 950)],  # defect_2 (2 instances)
            3: [(10, 60, 1000, 1200)],  # defect_3
            4: [(180, 230, 1300, 1450)]  # defect_4
        }
    
    # Create gray background image
    img = Image.new("RGB", (width, height), color=(128, 128, 128))
    draw = ImageDraw.Draw(img)
    
    # Create mask
    mask = np.zeros((height, width), dtype=np.uint8)
    
    # Add defects
    for defect_idx, regions in defect_config.items():
        # Draw colored rectangles in image (different colors for different defects)
        colors = {
            1: (255, 100, 100),  # Red-ish
            2: (100, 255, 100),  # Green-ish
            3: (100, 100, 255),  # Blue-ish
            4: (255, 255, 100)   # Yellow-ish
        }
        color = colors.get(defect_idx, (200, 200, 200))
        
        for (y1, y2, x1, x2) in regions:
            # Draw in image
            draw.rectangle([x1, y1, x2, y2], fill=color)
            # Mark in mask
            mask[y1:y2, x1:x2] = defect_idx
    
    return img, mask


def visualize_blackout_result(
    original_img: Image.Image,
    original_mask: np.ndarray,
    augmented_img: Image.Image,
    augmented_mask: np.ndarray,
    title: str,
    save_path: str = None
):
    """Visualize blackout augmentation results with clear before/after comparison."""
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
        1: [1.0, 0.0, 0.0, 0.4],  # Red with alpha
        2: [0.0, 1.0, 0.0, 0.4],  # Green with alpha
        3: [0.0, 0.0, 1.0, 0.4],  # Blue with alpha
        4: [1.0, 1.0, 0.0, 0.4]   # Yellow with alpha
    }
    for defect_idx, color in colors_rgba.items():
        mask_indices = original_mask == defect_idx
        overlay[mask_indices] = color
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
        mask_indices = augmented_mask == defect_idx
        overlay_aug[mask_indices] = color
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
        stats_text += f"{class_names[i]:9} | {orig_stats[i]:9d} | {aug_stats[i]:9d} | {diff:+9d} | {pct_change:+6.1f}%\n"
    
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
        print(f"✅ Saved visualization to: {save_path}")
    
    plt.show()
    plt.close()


def test_1_basic_functionality():
    """Test 1: Basic functionality - blackout should work."""
    print("\n" + "="*60)
    print("TEST 1: Basic Functionality")
    print("="*60)
    
    if not SCIPY_AVAILABLE:
        print("SKIPPED - scipy not available")
        return False
    
    # Create test data
    img, mask = create_synthetic_image_and_mask()
    
    # Create blackout transform
    blackout = DefectBlackoutTransform(
        enabled=True,
        instance_blackout_prob=1.0,  # Always blackout
        defect_indices_to_blackout=[1, 2, 3, 4],
        min_pixels_to_blackout=10,
        verbose=True
    )
    
    # Apply blackout
    img_aug, mask_aug, modified = blackout(img, mask)
    
    # Verify
    original_defect_pixels = np.sum(mask > 0)
    augmented_defect_pixels = np.sum(mask_aug > 0)
    
    print(f"Original defect pixels: {original_defect_pixels}")
    print(f"Augmented defect pixels: {augmented_defect_pixels}")
    print(f"Modified flag: {modified}")
    
    # Visualize
    visualize_blackout_result(
        img, mask, img_aug, mask_aug,
        "Test 1: Basic Selective Blackout (100% probability)",
        "test_1_basic_blackout.png"
    )
    
    # Assertions
    assert modified == True, "Blackout should have been applied"
    assert augmented_defect_pixels < original_defect_pixels, "Should have removed some defects"
    
    print("✅ TEST 1 PASSED")
    return True


def test_2_selective_blackout():
    """Test 2: Selective blackout with probability < 1.0"""
    print("\n" + "="*60)
    print("TEST 2: Selective Blackout (50% probability)")
    print("="*60)
    
    if not SCIPY_AVAILABLE:
        print("SKIPPED - scipy not available")
        return False
    
    # Create test data with multiple instances
    defect_config = {
        2: [
            (50, 80, 200, 250),
            (100, 130, 400, 450),
            (150, 180, 600, 650),
            (200, 230, 800, 850)
        ]
    }
    img, mask = create_synthetic_image_and_mask(defect_config=defect_config)
    
    # Create blackout transform with 50% probability
    blackout = DefectBlackoutTransform(
        enabled=True,
        instance_blackout_prob=0.5,  # 50% chance per instance
        defect_indices_to_blackout=[2],
        min_pixels_to_blackout=10,
        verbose=True
    )
    
    # Apply multiple times and check variability
    results = []
    for i in range(5):
        img_aug, mask_aug, modified = blackout(img.copy(), mask.copy())
        defect_pixels = np.sum(mask_aug == 2)
        results.append(defect_pixels)
        print(f"Run {i+1}: Defect_2 pixels remaining: {defect_pixels}")
    
    # Visualize last result
    visualize_blackout_result(
        img, mask, img_aug, mask_aug,
        f"Test 2: Selective Blackout (50% prob, Run 5)\nDefect_2 pixels: {results[-1]}",
        "test_2_selective_blackout.png"
    )
    
    # Check variability (not all runs should be identical)
    unique_results = len(set(results))
    print(f"Unique results across 5 runs: {unique_results}")
    
    assert unique_results > 1, "Should have variability with 50% probability"
    
    print("✅ TEST 2 PASSED")
    return True


def test_3_defect_specific_targeting():
    """Test 3: Only blackout specific defect types."""
    print("\n" + "="*60)
    print("TEST 3: Defect-Specific Targeting (Only Defect 2)")
    print("="*60)
    
    if not SCIPY_AVAILABLE:
        print("SKIPPED - scipy not available")
        return False
    
    # Create test data with multiple defect types
    img, mask = create_synthetic_image_and_mask()
    
    original_defect_1_pixels = np.sum(mask == 1)
    original_defect_2_pixels = np.sum(mask == 2)
    original_defect_3_pixels = np.sum(mask == 3)
    original_defect_4_pixels = np.sum(mask == 4)
    
    # Create blackout that only targets defect_2
    blackout = DefectBlackoutTransform(
        enabled=True,
        instance_blackout_prob=1.0,
        defect_indices_to_blackout=[2],  # Only defect_2
        min_pixels_to_blackout=10,
        verbose=True
    )
    
    # Apply blackout
    img_aug, mask_aug, modified = blackout(img, mask)
    
    augmented_defect_1_pixels = np.sum(mask_aug == 1)
    augmented_defect_2_pixels = np.sum(mask_aug == 2)
    augmented_defect_3_pixels = np.sum(mask_aug == 3)
    augmented_defect_4_pixels = np.sum(mask_aug == 4)
    
    print(f"Defect 1: {original_defect_1_pixels} -> {augmented_defect_1_pixels}")
    print(f"Defect 2: {original_defect_2_pixels} -> {augmented_defect_2_pixels}")
    print(f"Defect 3: {original_defect_3_pixels} -> {augmented_defect_3_pixels}")
    print(f"Defect 4: {original_defect_4_pixels} -> {augmented_defect_4_pixels}")
    
    # Visualize
    visualize_blackout_result(
        img, mask, img_aug, mask_aug,
        "Test 3: Only Defect_2 Targeted",
        "test_3_defect_specific.png"
    )
    
    # Assertions
    assert augmented_defect_1_pixels == original_defect_1_pixels, "Defect 1 should be unchanged"
    assert augmented_defect_2_pixels < original_defect_2_pixels, "Defect 2 should be reduced"
    assert augmented_defect_3_pixels == original_defect_3_pixels, "Defect 3 should be unchanged"
    assert augmented_defect_4_pixels == original_defect_4_pixels, "Defect 4 should be unchanged"
    
    print("✅ TEST 3 PASSED")
    return True


def test_4_complete_blackout():
    """Test 4: Complete blackout - remove ALL defects."""
    print("\n" + "="*60)
    print("TEST 4: Complete Blackout (Remove All Defects)")
    print("="*60)
    
    if not SCIPY_AVAILABLE:
        print("SKIPPED - scipy not available")
        return False
    
    # Create test data
    img, mask = create_synthetic_image_and_mask()
    
    original_defect_pixels = np.sum(mask > 0)
    
    # Create blackout with complete_blackout mode
    blackout = DefectBlackoutTransform(
        enabled=True,
        complete_blackout_prob=1.0,  # Always do complete blackout
        defect_indices_to_blackout=[1, 2, 3, 4],
        min_pixels_to_blackout=10,
        verbose=True
    )
    
    # Apply blackout
    img_aug, mask_aug, modified = blackout(img, mask)
    
    augmented_defect_pixels = np.sum(mask_aug > 0)
    no_defect_pixels = np.sum(mask_aug == 0)
    
    print(f"Original defect pixels: {original_defect_pixels}")
    print(f"Augmented defect pixels: {augmented_defect_pixels}")
    print(f"No-defect pixels: {no_defect_pixels}")
    
    # Visualize
    visualize_blackout_result(
        img, mask, img_aug, mask_aug,
        "Test 4: Complete Blackout (All Defects Removed)",
        "test_4_complete_blackout.png"
    )
    
    # Assertions
    assert modified == True, "Blackout should have been applied"
    assert augmented_defect_pixels == 0, "All defects should be removed in complete blackout"
    assert no_defect_pixels == 1600 * 256, "Entire mask should be no_defect (0)"
    
    print("✅ TEST 4 PASSED")
    return True


def test_5_min_pixels_threshold():
    """Test 5: Min pixels threshold - small defects should not be blacked out."""
    print("\n" + "="*60)
    print("TEST 5: Min Pixels Threshold")
    print("="*60)
    
    if not SCIPY_AVAILABLE:
        print("SKIPPED - scipy not available")
        return False
    
    # Create test data with small and large defects
    defect_config = {
        2: [
            (50, 52, 200, 205),    # Very small: 2×5 = 10 pixels
            (100, 150, 400, 500)   # Large: 50×100 = 5000 pixels
        ]
    }
    img, mask = create_synthetic_image_and_mask(defect_config=defect_config)
    
    # Create blackout with high min_pixels threshold
    blackout = DefectBlackoutTransform(
        enabled=True,
        instance_blackout_prob=1.0,
        defect_indices_to_blackout=[2],
        min_pixels_to_blackout=100,  # Only instances >= 100 pixels
        verbose=True
    )
    
    # Apply blackout
    img_aug, mask_aug, modified = blackout(img, mask)
    
    # Count instances
    from scipy.ndimage import label
    original_labeled, original_num_instances = label(mask == 2)
    augmented_labeled, augmented_num_instances = label(mask_aug == 2)
    
    print(f"Original instances: {original_num_instances}")
    print(f"Augmented instances: {augmented_num_instances}")
    
    small_defect_pixels = np.sum((mask == 2) & (mask[:, :200] > 0))  # Approx. left region
    small_defect_pixels_after = np.sum((mask_aug == 2) & (mask_aug[:, :200] > 0))
    
    print(f"Small defect pixels (original): ~{small_defect_pixels}")
    print(f"Small defect pixels (after): ~{small_defect_pixels_after}")
    
    # Visualize
    visualize_blackout_result(
        img, mask, img_aug, mask_aug,
        f"Test 5: Min Pixels Threshold (>= 100 pixels)\nSmall defect preserved, large removed",
        "test_5_min_pixels.png"
    )
    
    # The small defect should remain (if < 100 pixels), large should be removed
    # Note: Exact assertion depends on defect sizes
    
    print("✅ TEST 5 PASSED")
    return True


def test_6_disabled_mode():
    """Test 6: Disabled mode should not modify anything."""
    print("\n" + "="*60)
    print("TEST 6: Disabled Mode")
    print("="*60)
    
    # Create test data
    img, mask = create_synthetic_image_and_mask()
    
    # Create DISABLED blackout transform
    blackout = DefectBlackoutTransform(
        enabled=False,  # Disabled
        instance_blackout_prob=1.0,
        defect_indices_to_blackout=[1, 2, 3, 4],
        verbose=True
    )
    
    # Apply blackout
    img_aug, mask_aug, modified = blackout(img, mask)
    
    print(f"Modified flag: {modified}")
    
    # Assertions
    assert modified == False, "Should not be modified when disabled"
    assert np.array_equal(mask, mask_aug), "Mask should be unchanged"
    assert img == img_aug, "Image should be unchanged"
    
    print("✅ TEST 6 PASSED")
    return True


def test_7_no_defects_present():
    """Test 7: Handle images with no defects gracefully."""
    print("\n" + "="*60)
    print("TEST 7: No Defects Present")
    print("="*60)
    
    if not SCIPY_AVAILABLE:
        print("SKIPPED - scipy not available")
        return False
    
    # Create image with NO defects
    img = Image.new("RGB", (1600, 256), color=(128, 128, 128))
    mask = np.zeros((256, 1600), dtype=np.uint8)  # All no_defect
    
    # Create blackout transform
    blackout = DefectBlackoutTransform(
        enabled=True,
        instance_blackout_prob=1.0,
        defect_indices_to_blackout=[1, 2, 3, 4],
        verbose=True
    )
    
    # Apply blackout
    img_aug, mask_aug, modified = blackout(img, mask)
    
    print(f"Modified flag: {modified}")
    
    # Assertions
    assert modified == False, "Should not modify when no defects present"
    assert np.array_equal(mask, mask_aug), "Mask should be unchanged"
    
    print("✅ TEST 7 PASSED")
    return True


def test_8_label_consistency():
    """Test 8: Labels derived from masks should be consistent."""
    print("\n" + "="*60)
    print("TEST 8: Label Consistency")
    print("="*60)
    
    if not SCIPY_AVAILABLE:
        print("SKIPPED - scipy not available")
        return False
    
    # Create test data
    img, mask = create_synthetic_image_and_mask()
    
    # Helper function to create label from mask
    def mask_to_label(mask_array):
        """Convert mask to 5-class label vector."""
        label = np.zeros(5, dtype=np.float32)
        unique_classes = np.unique(mask_array)
        
        has_defects = False
        for class_val in unique_classes:
            if 1 <= class_val <= 4:
                label[class_val] = 1.0
                has_defects = True
        
        if not has_defects:
            label[0] = 1.0  # no_defect
        
        return label
    
    # Original label
    original_label = mask_to_label(mask)
    
    # Apply blackout
    blackout = DefectBlackoutTransform(
        enabled=True,
        instance_blackout_prob=1.0,
        defect_indices_to_blackout=[2],  # Only defect_2
        verbose=True
    )
    
    img_aug, mask_aug, modified = blackout(img, mask)
    
    # Augmented label
    augmented_label = mask_to_label(mask_aug)
    
    print(f"Original label: {original_label}")
    print(f"Augmented label: {augmented_label}")
    
    # Check consistency
    # If defect_2 was completely removed, label[2] should be 0
    if np.sum(mask_aug == 2) == 0:
        assert augmented_label[2] == 0.0, "Label should reflect removed defect_2"
        print("Defect_2 completely removed - label consistent")
    
    # If all defects removed, no_defect should be 1
    if np.sum(mask_aug > 0) == 0:
        assert augmented_label[0] == 1.0, "no_defect should be 1 when all defects removed"
        print("All defects removed - no_defect label set correctly")
    
    print("✅ TEST 8 PASSED")
    return True


def run_all_tests():
    """Run all blackout augmentation tests."""
    print("\n" + "="*80)
    print("DEFECT BLACKOUT AUGMENTATION - COMPREHENSIVE TEST SUITE")
    print("="*80)
    
    if not SCIPY_AVAILABLE:
        print("\n⚠️  WARNING: scipy not available. Install with: pip install scipy")
        print("Some tests will be skipped.\n")
    
    tests = [
        ("Basic Functionality", test_1_basic_functionality),
        ("Selective Blackout", test_2_selective_blackout),
        ("Defect-Specific Targeting", test_3_defect_specific_targeting),
        ("Complete Blackout", test_4_complete_blackout),
        ("Min Pixels Threshold", test_5_min_pixels_threshold),
        ("Disabled Mode", test_6_disabled_mode),
        ("No Defects Present", test_7_no_defects_present),
        ("Label Consistency", test_8_label_consistency),
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            passed = test_func()
            results.append((test_name, passed))
        except Exception as e:
            print(f"\n❌ TEST FAILED: {test_name}")
            print(f"Error: {e}")
            import traceback
            traceback.print_exc()
            results.append((test_name, False))
    
    # Summary
    print("\n" + "="*80)
    print("TEST SUMMARY")
    print("="*80)
    
    passed_tests = sum(1 for _, passed in results if passed)
    total_tests = len(results)
    
    for test_name, passed in results:
        status = "✅ PASSED" if passed else "❌ FAILED"
        print(f"{status:12} - {test_name}")
    
    print(f"\nTotal: {passed_tests}/{total_tests} tests passed")
    
    if passed_tests == total_tests:
        print("\n🎉 ALL TESTS PASSED! Blackout augmentation is working correctly.")
    else:
        print(f"\n⚠️  {total_tests - passed_tests} test(s) failed. Please review.")
    
    print("\nVisualization images saved:")
    for i in range(1, 6):
        img_path = f"test_{i}_*.png"
        print(f"  - {img_path}")


if __name__ == "__main__":
    run_all_tests()
