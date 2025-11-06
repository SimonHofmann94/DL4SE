"""
Quick validation script to verify all components are working.

Run this to check that the system is set up correctly before training.
"""

import sys
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Add code to path
sys.path.insert(0, str(Path(__file__).parent))


def test_imports():
    """Test that all imports work."""
    logger.info("Testing imports...")
    
    try:
        from core.models.attention.cbam import CBAM, ChannelAttention, SpatialAttention
        logger.info("✓ CBAM modules imported")
    except Exception as e:
        logger.error(f"✗ Failed to import CBAM: {e}")
        return False
    
    try:
        from core.models.backbones import ConvNextTinyCBAM
        logger.info("✓ ConvNextTinyCBAM imported")
    except Exception as e:
        logger.error(f"✗ Failed to import ConvNextTinyCBAM: {e}")
        return False
    
    try:
        from core.models.registry import get_registry as get_model_registry
        logger.info("✓ Model registry imported")
    except Exception as e:
        logger.error(f"✗ Failed to import model registry: {e}")
        return False
    
    try:
        from core.losses.base import BaseLoss
        from core.losses.focal_loss import FocalLoss
        from core.losses.bce_loss import BCEWithLogitsLossWrapper
        from core.losses.registry import get_registry as get_loss_registry
        logger.info("✓ Loss functions imported")
    except Exception as e:
        logger.error(f"✗ Failed to import losses: {e}")
        return False
    
    try:
        from core.augmentation import get_augmentation_pipeline
        logger.info("✓ Augmentation pipeline imported")
    except Exception as e:
        logger.error(f"✗ Failed to import augmentation: {e}")
        return False
    
    try:
        from core.data import SeverstalFullImageDataset, StratifiedSplitter, create_dataloaders
        logger.info("✓ Data components imported")
    except Exception as e:
        logger.error(f"✗ Failed to import data components: {e}")
        return False
    
    try:
        from core.training import Trainer, EarlyStoppingCallback, compute_metrics
        logger.info("✓ Training components imported")
    except Exception as e:
        logger.error(f"✗ Failed to import training components: {e}")
        return False
    
    return True


def test_model_instantiation():
    """Test that model can be instantiated."""
    logger.info("\nTesting model instantiation...")
    
    try:
        import torch
        from core.models.registry import get_registry
        
        registry = get_registry()
        model = registry.get("convnext_tiny_cbam", num_classes=4, pretrained=False)
        
        logger.info(f"✓ Model instantiated: {type(model).__name__}")
        
        # Test forward pass
        x = torch.randn(1, 3, 256, 1600)
        out = model(x)
        logger.info(f"✓ Forward pass successful: input {x.shape} -> output {out.shape}")
        
        return True
    except Exception as e:
        logger.error(f"✗ Model instantiation failed: {e}")
        return False


def test_loss_instantiation():
    """Test that loss functions can be instantiated."""
    logger.info("\nTesting loss function instantiation...")
    
    try:
        import torch
        from core.losses.registry import get_registry
        
        registry = get_registry()
        
        # Test focal loss
        loss_fn = registry.get("focal_loss", num_classes=4, alpha="dynamic", gamma=2.0)
        logger.info(f"✓ Focal loss instantiated: {loss_fn.get_name()}")
        
        # Test BCE loss
        loss_fn = registry.get("bce_with_logits", num_classes=4)
        logger.info(f"✓ BCE loss instantiated: {loss_fn.get_name()}")
        
        return True
    except Exception as e:
        logger.error(f"✗ Loss instantiation failed: {e}")
        return False


def test_augmentation_pipeline():
    """Test augmentation pipeline."""
    logger.info("\nTesting augmentation pipeline...")
    
    try:
        from PIL import Image
        from core.augmentation import get_augmentation_pipeline
        
        # Create dummy image
        dummy_img = Image.new("RGB", (1600, 256), color="white")
        
        # Get augmentation pipeline
        transform = get_augmentation_pipeline(
            image_size=(256, 1600),
            preset="standard"
        )
        
        # Apply transform
        transformed = transform(dummy_img)
        logger.info(f"✓ Augmentation pipeline working: output shape {transformed.shape}")
        
        return True
    except Exception as e:
        logger.error(f"✗ Augmentation pipeline failed: {e}")
        return False


def test_stratified_splitting():
    """Test stratified splitting."""
    logger.info("\nTesting stratified splitting...")
    
    try:
        import numpy as np
        from core.data import StratifiedSplitter
        
        # Create synthetic data
        np.random.seed(42)
        labels = np.random.rand(100, 4) > np.array([0.3, 0.7, 0.2, 0.85])
        
        splitter = StratifiedSplitter(random_state=42)
        train_idx, val_idx, test_idx = splitter.split(
            labels,
            split_ratios=(0.7, 0.15, 0.15)
        )
        
        logger.info(f"✓ Stratified splitting working:")
        logger.info(f"  Train: {len(train_idx)} samples")
        logger.info(f"  Val: {len(val_idx)} samples")
        logger.info(f"  Test: {len(test_idx)} samples")
        
        return True
    except Exception as e:
        logger.error(f"✗ Stratified splitting failed: {e}")
        return False


def test_registries():
    """Test model and loss registries."""
    logger.info("\nTesting registries...")
    
    try:
        from core.models.registry import get_registry as get_model_registry
        from core.losses.registry import get_registry as get_loss_registry
        
        model_registry = get_model_registry()
        loss_registry = get_loss_registry()
        
        models = model_registry.list_models()
        losses = loss_registry.list_losses()
        
        logger.info(f"✓ Model registry has {len(models)} models:")
        for name, desc in models.items():
            logger.info(f"    - {name}: {desc}")
        
        logger.info(f"✓ Loss registry has {len(losses)} losses:")
        for name, desc in losses.items():
            logger.info(f"    - {name}: {desc}")
        
        return True
    except Exception as e:
        logger.error(f"✗ Registry test failed: {e}")
        return False


def main():
    """Run all tests."""
    logger.info("="*70)
    logger.info("SYSTEM VALIDATION TESTS")
    logger.info("="*70)
    
    tests = [
        ("Imports", test_imports),
        ("Model Instantiation", test_model_instantiation),
        ("Loss Functions", test_loss_instantiation),
        ("Augmentation Pipeline", test_augmentation_pipeline),
        ("Stratified Splitting", test_stratified_splitting),
        ("Registries", test_registries),
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            logger.error(f"✗ {test_name} test crashed: {e}")
            results.append((test_name, False))
    
    logger.info("\n" + "="*70)
    logger.info("TEST SUMMARY")
    logger.info("="*70)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "✓ PASS" if result else "✗ FAIL"
        logger.info(f"{status}: {test_name}")
    
    logger.info(f"\nTotal: {passed}/{total} tests passed")
    
    if passed == total:
        logger.info("\n✓ ALL TESTS PASSED - System is ready for training!")
        return 0
    else:
        logger.error("\n✗ Some tests failed - please fix issues before training")
        return 1


if __name__ == "__main__":
    sys.exit(main())
