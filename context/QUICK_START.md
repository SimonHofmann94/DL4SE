# Quick Reference Guide - Severstal Training System

## 🚀 Quick Start (3 commands)

```bash
# 1. Validate system setup
python code/validate_system.py

# 2. Training with defaults
python code/train.py

# 3. Check results
ls code/experiments/results/experiment_*/results.json
```

## 📝 Common Commands

### Training with Different Models
```bash
# ConvNext-Tiny with CBAM (default)
python code/train.py model.name=convnext_tiny_cbam

# Add more models to core/models/registry.py and use them here
```

### Training with Different Loss Functions
```bash
# Focal Loss (recommended for imbalanced data)
python code/train.py loss.type=focal_loss loss.alpha=dynamic loss.gamma=2.0

# BCE with Logits (for comparison)
python code/train.py loss.type=bce_with_logits
```

### Different Data Splits
```bash
# 70% train, 15% val, 15% test (default)
python code/train.py data.split_strategy=stratified_70_15_15

# 80% train, 10% val, 10% test
python code/train.py data.split_strategy=stratified_80_10_10
```

### Batch Size & GPU Configuration
```bash
# Adjust batch size for your GPU memory
python code/train.py data.batch_size=6      # For RTX 4090
python code/train.py data.batch_size=4      # For RTX 3090
python code/train.py data.batch_size=2      # For RTX 3060

# Workers for data loading
python code/train.py data.num_workers=4     # Parallel data loading
```

### Training Duration
```bash
# More epochs with early stopping
python code/train.py training.num_epochs=100 training.early_stopping_patience=15

# Quick test run
python code/train.py training.num_epochs=5 training.early_stopping_patience=2
```

### Augmentation Control
```bash
# All augmentations (default standard)
python code/train.py augmentation.horizontal_flip=0.5 augmentation.vertical_flip=0.3

# Light augmentation
python code/train.py augmentation.horizontal_flip=0.3 augmentation.vertical_flip=false

# Heavy augmentation
python code/train.py augmentation.horizontal_flip=0.7 augmentation.vertical_flip=0.5 \
                     augmentation.color_jitter.saturation=0.2 augmentation.gaussian_blur=3

# No augmentation
python code/train.py augmentation=null
```

### Learning Rate & Optimization
```bash
# Higher learning rate
python code/train.py optimizer.lr=0.01

# Lower learning rate
python code/train.py optimizer.lr=0.0001

# Different optimizer
python code/train.py optimizer.type=adam
python code/train.py optimizer.type=sgd
```

### Early Stopping Tuning
```bash
# More patient (train longer)
python code/train.py training.early_stopping_patience=20

# Less patient (stop sooner)
python code/train.py training.early_stopping_patience=5
```

### Reproducibility
```bash
# Fixed random seed
python code/train.py experiment.seed=42

# Different seed for comparison
python code/train.py experiment.seed=123
```

## 🔍 Understanding Key Hyperparameters

| Parameter | Range | Effect |
|-----------|-------|--------|
| `loss.gamma` | 0.5-3.0 | Lower = gentle, Higher = hard negatives focused |
| `loss.alpha` | "dynamic" or 0.0-1.0 | "dynamic" = auto from data, float = manual |
| `training.warmup_epochs` | 0-10 | Gradual LR increase (helps training stability) |
| `data.batch_size` | 2-8 | Larger = noisier gradients but uses more memory |
| `training.learning_rate` | 1e-4 to 1e-2 | Default 1e-3 is usually good for transfer learning |

## 📊 Output Interpretation

### After Training Completes:
```
code/experiments/results/experiment_20231206_153022/
├── results.json                          # Main results file
├── model_final.pt                        # Final model weights
└── checkpoints/
    └── best_model_epoch42_f1_0.8934.pt  # Best checkpoint
```

### Key Metrics in results.json:
```json
{
  "best_val_metrics": {
    "f1_macro": 0.8934,           # Overall F1 score
    "f1_micro": 0.8956,           # Micro-averaged F1
    "precision_macro": 0.8812,    # Precision across classes
    "recall_macro": 0.9234,       # Recall across classes
    "class_0_f1": 0.85,           # defect_1 F1
    "class_1_f1": 0.12,           # defect_2 F1 (rarest)
    "class_2_f1": 0.95,           # defect_3 F1 (most common)
    "class_3_f1": 0.78            # defect_4 F1
  },
  "test_metrics": { ... }        # Test set performance
}
```

### What to Look For:
- ✓ `f1_macro` > 0.85 = Good overall performance
- ⚠️ `class_1_f1` << other classes = Rare class still struggling (expected)
- ✓ Test metrics close to val metrics = Good generalization
- ✗ Test metrics << val metrics = Overfitting (increase early stopping patience)

## 🛠️ Troubleshooting

### GPU Out of Memory
```bash
# Reduce batch size
python code/train.py data.batch_size=2

# Reduce image size in config if absolutely necessary (not recommended)
```

### Training Too Slow
```bash
# Increase num_workers for parallel data loading
python code/train.py data.num_workers=4

# Reduce number of epochs
python code/train.py training.num_epochs=50
```

### Poor Validation Metrics
```bash
# Increase early stopping patience to train longer
python code/train.py training.early_stopping_patience=20

# Try focal loss if using BCE
python code/train.py loss.type=focal_loss loss.gamma=2.0

# Try different learning rate
python code/train.py optimizer.lr=0.0005
```

### Rare Classes Performing Poorly
```bash
# Ensure focal loss is enabled
python code/train.py loss.type=focal_loss loss.alpha=dynamic

# Increase gamma for harder focus on hard examples
python code/train.py loss.gamma=2.5
```

## 🔄 Comparing Models/Losses

Run multiple trainings and compare results:

```bash
# Training 1: Focal Loss + CBAM
python code/train.py \
  model.name=convnext_tiny_cbam \
  loss.type=focal_loss \
  experiment.seed=42
# Results → code/experiments/results/experiment_A/results.json

# Training 2: BCE Loss + CBAM
python code/train.py \
  model.name=convnext_tiny_cbam \
  loss.type=bce_with_logits \
  experiment.seed=42
# Results → code/experiments/results/experiment_B/results.json

# Compare metrics (can be automated in UI later)
```

## 📚 Adding New Components

### Adding a Model
1. Implement in `core/models/backbones.py`
2. Register in `core/models/registry.py`
3. Use: `python code/train.py model.name=new_model`

### Adding a Loss
1. Implement in `core/losses/` (inherit `BaseLoss`)
2. Register in `core/losses/registry.py`
3. Use: `python code/train.py loss.type=new_loss`

### Adding an Augmentation
1. Add to `get_augmentation_pipeline()` in `core/augmentation/pipelines.py`
2. Use: `python code/train.py augmentation.new_aug=0.5`

## ✅ Pre-Training Checklist

- [ ] System validation passes: `python code/validate_system.py`
- [ ] Data is extracted to `data/images/` and `data/annotations/`
- [ ] Config looks correct: Check `config/train_config.yaml`
- [ ] Batch size fits GPU memory
- [ ] Early stopping patience is reasonable (10-20)
- [ ] Experiment seed is set for reproducibility

## 📞 Common Issues & Solutions

| Issue | Solution |
|-------|----------|
| Module not found | Run validation script first |
| CUDA out of memory | Reduce batch size |
| Data not found | Check paths in config |
| Training hangs | Check data loading with verbose logging |
| Poor metrics | Try focal loss, increase patience |
| Rare classes fail | Use focal loss with dynamic α |

---

**For full documentation, see ARCHITECTURE_README.md**
