# Complete Project Documentation Index

## 📚 Documentation Files

### 1. **QUICK_START.md** ⭐ START HERE
   - Common commands for training
   - Hyperparameter tuning guide
   - Troubleshooting tips
   - Adding new components

### 2. **ARCHITECTURE_README.md**
   - Deep dive into system design
   - Component explanations
   - Registry pattern explained
   - CBAM placement justification
   - Focal loss with dynamic parameters
   - UI integration design

### 3. **IMPLEMENTATION_SUMMARY.md**
   - What has been built
   - Project structure
   - Key design decisions
   - Metrics tracked
   - Next steps

### 4. **README.md** (Project README)
   - Original project overview
   - Dataset information
   - Getting started

---

## 🗂️ Project Structure

```
code/
├── validate_system.py              # Run first: validates setup
├── train.py                         # Main entry point (clean, 60 lines)
├── models.py, utils.py, datasets.py # Old code (deprecated)
│
├── core/                            # NEW: Core modular components
│   ├── models/
│   │   ├── attention/
│   │   │   └── cbam.py             # CBAM: channel + spatial attention
│   │   ├── backbones.py            # ConvNext-Tiny + CBAM
│   │   └── registry.py             # Model registry (extensible)
│   │
│   ├── losses/
│   │   ├── base.py                 # Abstract loss base class
│   │   ├── focal_loss.py           # Focal loss + dynamic α, γ
│   │   ├── bce_loss.py             # BCE wrapper
│   │   └── registry.py             # Loss registry
│   │
│   ├── augmentation/
│   │   └── pipelines.py            # Reusable augmentation components
│   │
│   ├── data/
│   │   ├── dataset.py              # SeverstalFullImageDataset (256x1600)
│   │   ├── splitting.py            # StratifiedSplitter
│   │   └── loaders.py              # DataLoader utilities
│   │
│   └── training/
│       ├── trainer.py              # Main training orchestrator
│       ├── callbacks.py            # Early stopping, checkpointing
│       └── metrics.py              # Metric computation
│
├── experiments/
│   ├── configs/                    # Experiment configs (optional)
│   └── results/                    # Training results (auto-generated)
│
└── config/
    └── train_config.yaml           # Main Hydra configuration

config/
├── train_config.yaml               # NEW: Main configuration file

data/
├── images/                         # Training images (256x1600)
├── annotations/                    # Image annotations (JSON)
└── Severstal/                      # Severstal dataset
```

---

## 🎯 Key Components Explained

### Model: ConvNext-Tiny with CBAM
- **Why ConvNext?** Modern CNN with good efficiency/accuracy tradeoff
- **CBAM Where?** Stages 3-4 (semantic level for defects)
- **Attention Types?** Both spatial + channel (CBAM = both together)
- **Parameters?** ~28M total, trained with ImageNet pretraining

### Loss: Focal Loss
- **Why Focal?** Class imbalance is severe (defect_2 is 1.6% vs defect_3 is 73%)
- **Dynamic α?** Automatically computed from class frequencies
- **γ = 2.0?** Standard focusing parameter (down-weights easy examples)
- **vs BCE?** Focal loss focuses on hard negatives, BCE treats equally

### Data: Stratified Splitting
- **Why Stratified?** Naive split can lose rare classes
- **How?** Each class split independently, then combined
- **Ratios?** 70/15/15 (main) or 80/10/10 (optional)
- **Result?** Each split has proportional class representation

### Training: Modular Pipeline
- **Entry Point?** `code/train.py` (clean, 60 lines)
- **Configuration?** YAML-based (Hydra)
- **Registries?** Models and losses via registries (easy swapping)
- **Monitoring?** Early stopping, checkpointing, full metrics logging

---

## 🚀 Usage Patterns

### Pattern 1: Quick Training
```bash
python code/train.py
# Uses all defaults from config/train_config.yaml
```

### Pattern 2: Experiment with Variations
```bash
# Focal Loss
python code/train.py loss.type=focal_loss loss.gamma=2.0

# BCE Loss (for comparison)
python code/train.py loss.type=bce_with_logits

# Different split
python code/train.py data.split_strategy=stratified_80_10_10

# Results auto-saved to code/experiments/results/
```

### Pattern 3: Hyperparameter Tuning
```bash
python code/train.py \
  optimizer.lr=0.001 \
  data.batch_size=6 \
  training.num_epochs=100 \
  training.early_stopping_patience=15
```

---

## 📊 Metrics & Results

Each training generates:

```
code/experiments/results/experiment_20231206_153022/
├── results.json                     # All metrics + config
├── model_final.pt                   # Final model weights
└── checkpoints/
    └── best_model_epoch42_f1_0.89.pt
```

### Key Metrics
- **Per-class:** Precision, Recall, F1
- **Macro:** Average across classes (considers rare classes equally)
- **Micro:** Average across all samples (weighted by class frequency)
- **Overall:** Accuracy, Hamming loss

---

## 🔧 Extending the System

### Adding a New Model
```python
# 1. core/models/backbones.py
class MyModel(nn.Module):
    def __init__(self, num_classes=4, **kwargs):
        ...

# 2. core/models/registry.py - in _register_default_models()
registry.register("my_model", MyModel, "Description...")

# 3. Use in training
python code/train.py model.name=my_model
```

### Adding a New Loss
```python
# 1. core/losses/my_loss.py
class MyLoss(BaseLoss):
    def forward(self, predictions, targets):
        ...
    def get_params(self):
        return {...}

# 2. core/losses/registry.py - in _register_default_losses()
registry.register("my_loss", MyLoss, "Description...")

# 3. Use in training
python code/train.py loss.type=my_loss
```

### Adding Augmentation
```python
# Edit core/augmentation/pipelines.py
# Add to AugmentationPipeline.build() method
elif aug_name == "new_aug" and aug_config:
    self.transform_list.append(new_transform)

# Use in config/train_config.yaml
augmentation:
  new_aug: <config>
```

---

## ✅ System Validation

Run validation before first training:

```bash
python code/validate_system.py
```

This checks:
- All imports working
- Model instantiation
- Loss functions
- Augmentation pipeline
- Stratified splitting
- Registries populated

---

## 🎓 Understanding Key Concepts

### Registry Pattern (Why It Matters)

**Without registries:**
```python
if loss_type == "focal":
    loss = FocalLoss(...)
elif loss_type == "bce":
    loss = BCELoss(...)
# Add new loss → modify this code
```

**With registries:**
```python
registry = LossRegistry()
loss = registry.get(loss_type, **params)
# Add new loss → register it, no code changes
```

### Dynamic Focal Loss α

For imbalanced data, α should weight rare classes higher:

$$\alpha_c = 1 - \frac{\text{freq}_c}{\text{total_samples}}$$

This is computed automatically on first forward pass!

### Stratified Splitting

For multi-label data:
1. For each class, split samples proportionally
2. Combine all class splits
3. Result: balanced class distribution across train/val/test

---

## 📋 Pre-Training Checklist

- [ ] Validation passes: `python code/validate_system.py`
- [ ] Data in `data/images/` and `data/annotations/`
- [ ] Config reviewed: `config/train_config.yaml`
- [ ] Batch size appropriate for GPU
- [ ] Random seed set for reproducibility
- [ ] Experiment name noted

---

## 🎯 Recommended Training Settings (By GPU)

### RTX 5090 (32GB)
```bash
python code/train.py data.batch_size=8 training.num_epochs=100
```

### RTX 4090 (24GB)
```bash
python code/train.py data.batch_size=6 training.num_epochs=100
```

### RTX 3090 (24GB)
```bash
python code/train.py data.batch_size=4 training.num_epochs=100
```

### RTX 3060 (12GB)
```bash
python code/train.py data.batch_size=2 training.num_epochs=80
```

---

## 🚀 Next: Building the UI

The modular architecture is ready for UI:

1. **Component Discovery:** Registries list models/losses/augmentations
2. **Configuration:** UI form → YAML → training script
3. **Experiment Tracking:** JSON results → comparison dashboard
4. **Metrics Visualization:** Training history → graphs

All infrastructure is in place!

---

## 📞 Support

### For Questions About:
- **System Design:** See ARCHITECTURE_README.md
- **Quick Commands:** See QUICK_START.md
- **Code Structure:** See this document + inline code comments
- **What Was Built:** See IMPLEMENTATION_SUMMARY.md

### Key Code Files to Review:
- `code/train.py` - Entry point, read first
- `code/core/training/trainer.py` - Training loop
- `code/core/models/backbones.py` - Model architecture
- `code/core/losses/focal_loss.py` - Loss implementation
- `config/train_config.yaml` - All parameters

---

**Ready to start training! Run:** `python code/validate_system.py` first, then `python code/train.py`
