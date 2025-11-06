"""
README for the modular, extensible training system.

Architecture overview and usage guide.
"""

# Severstal Defect Classification - Modular Training System

## 🏗️ Architecture Overview

The training system is built on three core principles:

1. **Modularity**: Each component (models, losses, augmentations, data) is independent
2. **Extensibility**: New models/losses/augmentations can be added via registries
3. **Configuration-Driven**: All training parameters are in Hydra YAML configs

### Directory Structure

```
code/
├── core/                          # Core modular components
│   ├── models/
│   │   ├── attention/             # Attention mechanisms (CBAM)
│   │   ├── backbones.py           # Model implementations
│   │   └── registry.py            # Model registry (easy model switching)
│   │
│   ├── losses/
│   │   ├── base.py                # Abstract loss base class
│   │   ├── focal_loss.py          # Focal loss with dynamic α, γ
│   │   ├── bce_loss.py            # BCE loss wrapper
│   │   └── registry.py            # Loss registry
│   │
│   ├── augmentation/
│   │   └── pipelines.py           # Reusable, toggleable augmentations
│   │
│   ├── data/
│   │   ├── dataset.py             # Full image dataset (256x1600)
│   │   ├── splitting.py           # Stratified splitting (preserves class distribution)
│   │   └── loaders.py             # DataLoader creation
│   │
│   └── training/
│       ├── trainer.py             # Main training orchestrator
│       ├── callbacks.py           # Early stopping, checkpointing
│       └── metrics.py             # Evaluation metrics
│
├── train.py                       # Clean entry point (minimal code)
│
├── experiments/
│   ├── configs/                   # Experiment configs (optional)
│   └── results/                   # Training results, logs, checkpoints
│
└── config/
    └── train_config.yaml          # Main Hydra configuration
```

## 🚀 Quick Start

### Basic Training

```bash
# Training with default configuration
python code/train.py

# Training with custom model and loss
python code/train.py model.name=convnext_tiny_cbam loss.type=focal_loss

# Custom batch size and epochs
python code/train.py data.batch_size=6 training.num_epochs=100

# Different data split
python code/train.py data.split_strategy=stratified_80_10_10
```

### Configuration Override

Any YAML parameter can be overridden from command line:

```bash
# Multiple overrides
python code/train.py \
  model.name=convnext_tiny_cbam \
  loss.alpha=dynamic \
  loss.gamma=2.0 \
  training.num_epochs=50 \
  training.early_stopping_patience=15 \
  data.batch_size=4
```

## 📦 Core Components

### 1. Models (Registry Pattern)

**Why registries?** Allows easy switching between models without code changes.

```python
from core.models.registry import get_registry

registry = get_registry()
model = registry.get("convnext_tiny_cbam", num_classes=4, pretrained=True)

# List available models
print(registry.list_models())
```

**Currently available:**
- `convnext_tiny_cbam`: ConvNext-Tiny with CBAM at stages 3-4

**To add a new model:**
1. Implement model class in `core/models/backbones.py`
2. Register in `core/models/registry.py`

### 2. Loss Functions (Registry Pattern)

**Why registries?** Easy comparison of different loss functions.

```python
from core.losses.registry import get_registry

registry = get_registry()

# Focal Loss with dynamic α computation
loss_fn = registry.get("focal_loss", num_classes=4, alpha="dynamic", gamma=2.0)

# BCE with logits
loss_fn = registry.get("bce_with_logits", num_classes=4)
```

**Features:**
- **Dynamic α computation**: Automatically weights classes by frequency
- **Multi-label support**: Binary classification per class
- **Consistent interface**: All losses inherit from `BaseLoss`

**Currently available:**
- `focal_loss`: Focal Loss with configurable α, γ
- `bce_with_logits`: Binary Cross-Entropy with Logits

### 3. Data Augmentations

**Toggleable components** - each augmentation can be on/off:

```yaml
# Config example
augmentation:
  horizontal_flip: 0.5
  vertical_flip: 0.3
  rotation:
    degrees: 15
    probability: 0.5
  brightness_contrast:
    brightness: 0.2
    contrast: 0.2
  color_jitter:
    saturation: 0.1
    hue: 0.05
  gaussian_blur:
    kernel_size: 3
```

Set to `null` or `false` to disable any augmentation.

### 4. Data & Stratified Splitting

**Severstal Dataset:**
- Loads full 256x1600 images
- No resizing or cropping (preserves small defects)
- Multi-label classification (up to 4 classes per image)

**Stratified Splitting:**
- Preserves class distribution across train/val/test splits
- Essential for imbalanced datasets
- Options: 70/15/15 or 80/10/10

```python
from core.data import StratifiedSplitter

splitter = StratifiedSplitter(random_state=42)
train_idx, val_idx, test_idx = splitter.split(
    labels,  # (N, C) binary matrix
    split_ratios=(0.7, 0.15, 0.15)
)
```

### 5. Training Pipeline

The `Trainer` class orchestrates:
- Training loop with loss computation
- Validation with metric tracking
- Testing on held-out set
- Early stopping with checkpointing
- Experiment logging and result saving

**Key features:**
- Automatic checkpoint saving of best model
- Per-class and macro/micro metrics
- JSON experiment tracking for UI comparison
- Tensorboard-ready logging

## 📊 Configuration System (Hydra)

All training parameters in `config/train_config.yaml`:

```yaml
model:
  name: "convnext_tiny_cbam"
  num_classes: 4
  pretrained: true
  cbam_stages: [3, 4]

loss:
  type: "focal_loss"
  alpha: "dynamic"
  gamma: 2.0
  reduction: "mean"

training:
  num_epochs: 100
  learning_rate: 0.001
  weight_decay: 0.0001
  early_stopping_patience: 15
  warmup_epochs: 5

data:
  batch_size: 4
  split_strategy: "stratified_70_15_15"
```

### Configuration Hierarchy

```
train_config.yaml (main)
    ↓
Command line overrides
    ↓
Final merged configuration
```

## 🔍 Model Registry Explanation

### Why Use Registries?

**Problem:** Without registries, adding new models requires:
- Modifying training script
- Adding conditional imports
- Hard to discover available options

**Solution:** Registries are dictionaries that map names → classes:

```python
class ModelRegistry:
    def register(self, name: str, model_class):
        self._registry[name] = model_class
    
    def get(self, name: str, **kwargs):
        return self._registry[name](**kwargs)
```

**Benefits:**
✓ Add new models without modifying training code
✓ Easy to list available models
✓ UI can dynamically populate model choices
✓ Configuration-driven model selection

### How It Works

1. **Define model class** (inherits from `nn.Module`)
2. **Register in registry** (`registry.register("name", ModelClass)`)
3. **Use in config** (`model.name: convnext_tiny_cbam`)
4. **Trainer instantiates** (`model_registry.get(cfg.model.name, **params)`)

### Example: Adding a New Model

```python
# 1. Define in core/models/backbones.py
class MyNewModel(nn.Module):
    def __init__(self, num_classes=4, pretrained=True):
        super().__init__()
        # Implementation...

# 2. Register in core/models/registry.py _register_default_models()
registry.register(
    "my_new_model",
    MyNewModel,
    description="My new model for defect detection"
)

# 3. Use in config
model:
  name: "my_new_model"
  num_classes: 4
  pretrained: true

# 4. Train
python code/train.py
```

## 📈 Understanding CBAM Placement in ConvNext-Tiny

### Why Stages 3-4?

ConvNext-Tiny has 4 stages with increasing receptive fields:

```
Stage 0 (stem):     Small features, high resolution (no CBAM needed)
Stage 1:            Low-level features (no CBAM)
Stage 2:            Mid-level features (no CBAM)
Stage 3: ← CBAM     Semantic features, 64-channel (DEFECT SIGNATURES)
Stage 4: ← CBAM     High-level features, 768-channel (SPATIAL LOCALIZATION)
```

**Why stages 3-4?**
- Receptive fields are large enough to capture defect context
- Channel dimensions high enough for meaningful attention
- Defect patterns become identifiable at semantic level

### CBAM = Channel + Spatial Attention

```
Input ──→ Channel Attention ──→ Spatial Attention ──→ Output
            (which channels?)    (where in image?)
```

**Channel Attention:** Learns which feature channels encode defect information
**Spatial Attention:** Learns where defects appear in the feature maps

## 🎯 Focal Loss with Dynamic Parameters

### The Problem (Class Imbalance)

Severstal dataset:
- defect_3: 73.47% of samples (common)
- defect_1: 15.38% of samples
- defect_4: 9.55% of samples
- defect_2: 1.60% of samples (rare)

Standard BCE loss treats all classes equally → poor rare class performance

### Focal Loss Solution

$$FL(p_t) = -\alpha_t(1 - p_t)^{\gamma} \log(p_t)$$

- **α (alpha):** Class-specific weighting. Computed from class frequencies.
- **γ (gamma):** Focusing parameter (typically 2.0). Down-weights easy examples.

### Dynamic α Computation

```python
loss_fn = FocalLoss(num_classes=4, alpha="dynamic", gamma=2.0)

# On first forward pass, α is computed from target distribution
α_c = 1 - (freq_c / total_samples)
```

This ensures rare classes get higher weights automatically!

## 💾 Experiment Tracking

Each training creates an experiment directory:

```
code/experiments/results/
├── experiment_20231206_153022/
│   ├── results.json                 # Metrics, config, hyperparams
│   ├── model_final.pt               # Final model weights
│   ├── checkpoints/
│   │   └── best_model_epoch*.pt     # Best checkpoint
│   └── logs/                         # Optional tensorboard logs
```

**results.json contains:**
- Model architecture info
- Training hyperparameters
- Loss function configuration
- Best validation metrics
- Test metrics
- Full training history

## 🔌 Preparing for UI Integration

The modular architecture is designed for future UI:

1. **Component Discovery:** Registry → list all available options
2. **Configuration:** YAML configs → easily represent model/loss/augmentation choices
3. **Experiment Tracking:** JSON results → compare experiments
4. **Metrics Logging:** Structured metrics → visualize training curves

### What the UI will need:

```python
# 1. List available models
models = model_registry.list_models()

# 2. List available losses
losses = loss_registry.list_losses()

# 3. Get experiment results
results = load_json("experiments/results/*/results.json")

# 4. Compare experiments
compare_results([results1, results2, results3])
```

All infrastructure is already in place!

## 📝 Example: Reproducing Training

```bash
# Original training with Focal Loss + CBAM
python code/train.py \
  model.name=convnext_tiny_cbam \
  loss.type=focal_loss \
  loss.alpha=dynamic \
  loss.gamma=2.0 \
  data.batch_size=6 \
  training.num_epochs=100 \
  training.early_stopping_patience=15 \
  experiment.seed=42

# Alternative: BCE Loss for comparison
python code/train.py \
  model.name=convnext_tiny_cbam \
  loss.type=bce_with_logits \
  data.batch_size=6 \
  training.num_epochs=100 \
  experiment.seed=42

# Results will be in code/experiments/results/<timestamp>/results.json
```

## 🛠️ Extensibility Checklist

To add new features, follow these patterns:

### Adding a New Model
- [ ] Implement in `core/models/backbones.py`
- [ ] Register in `core/models/registry.py`
- [ ] Update config schema if needed
- [ ] Test with `python code/train.py model.name=new_model`

### Adding a New Loss Function
- [ ] Implement in `core/losses/` (inherit from `BaseLoss`)
- [ ] Register in `core/losses/registry.py`
- [ ] Implement `get_params()` for logging
- [ ] Test with `python code/train.py loss.type=new_loss`

### Adding a New Augmentation
- [ ] Add to `AugmentationPipeline.build()` in `core/augmentation/pipelines.py`
- [ ] Add to config YAML
- [ ] Can be toggled on/off via config

### Adding Metrics
- [ ] Add to `compute_metrics()` in `core/training/metrics.py`
- [ ] Will automatically be tracked in experiment results

## ✅ Next Steps

1. **Test training script**: `python code/train.py`
2. **Check experiment results**: Look in `code/experiments/results/`
3. **Modify config**: Edit `config/train_config.yaml`
4. **Add new models**: Follow the registry pattern
5. **Build UI**: Use registries + JSON results for experiment comparison

---

**The system is now ready for production training and future UI development!**
