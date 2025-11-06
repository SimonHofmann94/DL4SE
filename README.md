# ConvNext-Tiny with CBAM for Severstal Steel Defect Detection

Deep learning model for multi-label classification of steel surface defects using ConvNext-Tiny backbone enhanced with CBAM (Convolutional Block Attention Module) attention mechanisms.

## 🎯 Features

- **ConvNext-Tiny + CBAM**: Modern CNN with spatial and channel attention at stages 3-4
- **Focal Loss**: Handles severe class imbalance with dynamic α computation
- **Full Resolution**: Processes 256×1600 images without resizing to preserve small defects
- **Modular Architecture**: Registry pattern for easy model/loss swapping
- **Production Ready**: Type hints, logging, checkpointing, early stopping

## 🚀 Quick Start (Local)

```bash
# Install dependencies
pip install -r requirements.txt

# Validate system
python code/validate_system.py

# Train with default config
python code/train.py

# Or customize
python code/train.py data.batch_size=6 training.num_epochs=150
```

## ☁️ RunPod Setup

See [GIT_LFS_SETUP.md](GIT_LFS_SETUP.md) for detailed GitHub/LFS setup instructions.

**Quick RunPod workflow:**

```bash
# 1. Clone on RunPod
git clone https://github.com/YOUR_USERNAME/YOUR_REPO.git
cd YOUR_REPO

# 2. Run setup script
chmod +x setup_runpod.sh
./setup_runpod.sh

# 3. Start training
python code/train.py
```

## 📁 Project Structure

```
├── code/
│   ├── core/                 # Core modules
│   │   ├── models/          # ConvNext-Tiny + CBAM
│   │   ├── losses/          # Focal Loss, BCE
│   │   ├── data/            # Dataset, splitting
│   │   ├── augmentation/    # Image transforms
│   │   └── training/        # Trainer, callbacks, metrics
│   ├── train.py             # Training entry point
│   └── validate_system.py   # System validation
├── config/
│   └── train_config.yaml    # Hydra configuration
├── data/
│   ├── annotations/         # JSON annotations
│   ├── zips/               # Data archives (Git LFS)
│   └── images/             # Extracted images (not in git)
└── context/                 # Documentation
```

## 📊 Model Architecture

- **Backbone**: ConvNext-Tiny (28M parameters, ImageNet pretrained)
- **Attention**: CBAM modules at stages 3-4 (semantic level)
- **Head**: Dropout(0.5) → Linear(num_features → 4)
- **Output**: 4 logits for multi-label classification

## 🔧 Configuration

All parameters in `config/train_config.yaml`:

```yaml
model:
  name: convnext_tiny_cbam
  cbam_stages: [3, 4]
  
loss:
  type: focal_loss
  gamma: 2.0
  
training:
  num_epochs: 100
  early_stopping_patience: 15
```

Override via command line:
```bash
python code/train.py loss.type=bce_with_logits data.batch_size=4
```

## 📚 Documentation

- [QUICK_START.md](context/QUICK_START.md) - Common commands
- [ARCHITECTURE_README.md](context/ARCHITECTURE_README.md) - Design deep dive
- [PROJECT_INDEX.md](context/PROJECT_INDEX.md) - Complete file index
- [GIT_LFS_SETUP.md](GIT_LFS_SETUP.md) - GitHub & RunPod setup

## 🎓 Key Design Decisions

1. **CBAM at stages 3-4 only**: Optimal receptive field for small defects
2. **Focal Loss with dynamic α**: Auto-computed from class frequencies
3. **No image resizing**: Preserves spatial detail in 256×1600 images
4. **Registry pattern**: UI-ready architecture for model/loss selection
5. **Stratified splitting**: Maintains class distribution in imbalanced data

## 📈 Expected Performance

Training on Severstal dataset (4 defect classes):
- **Class 1 (defect_1)**: ~85% F1
- **Class 2 (defect_2)**: ~15% F1 (rare class - 1.6%)
- **Class 3 (defect_3)**: ~95% F1 (common - 73%)
- **Class 4 (defect_4)**: ~78% F1
- **Macro F1**: ~89%

## 🛠️ Requirements

- Python 3.8+
- PyTorch 2.0+
- CUDA-capable GPU (8GB+ VRAM recommended)
- See `requirements.txt` for full list

## 📝 License

[Add your license here]

## 🙏 Acknowledgments

- Severstal dataset from Kaggle competition
- ConvNext architecture from Facebook Research
- CBAM attention module from "CBAM: Convolutional Block Attention Module" (Woo et al., 2018)
