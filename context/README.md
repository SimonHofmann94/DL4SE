# Severstal Steel Defect Detection Project

This is a deep learning project for multi-label steel defect detection based on the Severstal Steel Defect Detection Kaggle dataset.

## 🎯 Project Goal

Develop a robust classification model to detect four types of steel defects in industrial production images:
- **Class 1**: Scratches/cracks (ClassId_1)
- **Class 2**: Surface defects (ClassId_2)
- **Class 3**: Inclusions (ClassId_3)
- **Class 4**: Spots (ClassId_4)

## 📊 Dataset

- **Number of images**: 6,666
- **Image size**: 1600 x 256 pixels (grayscale → converted to RGB)
- **Annotations**: Multi-label format with binary labels per class
- **Class distribution**: Imbalanced, different defect frequencies

### Data layout
```
data/
├── annotations/
│   └── train.csv          # Contains ImageId and EncodedPixels
├── images/
│   └── [6658 .jpg files]  # Production images
├── ann_dmg.zip           # Original archive of annotations
└── img_dmg.zip           # Original archive of images
```

## 🏗️ Project structure

```
DL & SE/
├── code/
│   ├── models.py              # Main model architectures
│   ├── severstal_dataset.py   # Dataset implementation
│   ├── train.py               # Training pipeline
│   ├── config/
│   │   └── train_config.yaml  # Hydra configuration
│   └── additional/            # Test and analysis scripts
│       ├── test_data_quick.py
│       ├── test_dataset.py
│       ├── test_models.py
│       ├── test_severstal_dimensions.py
│       ├── test_efficientnet_simple.py
│       ├── data_analysis.py
│       └── analyze_distribution.py
├── data/                      # Datasets (see above)
├── context/                   # Project documentation
└── README.md                  # This file
```

## 🤖 Model architectures

The project supports multiple CNN backbones for flexible experiments:

### Supported backbones
- **DenseNet121**: Compact, good feature reuse
- **EfficientNet-B0/B1**: Strong efficiency-to-accuracy tradeoff
- **ResNet50**: Proven architecture with skip connections

### Model features
- **Multi-label classification**: Each image can have multiple defect types
- **Pretrained weights**: ImageNet pretrained backbones available
- **Custom classifier head**: Final layers adapted for 4 output classes
- **Flexible backbone selection**: Easy switch between architectures

## 📈 Loss functions

- **Binary Cross-Entropy with Logits**: Main loss for multi-label classification
- **Focal Loss**: Useful for imbalanced classes (α and γ parameters)
- **Positive class weights**: Automatically derived from class distribution when needed

## 🔧 Dataset implementation

### SeverstalFullImageDataset
- **Full-image processing**: Uses the entire 1600x256 images
- **Multi-label support**: Binary labels for all 4 defect classes
- **Augmentations**: Normalization and optional transforms
- **Debug mode**: Limit the number of images for fast tests

### Key features
- Automatic conversion from grayscale to RGB
- Robust handling of missing images/annotations
- Efficient integration with PyTorch DataLoader

## 🧪 Testing & validation

The project includes a comprehensive test suite:

### Data validation
- **test_data_quick.py**: Quick end-to-end sanity checks
- **data_analysis.py**: Detailed data distribution analysis
- **analyze_distribution.py**: Class distribution and statistics

### Dataset tests
- **test_dataset.py**: Validates the SeverstalFullImageDataset implementation
- Verifies image loading, label mapping and transforms

### Model tests
- **test_models.py**: Full tests of all model backbones using real data
- **test_severstal_dimensions.py**: Tests specific to Severstal image dimensions (256x1600)
- **test_efficientnet_simple.py**: Lightweight EfficientNet tests

### All tests passed ✅

## ⚙️ Configuration management

- **Hydra**: Structured configuration management
- **YAML-based**: Readable and version-controllable configs
- **Flexible parameters**: Easy to change model, training and data settings

### Example configuration
```yaml
model:
  backbone: "densenet121"
  num_classes: 4
  pretrained: true

training:
  batch_size: 8
  learning_rate: 0.001
  num_epochs: 50

data:
  image_size: [256, 1600]
  num_workers: 4
```

## 🚀 Getting started

### Requirements
```bash
pip install torch torchvision
pip install timm  # for EfficientNet
pip install hydra-core
pip install pandas numpy pillow
```

### Prepare the data
1. Extract `ann_dmg.zip` into `data/annotations/`
2. Extract `img_dmg.zip` into `data/images/`

### Run tests
```bash
# Quick sanity test
python code/additional/test_data_quick.py

# Full model tests
python code/additional/test_models.py

# Dataset validation
python code/additional/test_dataset.py
```

### Start training
```bash
python code/train.py
```

## 📝 Development history

### Implemented features
- ✅ Multi-backbone model architecture (DenseNet, EfficientNet, ResNet)
- ✅ Robust full-image dataset implementation
- ✅ Comprehensive test-suite for all components
- ✅ Flexible configuration with Hydra
- ✅ Multi-label classification with multiple loss functions
- ✅ Pretrained model integration
- ✅ Data validation and analysis tools

### Validated components
- **Data integrity**: 6,658 exact matches between images and annotations
- **Dataset functionality**: SeverstalFullImageDataset loads images correctly (1600x256)
- **Model backbones**: All supported backbones work with real Severstal data
- **EfficientNet compatibility**: Special tests for EfficientNet with correct dimensions
- **Loss functions**: BCEWithLogitsLoss and Focal Loss implemented and validated

## 🔍 Next steps

- [ ] Hyperparameter tuning for different model backbones
- [ ] More advanced data augmentation strategies
- [ ] Model ensemble techniques
- [ ] Production deployment pipeline
- [ ] Metric logging and visualization

## 📚 Technical details

- **Framework**: PyTorch
- **Python version**: 3.8+
- **GPU support**: CUDA-compatible
- **Image processing**: PIL, torchvision.transforms
- **Annotation format**: CSV-based with run-length encoded pixels

---

*Project developed for industrial steel defect detection using modern deep learning techniques.*