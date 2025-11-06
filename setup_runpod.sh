#!/bin/bash
# ============================================================================
# RunPod Setup Script for Severstal Defect Detection Training
# ============================================================================
# PURPOSE:
#   - Extract training data from Git LFS zip files
#   - Set up Python virtual environment
#   - Install all dependencies
#   - Validate system is ready for training
#   - Check GPU availability
#
# USAGE:
#   After cloning repo on RunPod:
#     chmod +x setup_runpod.sh
#     ./setup_runpod.sh
# ============================================================================

set -e  # Exit immediately if any command fails (safety feature)

echo "=================================="
echo "RunPod Environment Setup"
echo "=================================="

# SAFETY CHECK: Ensure we're in the project root
# (requirements.txt exists in root, so this confirms correct location)
if [ ! -f "requirements.txt" ]; then
    echo "Error: Run this script from the project root directory"
    exit 1
fi

# ============================================================================
# STEP 1: Extract Data from Zips
# ============================================================================
# WHY: Git LFS stores large files (images) as zips to reduce bandwidth
# WHAT: Extracts zips into proper directories for training
# ============================================================================
echo ""
echo "Step 1: Extracting data from zips..."
echo "-----------------------------------"

# Create target directories (these were tracked with .gitkeep in git)
# -p flag: create parent directories if needed, don't error if exists
mkdir -p data/images
mkdir -p data/Severstal/train_images
mkdir -p data/Severstal/test_images

# Check if zip directory exists (downloaded from Git LFS)
if [ -d "data/zips" ]; then
    echo "Extracting files from data/zips/..."
    
    # Loop through all zip files in data/zips/
    # CUSTOMIZE THIS: Adjust extraction path (-d flag) based on your zip structure
    for zipfile in data/zips/*.zip; do
        if [ -f "$zipfile" ]; then
            echo "  Extracting: $(basename $zipfile)"
            # unzip flags:
            #   -q : quiet mode (less output)
            #   -d : destination directory
            unzip -q "$zipfile" -d data/
        fi
    done
    
    echo "✓ Data extraction complete"
else
    echo "⚠ Warning: data/zips/ directory not found"
    echo "   Make sure Git LFS downloaded the zip files"
fi

# ============================================================================
# STEP 2: Create Python Virtual Environment
# ============================================================================
# WHY: Isolate dependencies from system Python (avoid conflicts)
# WHAT: Creates a venv/ directory with isolated Python environment
# ============================================================================
echo ""
echo "Step 2: Setting up Python environment..."
echo "-----------------------------------"

# Create virtual environment named "venv"
# python3 -m venv: Built-in venv module
python3 -m venv venv

# Activate the virtual environment (for this script session)
# All subsequent pip/python commands will use this venv
source venv/bin/activate

echo "✓ Virtual environment created"

# ============================================================================
# STEP 3: Install Python Dependencies
# ============================================================================
# WHY: Training requires PyTorch, torchvision, and other packages
# WHAT: Installs all packages listed in requirements.txt
# ============================================================================
echo ""
echo "Step 3: Installing dependencies..."
echo "-----------------------------------"

# Upgrade pip to latest version (ensures compatibility)
pip install --upgrade pip

# Install all project dependencies
# This includes: PyTorch, torchvision, hydra, numpy, PIL, etc.
pip install -r requirements.txt

echo "✓ Dependencies installed"

# ============================================================================
# STEP 4: Validate System
# ============================================================================
# WHY: Catch any import/configuration errors before training starts
# WHAT: Runs validate_system.py which tests:
#       - All imports work
#       - Model can be instantiated
#       - Loss functions work
#       - Data pipeline works
#       - Augmentations work
# ============================================================================
echo ""
echo "Step 4: Verifying system..."
echo "-----------------------------------"

python code/validate_system.py

# ============================================================================
# STEP 5: GPU Check
# ============================================================================
# WHY: Confirm CUDA is available and see what GPU we have
# WHAT: Prints GPU info (device count, name, CUDA availability)
# ============================================================================
echo ""
echo "Step 5: GPU Check..."
echo "-----------------------------------"

python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'CUDA devices: {torch.cuda.device_count()}'); print(f'Device name: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"No GPU\"}')"

# ============================================================================
# COMPLETION MESSAGE
# ============================================================================
echo ""
echo "=================================="
echo "✓ Setup Complete!"
echo "=================================="
echo ""
echo "You can now start training:"
echo "  python code/train.py"
echo ""
echo "Or run a quick test:"
echo "  python code/train.py training.num_epochs=5"
echo ""
