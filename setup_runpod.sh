#!/bin/bash
# ============================================================================
# RunPod Setup Script for Severstal Defect Detection Training
# ============================================================================
# PURPOSE:
#   - Install Git LFS (required on RunPod)
#   - Pull Severstal dataset from Git LFS
#   - Organize data into data/images and data/annotations
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
# STEP 1: Install and Configure Git LFS
# ============================================================================
# WHY: RunPod doesn't have Git LFS pre-installed, need it to download large files
# WHAT: Installs Git LFS and pulls Severstal dataset from LFS storage
# ============================================================================
echo ""
echo "Step 1: Setting up Git LFS..."
echo "-----------------------------------"

# Check if git lfs is already installed
if ! command -v git-lfs &> /dev/null; then
    echo "Git LFS not found. Installing..."
    apt-get update -qq
    apt-get install -y git-lfs > /dev/null 2>&1
    echo "✓ Git LFS installed"
else
    echo "✓ Git LFS already installed"
fi

# Initialize Git LFS (required to enable LFS support in this repo)
git lfs install --skip-repo > /dev/null 2>&1

# Pull all Git LFS files (downloads the actual large files instead of pointers)
echo "Pulling Git LFS files (Severstal dataset)..."
git lfs pull --include="data/Severstal/*" 2>&1 | grep -E "(Downloading|Fetched|Downloaded)" || echo "  ✓ LFS files ready"

echo "✓ Git LFS setup complete"

# ============================================================================
# STEP 2: Extract and Organize Severstal Dataset
# ============================================================================
# WHY: Severstal data needs to be organized into data/images and data/annotations
# WHAT: Copies Severstal/train data to standard training directories
# ============================================================================
echo ""
echo "Step 2: Organizing Severstal dataset..."
echo "-----------------------------------"

# Create target directories
mkdir -p data/images
mkdir -p data/annotations

# Check if Severstal training data exists (should exist after git lfs pull)
if [ -d "data/Severstal/train" ]; then
    echo "Setting up Severstal training data..."
    
    # Copy training images to data/images
    if [ -d "data/Severstal/train/img" ]; then
        echo "  Copying Severstal/train/img → data/images/"
        cp -r data/Severstal/train/img/* data/images/ 2>/dev/null || true
        echo "  ✓ Images copied"
    else
        echo "  ⚠ Warning: data/Severstal/train/img/ not found"
    fi
    
    # Copy training annotations to data/annotations
    if [ -d "data/Severstal/train/ann" ]; then
        echo "  Copying Severstal/train/ann → data/annotations/"
        cp -r data/Severstal/train/ann/* data/annotations/ 2>/dev/null || true
        echo "  ✓ Annotations copied"
    else
        echo "  ⚠ Warning: data/Severstal/train/ann/ not found"
    fi
    
    echo "✓ Severstal dataset organization complete"
else
    echo "⚠ Warning: data/Severstal/ directory not found"
    echo "   This likely means Git LFS pull didn't retrieve the files"
    exit 1
fi

# ============================================================================
# STEP 3: Create Python Virtual Environment
# ============================================================================
# WHY: Isolate dependencies from system Python (avoid conflicts)
# WHAT: Creates a venv/ directory with isolated Python environment
# ============================================================================
echo ""
echo "Step 3: Setting up Python environment..."
echo "-----------------------------------"

# Create virtual environment named "venv"
# python3 -m venv: Built-in venv module
python3 -m venv venv

# Activate the virtual environment (for this script session)
# All subsequent pip/python commands will use this venv
source venv/bin/activate

echo "✓ Virtual environment created"

# ============================================================================
# STEP 4: Install Python Dependencies
# ============================================================================
# WHY: Training requires PyTorch, torchvision, and other packages
# WHAT: Installs all packages listed in requirements.txt
# ============================================================================
echo ""
echo "Step 4: Installing dependencies..."
echo "-----------------------------------"

# Upgrade pip to latest version (ensures compatibility)
pip install --upgrade pip

# Install all project dependencies
# This includes: PyTorch, torchvision, hydra, numpy, PIL, etc.
pip install -r requirements.txt

echo "✓ Dependencies installed"

# ============================================================================
# STEP 5: Validate System
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
echo "Step 5: Verifying system..."
echo "-----------------------------------"

python code/validate_system.py

# ============================================================================
# STEP 6: GPU Check
# ============================================================================
# WHY: Confirm CUDA is available and see what GPU we have
# WHAT: Prints GPU info (device count, name, CUDA availability)
# ============================================================================
echo ""
echo "Step 6: GPU Check..."
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
echo "IMPORTANT: Activate the virtual environment first:"
echo "  source venv/bin/activate"
echo ""
echo "Then you can start training:"
echo "  python code/train.py"
echo ""
echo "Or run a quick test:"
echo "  python code/train.py training.num_epochs=5"
echo ""
