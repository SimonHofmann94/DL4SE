# Git LFS Setup for Severstal Dataset

## Overview
The Severstal dataset (12,000 images + annotations) is tracked via Git LFS. On RunPod, the setup script automatically:
1. Installs Git LFS (not pre-installed on RunPod)
2. Pulls the Severstal data from Git LFS
3. Organizes it into `data/images/` and `data/annotations/`

## Changes Made

### 1. Updated `.gitattributes`
Added LFS tracking for the Severstal dataset directories:
```
data/Severstal/train/img/ filter=lfs diff=lfs merge=lfs -text
data/Severstal/train/ann/ filter=lfs diff=lfs merge=lfs -text
data/Severstal/test/img/ filter=lfs diff=lfs merge=lfs -text
data/Severstal/test/ann/ filter=lfs diff=lfs merge=lfs -text
```

This ensures large image and annotation files are tracked by Git LFS, not directly in Git.

### 2. Updated `setup_runpod.sh`
Complete redesign for RunPod compatibility:

**Step 1: Install and Configure Git LFS**
- Checks if Git LFS is installed
- Installs it via `apt-get` if needed (RunPod environment)
- Runs `git lfs pull --include="data/Severstal/*"` to download files
- Initializes Git LFS support

**Step 2: Organize Dataset**
- Copies `data/Severstal/train/img/*` → `data/images/`
- Copies `data/Severstal/train/ann/*` → `data/annotations/`
- Creates target directories if needed
- Exits if Severstal data is not found (LFS pull failure)

**Steps 3-6**: Python environment, dependencies, validation, GPU check (unchanged)

## Usage on RunPod

### Initial Setup
```bash
# Clone the repo (shallow clone to save time)
git clone https://github.com/SimonHofmann94/DL4SE.git
cd DL4SE

# Make setup script executable
chmod +x setup_runpod.sh

# Run setup (handles everything automatically)
./setup_runpod.sh
```

The script will:
- ✅ Install Git LFS automatically
- ✅ Download Severstal dataset from LFS
- ✅ Organize files into standard locations
- ✅ Create Python venv and install dependencies
- ✅ Validate system configuration
- ✅ Check GPU availability

### Start Training
```bash
source venv/bin/activate
python code/train.py
```

## Local Development (Your Machine)

### First Time Setup
```bash
# Ensure Git LFS is installed
git lfs install

# Add files to LFS (if not already done)
git lfs track "data/Severstal/train/img/*"
git lfs track "data/Severstal/train/ann/*"

# Commit changes
git add .gitattributes setup_runpod.sh GIT_LFS_SETUP.md
git commit -m "Update: Git LFS for Severstal dataset"

# Add Severstal data to repo
git add data/Severstal/
git commit -m "Add: Severstal dataset tracked via LFS"
```

### Pulling Data Locally
```bash
# Get the actual files (not just pointers)
git lfs pull --include="data/Severstal/*"

# Or run the setup script
chmod +x setup_runpod.sh
./setup_runpod.sh
```

## Data Structure After Setup

```
data/
├── images/                  ← 12k images (copied from Severstal/train/img/)
├── annotations/             ← JSON annotations (copied from Severstal/train/ann/)
├── Severstal/              ← Original Git LFS tracked folder (stays as backup)
│   ├── train/
│   │   ├── img/            ← 12k training images (LFS tracked)
│   │   └── ann/            ← JSON annotations (LFS tracked)
│   └── test/
│       ├── img/            ← Test set (not used for training)
│       └── ann/            ← Test annotations
├── reports/
└── ...
```

## Training Dataset Facts

- **Total images**: 12,000
- **Defective images**: ~6,000 (with defect annotations)
- **No-defect images**: ~6,000
- **Classes**: 5 (no-defect + 4 defect types)
- **Image format**: JPG (8160 × 4624 resolution)
- **Annotation format**: JSON with bitmap masks (base64 encoded)
- **Total storage**: ~30-50 GB (with LFS optimization)

## Troubleshooting

### Git LFS Pull Fails on RunPod
```bash
# Check LFS status
git lfs ls-files

# Force pull
git lfs pull --all

# Or specific files
git lfs pull --include="data/Severstal/train/img/*"
```

### "data/Severstal/train not found" Error
- Ensure `git lfs pull` completed successfully
- Check `git log` shows LFS files were pulled
- Verify `.gitattributes` contains the patterns

### Out of Disk Space on RunPod
- RunPod provides 50-100GB by default
- 12k images ≈ 30-50GB decompressed
- Consider using `--sparse-checkout` if needed:
  ```bash
  git sparse-checkout init --cone
  git sparse-checkout set data/Severstal/train
  ```
