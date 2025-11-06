# Git + RunPod Workflow Summary

## The Problem
Your laptop doesn't have a strong GPU, so you need to train on RunPod cloud instances.

## The Solution
1. **Store code on GitHub** (easy access from anywhere)
2. **Use Git LFS for large files** (zip files with training images)
3. **Auto-setup on RunPod** (one script to prepare everything)

---

## File Categories

### ✅ Tracked in Git (Normal)
- **Python code**: All `.py` files in `code/`
- **Config files**: `config/train_config.yaml`
- **Documentation**: All `.md` files
- **Annotations**: `data/annotations/*.json` (small JSON files)
- **Directory placeholders**: `.gitkeep` files

### 📦 Tracked with Git LFS (Large File Storage)
- **Data archives**: `data/zips/*.zip` (training images)
- **Model checkpoints**: `*.pt`, `*.pth`, `*.ckpt` (if you save them to git)

### ❌ NOT Tracked (Ignored by `.gitignore`)
- **Virtual environment**: `venv/`
- **Python cache**: `__pycache__/`
- **Extracted images**: `data/images/*` (extracted from zips on RunPod)
- **Training outputs**: `code/experiments/results/*/`
- **IDE files**: `.vscode/`, `.idea/`

---

## Why This Setup?

### `.gitkeep` Files
Git doesn't track empty directories. But we need `data/images/` to exist for the extraction script.

**Solution**: Put a `.gitkeep` placeholder file in empty directories.

- `data/images/` → **Empty** → Needs `.gitkeep`
- `data/annotations/` → **Has JSON files** → Doesn't need `.gitkeep`

### Git LFS for Zips
GitHub limits regular files to 100MB. Training images are much larger.

**Solution**: Git LFS stores large files separately and downloads them on-demand.

---

## `setup_runpod.sh` - What It Does

This script runs **after cloning** on RunPod to prepare the environment:

### Step 1: Extract Data (30 sec - 2 min)
```bash
# Finds all zips in data/zips/
# Extracts them to data/ directory
unzip -q "data/zips/*.zip" -d data/
```

**Why**: Git LFS downloads zips, but training needs extracted images

### Step 2: Virtual Environment (10 sec)
```bash
# Creates isolated Python environment
python3 -m venv venv
source venv/bin/activate
```

**Why**: Keeps dependencies separate from system Python

### Step 3: Install Dependencies (2-5 min)
```bash
# Installs PyTorch, torchvision, hydra, etc.
pip install -r requirements.txt
```

**Why**: Training code needs these packages

### Step 4: Validate System (5 sec)
```bash
# Runs code/validate_system.py
python code/validate_system.py
```

**Why**: Catches any setup errors before training starts
- Tests imports
- Tests model instantiation
- Tests data loading
- Tests GPU availability

### Step 5: GPU Check (1 sec)
```bash
# Prints GPU info
torch.cuda.is_available()  # True or False
torch.cuda.get_device_name(0)  # e.g., "NVIDIA RTX 4090"
```

**Why**: Confirms you have GPU access and can use CUDA

---

## Complete Workflow

### On Your Laptop (Once)

```bash
# 1. Initialize Git with LFS
git init
git lfs install

# 2. Add all files
git add .

# 3. Commit
git commit -m "Initial commit"

# 4. Create GitHub repo and push
git remote add origin https://github.com/YOUR_USERNAME/YOUR_REPO.git
git push -u origin main
```

### On RunPod (Each Time You Start Instance)

```bash
# 1. Clone repo (Git LFS auto-downloads zips)
git clone https://github.com/YOUR_USERNAME/YOUR_REPO.git
cd YOUR_REPO

# 2. Run setup script (one command does everything!)
chmod +x setup_runpod.sh
./setup_runpod.sh

# 3. Start training
python code/train.py
```

---

## Key Benefits

✅ **One-time setup**: Push code once, clone anywhere  
✅ **Automated**: `setup_runpod.sh` handles all setup steps  
✅ **Validated**: System checks pass before training starts  
✅ **Reproducible**: Same environment every time  
✅ **Version controlled**: Track code changes with git  

---

## Troubleshooting

### If Git LFS files don't download:
```bash
git lfs pull
```

### If zip extraction fails:
Check zip file structure - you may need to adjust the extraction path in `setup_runpod.sh`:
```bash
# Current:
unzip -q "$zipfile" -d data/

# If zips contain a nested folder:
unzip -q "$zipfile" -d data/images/
```

### If validation fails:
- Check error message from `validate_system.py`
- Common issues: Missing dependencies, wrong Python version
- Solution: Run `pip install -r requirements.txt` again

### If GPU not detected:
- Check RunPod instance has GPU enabled
- Run: `nvidia-smi` to see GPU status
- May need to restart kernel/instance

---

## Quick Reference

| Task | Command |
|------|---------|
| Setup on RunPod | `./setup_runpod.sh` |
| Train (default) | `python code/train.py` |
| Quick test | `python code/train.py training.num_epochs=5` |
| Check GPU | `nvidia-smi` |
| Validate system | `python code/validate_system.py` |
| Update code from git | `git pull` |
