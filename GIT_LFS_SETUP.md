# Git LFS and GitHub Setup Guide

This guide will help you set up the repository with Git LFS and push to GitHub for use on RunPod.

## Prerequisites

Install Git LFS if you haven't already:
```bash
# Windows (using Git for Windows - already includes LFS)
git lfs install

# Or download from: https://git-lfs.github.com/
```

## Step 1: Initialize Git Repository

```bash
# Initialize git (if not already done)
git init

# Install Git LFS in this repo
git lfs install
```

## Step 2: Verify LFS is Tracking Zip Files

```bash
# Check what LFS is tracking
git lfs track

# Should show:
# Listing tracked patterns
#     *.zip (.gitattributes)
#     *.pt (.gitattributes)
#     *.pth (.gitattributes)
#     *.ckpt (.gitattributes)
```

## Step 3: Add Files to Git

```bash
# Add all files
git add .

# Check which files will be tracked by LFS
git lfs ls-files

# Commit
git commit -m "Initial commit: ConvNext-Tiny with CBAM for Severstal defect detection"
```

## Step 4: Create GitHub Repository

1. Go to https://github.com/new
2. Create a new repository (e.g., `severstal-convnext-cbam`)
3. **Do NOT initialize with README** (you already have files)
4. Copy the repository URL

## Step 5: Push to GitHub

```bash
# Add remote (replace with your repo URL)
git remote add origin https://github.com/YOUR_USERNAME/severstal-convnext-cbam.git

# Push to GitHub
git branch -M main
git push -u origin main

# Git LFS will automatically upload the large files
```

**Note:** If you have large zip files, the push might take a while. Git LFS will show progress.

## Step 6: Verify on GitHub

1. Go to your GitHub repository
2. Check that code files are visible
3. Click on `data/zips/` - you should see LFS pointers for zip files
4. Files should show "Stored with Git LFS" badge

## Step 7: Clone on RunPod

Once on RunPod instance:

```bash
# Clone repo (Git LFS will automatically download LFS files)
git clone https://github.com/YOUR_USERNAME/severstal-convnext-cbam.git

# Navigate to directory
cd severstal-convnext-cbam

# Make setup script executable
chmod +x setup_runpod.sh

# Run setup script
./setup_runpod.sh
```

The setup script will:
- Extract data from zips
- Create virtual environment
- Install dependencies
- Validate the system
- Check GPU availability

## Troubleshooting

### If LFS files didn't upload:
```bash
# Manually push LFS files
git lfs push --all origin main
```

### If zip files are too large for GitHub LFS:
GitHub LFS has limits (1GB per file on free tier, 2GB bandwidth per month).

**Alternative: Use RunPod's direct upload**
1. Don't track zips with LFS (remove from .gitattributes)
2. Upload zips directly to RunPod via their interface
3. Or use a cloud storage link (Google Drive, Dropbox, S3)

### Check LFS quota:
```bash
git lfs ls-files -s  # Show size of LFS files
```

## Quick Commands Reference

```bash
# Check what's tracked by LFS
git lfs ls-files

# See LFS file details
git lfs ls-files -s

# Migrate existing files to LFS
git lfs migrate import --include="*.zip"

# Check LFS status
git lfs status
```

## Alternative: Skip LFS if Files Too Large

If your zips exceed GitHub LFS limits, modify `.gitattributes`:

```bash
# Comment out zip line in .gitattributes
# *.zip filter=lfs diff=lfs merge=lfs -text
```

Then add to `.gitignore`:
```
data/zips/*.zip
```

And manually upload zips to RunPod or use cloud storage download in setup script.
