# Step-by-Step: Push to GitHub

## Step 1: Create GitHub Repository

1. Go to https://github.com/new
2. Repository name: `severstal-convnext-cbam` (or your preferred name)
3. Description: "ConvNext-Tiny with CBAM for Severstal steel defect detection"
4. Choose Public or Private
5. **IMPORTANT**: Do NOT check "Initialize this repository with a README"
6. Click "Create repository"

You'll see a page with commands. Copy the repository URL (HTTPS or SSH).
Example: `https://github.com/YOUR_USERNAME/severstal-convnext-cbam.git`

---

## Step 2: Initialize Git LFS (One-time setup)

Run this in PowerShell (in your project directory):

```powershell
git lfs install
```

---

## Step 3: Initialize Git Repository

```powershell
git init
```

---

## Step 4: Add Remote

Replace `YOUR_REPO_URL` with your actual GitHub repo URL:

```powershell
git remote add origin YOUR_REPO_URL
```

Example:
```powershell
git remote add origin https://github.com/YOUR_USERNAME/severstal-convnext-cbam.git
```

---

## Step 5: Add All Files to Git

```powershell
git add .
```

Verify what will be committed:
```powershell
git status
```

---

## Step 6: Commit

```powershell
git commit -m "Initial commit: ConvNext-Tiny with CBAM for Severstal defect detection"
```

---

## Step 7: Push to GitHub

```powershell
git branch -M main
git push -u origin main
```

This will:
- Rename branch to `main`
- Push all files
- **Git LFS will automatically handle the 600MB zip file**
- Set upstream tracking (next push just needs `git push`)

---

## Verification

1. Go to your GitHub repo URL
2. Check that files are visible
3. Click on `data/zips/` folder
4. You should see your zip files with "Stored with Git LFS" badge

---

## Troubleshooting

### If you get "Git is not recognized"
- Install Git: https://git-scm.com/download/win
- Restart PowerShell

### If you get "Permission denied"
- You may need SSH keys or personal access token
- See: https://docs.github.com/en/authentication

### If LFS files didn't upload
```powershell
git lfs push --all origin main
```

### If you made a mistake
```powershell
git reset --soft HEAD~1  # Undo last commit but keep changes
git reset HEAD file.txt  # Unstage a file
```

---

## What Happens During Push

1. ✅ Code files uploaded normally (~30 sec)
2. ✅ Git LFS takes over for `*.zip` files (5-10 min depending on internet)
3. ✅ All other files uploaded

**Total time**: 5-15 minutes

---

## Ready?

Run these commands in order:

```powershell
git lfs install
git init
git remote add origin YOUR_REPO_URL
git add .
git commit -m "Initial commit: ConvNext-Tiny with CBAM for Severstal defect detection"
git branch -M main
git push -u origin main
```

Need help with any step?
