# 🚀 Quick Start with kagglehub

## What Changed?

We've switched from the old `kaggle` API to the newer `kagglehub` library because:

✅ **Simpler** - No need to manually accept dataset terms on the website  
✅ **More reliable** - Fewer 403 errors  
✅ **Automatic fallback** - Tries multiple datasets if one fails  
✅ **Better caching** - Faster re-downloads  

---

## Step-by-Step Setup (15 minutes)

### 1. Update Dependencies

```bash
# Make sure you're in the project directory and venv is activated
source venv/bin/activate

# Run the update script
./update_dependencies.sh

# OR manually:
pip install --upgrade kagglehub
```

### 2. Verify Kaggle Credentials

```bash
# Check if kaggle.json exists
ls -la ~/.kaggle/kaggle.json

# If not found, get it from Kaggle:
# 1. Go to https://www.kaggle.com/account
# 2. Click "Create New API Token"
# 3. Move kaggle.json to ~/.kaggle/
# 4. Set permissions: chmod 600 ~/.kaggle/kaggle.json
```

### 3. Download Dataset

```bash
# This will try two datasets automatically
python data/download_kaggle.py
```

**What it does:**
- First tries: `devdope/900k-spotify`
- If that fails, tries: `xingweizhao/900k-spotify-songs-with-lyrics-emotions-more`
- Copies files to `data/raw/`

### 4. Verify Download

```bash
# Check what was downloaded
ls -lh data/raw/

# You should see CSV files (100+ MB)
```

### 5. Continue with Preprocessing

```bash
python data/preprocess.py
```

---

## Differences from Old API

| Feature | Old `kaggle` API | New `kagglehub` |
|---------|-----------------|-----------------|
| Installation | `pip install kaggle` | `pip install kagglehub` |
| Accept terms | ✋ Required (manual) | ✅ Automatic |
| Error handling | ❌ Fails on 403 | ✅ Tries alternatives |
| Cache location | `~/.kaggle/` | `~/.cache/kagglehub/` |
| Code complexity | More verbose | Simpler |

---

## Troubleshooting

### Issue: kagglehub not found

```bash
pip install --upgrade kagglehub
```

### Issue: Still getting 403 errors

```bash
# This shouldn't happen with kagglehub, but if it does:
# 1. Make sure kaggle.json has correct format:
cat ~/.kaggle/kaggle.json

# Should look like:
# {"username":"YOUR_USERNAME","key":"YOUR_API_KEY"}

# 2. Verify permissions:
chmod 600 ~/.kaggle/kaggle.json

# 3. Try manual login:
# Go to the dataset page and just view it:
# https://www.kaggle.com/datasets/devdope/900k-spotify
```

### Issue: Download is slow

```bash
# kagglehub caches downloads, so:
# 1. First download: 5-15 minutes (depending on internet)
# 2. Subsequent runs: Very fast (uses cache)

# The script copies from cache to data/raw/ automatically
```

### Issue: Dataset not found

```bash
# If both datasets fail, the script will show alternatives
# You can also manually specify a dataset by editing:
# data/download_kaggle.py

# Change line 47-50 to use a different dataset
```

---

## Files Changed

### ✅ Updated
- `requirements.txt` - Changed `kaggle` to `kagglehub`
- `data/download_kaggle.py` - Complete rewrite using kagglehub
- `config.py` - Removed hardcoded dataset name
- `check_setup.py` - Updated to check for kagglehub
- `README.md` - Updated troubleshooting section

### ❌ Deleted (no longer needed)
- `fix_kaggle.py` - Kagglehub doesn't need this
- `data/manual_download_instructions.md` - Simpler now

### ➕ New
- `update_dependencies.sh` - Easy dependency update script
- `QUICKSTART_KAGGLEHUB.md` - This file!

---

## Ready to Go!

Now you can run:

```bash
# 1. Update dependencies
./update_dependencies.sh

# 2. Download data (no manual steps needed!)
python data/download_kaggle.py

# 3. Preprocess
python data/preprocess.py

# 4. Train models
python train.py --model lyrics_only --epochs 3
python train.py --model audio_only --epochs 3
python train.py --model multimodal --epochs 3

# 5. Compare
python compare_models.py
```

**That's it! Much simpler than before! 🎉**

