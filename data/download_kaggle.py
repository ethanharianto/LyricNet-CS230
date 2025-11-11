"""
Download Kaggle dataset for LyricNet project using kagglehub.

Setup:
1. Create Kaggle account: https://www.kaggle.com
2. Go to Account -> API -> Create New API Token
3. This downloads kaggle.json
4. Place kaggle.json in ~/.kaggle/ directory

Note: kagglehub is easier than the old kaggle API.

Usage:
    python data/download_kaggle.py
"""

import os
import sys
import shutil
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from config import RAW_DATA_DIR


def download_kaggle_dataset():
    """Download the Spotify dataset from Kaggle using kagglehub."""
    
    # Check if kaggle credentials exist
    kaggle_dir = Path.home() / ".kaggle"
    kaggle_json = kaggle_dir / "kaggle.json"
    
    if not kaggle_json.exists() and not (os.getenv("KAGGLE_USERNAME") and os.getenv("KAGGLE_KEY")):
        print("ERROR: Kaggle credentials not found!")
        print("\nSetup instructions:")
        print("1. Go to https://www.kaggle.com/account")
        print("2. Click 'Create New API Token'")
        print("3. Move downloaded kaggle.json to ~/.kaggle/")
        print("4. Run: chmod 600 ~/.kaggle/kaggle.json")
        return False
    
    try:
        import kagglehub
        
        # Try multiple dataset options (in case one is unavailable)
        datasets_to_try = [
            ("devdope/900k-spotify", "DevDope 900K Spotify"),
            ("xingweizhao/900k-spotify-songs-with-lyrics-emotions-more", "XingWei Zhao 900K Spotify")
        ]
        
        downloaded = False
        for dataset_id, dataset_name in datasets_to_try:
            try:
                print(f"\nAttempting to download: {dataset_name}")
                print(f"   Dataset ID: {dataset_id}")
                
                # Download latest version
                download_path = kagglehub.dataset_download(dataset_id)
                
                print(f"Dataset downloaded successfully!")
                print(f"   Downloaded to: {download_path}")
                
                # Create raw data directory
                os.makedirs(RAW_DATA_DIR, exist_ok=True)
                
                # Copy files from kagglehub cache to our data/raw directory
                print(f"\nCopying files to {RAW_DATA_DIR}...")
                
                for file in os.listdir(download_path):
                    src = os.path.join(download_path, file)
                    dst = os.path.join(RAW_DATA_DIR, file)
                    
                    if os.path.isfile(src):
                        shutil.copy2(src, dst)
                        size_mb = os.path.getsize(src) / (1024 * 1024)
                        print(f"   {file} ({size_mb:.2f} MB)")
                
                downloaded = True
                break  # Success! Exit the loop
                
            except Exception as e:
                print(f"   WARNING: Failed - {e}")
                continue
        
        if not downloaded:
            print("\nERROR: All dataset download attempts failed")
            print("\nAlternative: Download manually from:")
            print("   https://www.kaggle.com/datasets/devdope/900k-spotify")
            print("   or")
            print("   https://www.kaggle.com/datasets/xingweizhao/900k-spotify-songs-with-lyrics-emotions-more")
            return False
        
        # List all files in raw directory
        print("\nFiles in data/raw/:")
        total_size = 0
        for file in os.listdir(RAW_DATA_DIR):
            file_path = os.path.join(RAW_DATA_DIR, file)
            if os.path.isfile(file_path):
                size_mb = os.path.getsize(file_path) / (1024 * 1024)
                total_size += size_mb
                print(f"   - {file} ({size_mb:.2f} MB)")
        
        print(f"\nTotal size: {total_size:.2f} MB")
        
        return True
        
    except Exception as e:
        print(f"ERROR: Error downloading dataset - {e}")
        print("\nMake sure you have kagglehub installed:")
        print("   pip install kagglehub")
        return False


if __name__ == "__main__":
    print("=" * 60)
    print("LyricNet - Kaggle Dataset Downloader")
    print("=" * 60)
    
    success = download_kaggle_dataset()
    
    if success:
        print("\nNext step: Run data/preprocess.py to prepare the data")
    else:
        print("\nPlease fix the errors above and try again")

