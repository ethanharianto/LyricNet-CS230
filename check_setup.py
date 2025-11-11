"""
Check if environment is set up correctly for LyricNet.

Usage:
    python check_setup.py
"""

import sys
import subprocess
import os

def check_python_version():
    """Check Python version."""
    print("1. Checking Python version...")
    version = sys.version_info
    print(f"   Python {version.major}.{version.minor}.{version.micro}")
    
    if version.major >= 3 and version.minor >= 8:
        print("   Python version OK")
        return True
    else:
        print("   ERROR: Python 3.8+ required")
        return False


def check_package(package_name, import_name=None):
    """Check if a package is installed."""
    if import_name is None:
        import_name = package_name
    
    try:
        __import__(import_name)
        print(f"   {package_name}")
        return True
    except ImportError:
        print(f"   ERROR: {package_name} not installed")
        return False


def check_dependencies():
    """Check all required packages."""
    print("\n2. Checking dependencies...")
    
    packages = [
        ('torch', 'torch'),
        ('transformers', 'transformers'),
        ('pandas', 'pandas'),
        ('numpy', 'numpy'),
        ('scikit-learn', 'sklearn'),
        ('matplotlib', 'matplotlib'),
        ('seaborn', 'seaborn'),
        ('tqdm', 'tqdm'),
        ('kagglehub', 'kagglehub'),
    ]
    
    all_installed = True
    for package, import_name in packages:
        if not check_package(package, import_name):
            all_installed = False
    
    if not all_installed:
        print("\n   Install missing packages with:")
        print("   pip install -r requirements.txt")
    
    return all_installed


def check_torch_device():
    """Check available PyTorch devices."""
    print("\n3. Checking PyTorch devices...")
    
    try:
        import torch
        
        print(f"   PyTorch version: {torch.__version__}")
        
        # Check CUDA
        if torch.cuda.is_available():
            print(f"   CUDA available: {torch.cuda.get_device_name(0)}")
        else:
            print("   INFO: CUDA not available")
        
        # Check MPS (Apple Silicon)
        if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            print("   MPS (Apple Silicon GPU) available")
        else:
            print("   INFO: MPS not available")
        
        if not torch.cuda.is_available() and not (hasattr(torch.backends, 'mps') and torch.backends.mps.is_available()):
            print("   WARNING: Will use CPU (training will be slow)")
        
        return True
    except ImportError:
        print("   ERROR: PyTorch not installed")
        return False


def check_kaggle_credentials():
    """Check Kaggle API credentials."""
    print("\n4. Checking Kaggle credentials...")
    
    kaggle_dir = os.path.expanduser("~/.kaggle")
    kaggle_json = os.path.join(kaggle_dir, "kaggle.json")
    
    if os.path.exists(kaggle_json):
        print(f"   Found kaggle.json at {kaggle_json}")
        
        # Check permissions
        import stat
        st = os.stat(kaggle_json)
        if st.st_mode & stat.S_IROTH:
            print("   WARNING: kaggle.json has wrong permissions")
            print(f"   Run: chmod 600 {kaggle_json}")
        else:
            print("   Permissions OK")
        
        return True
    else:
        print(f"   ERROR: kaggle.json not found at {kaggle_json}")
        print("\n   Setup instructions:")
        print("   1. Go to https://www.kaggle.com/account")
        print("   2. Click 'Create New API Token'")
        print("   3. Move kaggle.json to ~/.kaggle/")
        print("   4. Run: chmod 600 ~/.kaggle/kaggle.json")
        return False


def check_directories():
    """Check if required directories exist."""
    print("\n5. Checking project directories...")
    
    dirs = [
        'data',
        'data/raw',
        'data/processed',
        'models',
        'models/checkpoints',
        'logs',
        'notebooks'
    ]
    
    all_exist = True
    for directory in dirs:
        if os.path.exists(directory):
            print(f"   {directory}/")
        else:
            print(f"   INFO: {directory}/ not found (will be created when needed)")
    
    return True


def check_data():
    """Check if data is downloaded and processed."""
    print("\n6. Checking data...")
    
    # Check raw data
    raw_files = []
    if os.path.exists('data/raw'):
        raw_files = [f for f in os.listdir('data/raw') if f.endswith('.csv')]
    
    if raw_files:
        print(f"   Raw data found: {raw_files[0]}")
    else:
        print("   INFO: Raw data not downloaded yet")
        print("   Run: python data/download_kaggle.py")
    
    # Check processed data
    processed_files = ['train.csv', 'val.csv', 'test.csv', 'label_mappings.json']
    all_processed = True
    
    if os.path.exists('data/processed'):
        for file in processed_files:
            file_path = os.path.join('data/processed', file)
            if os.path.exists(file_path):
                print(f"   {file}")
            else:
                print(f"   INFO: {file} not found")
                all_processed = False
    else:
        print("   INFO: Processed data not ready")
        all_processed = False
    
    if not all_processed:
        print("   Run: python data/preprocess.py")
    
    return raw_files and all_processed


def main():
    """Run all environment checks."""
    print("="*70)
    print("LyricNet - Environment Setup Check")
    print("="*70)
    
    checks = [
        check_python_version(),
        check_dependencies(),
        check_torch_device(),
        check_kaggle_credentials(),
        check_directories(),
    ]
    
    # Data check is informational only
    check_data()
    
    print("\n" + "="*70)
    if all(checks):
        print("Environment setup complete!")
        print("="*70)
        print("\nNext steps:")
        print("1. python data/download_kaggle.py     # Download dataset")
        print("2. python data/preprocess.py          # Preprocess data")
        print("3. python train.py --model [MODEL]    # Train models")
        print("4. python evaluate.py --model [MODEL] # Evaluate models")
    else:
        print("Some checks failed")
        print("="*70)
        print("\nPlease fix the issues above and run this script again.")
    
    print("\n" + "="*70)


if __name__ == "__main__":
    main()

