"""
Create a smaller subset of data for faster MVP training.

This creates a 50K sample subset (approximately 10% of full dataset)
while maintaining class balance.

Usage:
    python create_small_dataset.py
"""

import os
import json
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

PROCESSED_DATA_DIR = "data/processed"
SMALL_DATA_DIR = "data/processed_small"
SUBSET_SIZE = 50000  # Total samples for small dataset

def create_small_subset():
    """Create balanced small subset."""
    print("="*70)
    print("Creating Small Dataset for Fast MVP Training")
    print("="*70)
    
    # Load full dataset
    print("\nLoading full dataset...")
    train_df = pd.read_csv(os.path.join(PROCESSED_DATA_DIR, 'train.csv'))
    train_audio = np.load(os.path.join(PROCESSED_DATA_DIR, 'train_audio.npy'))
    
    print(f"   Full dataset: {len(train_df):,} samples")
    
    # Load label mappings
    with open(os.path.join(PROCESSED_DATA_DIR, 'label_mappings.json'), 'r') as f:
        mappings = json.load(f)
    
    # Sample with stratification (maintain class balance)
    print(f"\nCreating subset of {SUBSET_SIZE:,} samples...")
    
    if len(train_df) > SUBSET_SIZE:
        # Sample proportionally from each class
        sampled_indices = []
        for label in train_df['labels'].unique():
            label_indices = train_df[train_df['labels'] == label].index.tolist()
            n_samples = int(len(label_indices) * (SUBSET_SIZE / len(train_df)))
            sampled = np.random.choice(label_indices, size=min(n_samples, len(label_indices)), replace=False)
            sampled_indices.extend(sampled)
        
        # If we didn't get enough samples, add random ones
        if len(sampled_indices) < SUBSET_SIZE:
            remaining = SUBSET_SIZE - len(sampled_indices)
            all_indices = set(train_df.index)
            available = list(all_indices - set(sampled_indices))
            additional = np.random.choice(available, size=min(remaining, len(available)), replace=False)
            sampled_indices.extend(additional)
        
        # Subset the data
        sampled_indices = sorted(sampled_indices[:SUBSET_SIZE])
        train_df_small = train_df.iloc[sampled_indices].reset_index(drop=True)
        train_audio_small = train_audio[sampled_indices]
    else:
        train_df_small = train_df
        train_audio_small = train_audio
    
    print(f"   Small dataset: {len(train_df_small):,} samples")
    
    # Show class distribution
    print("\nClass distribution in small dataset:")
    for label in sorted(train_df_small['labels'].unique()):
        count = (train_df_small['labels'] == label).sum()
        emotion = mappings['id_to_emotion'][str(label)]
        print(f"   {emotion}: {count:,} samples ({count/len(train_df_small)*100:.1f}%)")
    
    # Create small processed directory
    os.makedirs(SMALL_DATA_DIR, exist_ok=True)
    
    # Save small training set
    train_df_small.to_csv(os.path.join(SMALL_DATA_DIR, 'train.csv'), index=False)
    np.save(os.path.join(SMALL_DATA_DIR, 'train_audio.npy'), train_audio_small)
    
    # Use original val/test sets (they're already small enough)
    print("\nUsing original val/test sets...")
    
    # Copy val set
    val_df = pd.read_csv(os.path.join(PROCESSED_DATA_DIR, 'val.csv'))
    val_audio = np.load(os.path.join(PROCESSED_DATA_DIR, 'val_audio.npy'))
    val_df.to_csv(os.path.join(SMALL_DATA_DIR, 'val.csv'), index=False)
    np.save(os.path.join(SMALL_DATA_DIR, 'val_audio.npy'), val_audio)
    
    # Copy test set
    test_df = pd.read_csv(os.path.join(PROCESSED_DATA_DIR, 'test.csv'))
    test_audio = np.load(os.path.join(PROCESSED_DATA_DIR, 'test_audio.npy'))
    test_df.to_csv(os.path.join(SMALL_DATA_DIR, 'test.csv'), index=False)
    np.save(os.path.join(SMALL_DATA_DIR, 'test_audio.npy'), test_audio)
    
    # Copy label mappings
    with open(os.path.join(SMALL_DATA_DIR, 'label_mappings.json'), 'w') as f:
        json.dump(mappings, f, indent=2)
    
    print(f"\nSmall dataset saved to {SMALL_DATA_DIR}/")
    print(f"   Train: {len(train_df_small):,} samples")
    print(f"   Val:   {len(val_df):,} samples")
    print(f"   Test:  {len(test_df):,} samples")
    
    print("\n" + "="*70)
    print("Small dataset created!")
    print("="*70)
    print("\nTo use the small dataset for training:")
    print("   1. Update config.py:")
    print('      PROCESSED_DATA_DIR = "data/processed_small"')
    print("   2. Train as normal:")
    print("      python train.py --model lyrics_only --epochs 3")
    print("\nThis will be approximately 10x faster (30 min per epoch instead of 5 hours)")


if __name__ == "__main__":
    np.random.seed(42)  # For reproducibility
    create_small_subset()

