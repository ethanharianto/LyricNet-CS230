"""
Exploratory Data Analysis for LyricNet
Run this after preprocessing to understand your data

Can be run as a Python script or converted to Jupyter notebook
"""

import os
import sys
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import PROCESSED_DATA_DIR

# Set style
sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (12, 6)


def load_data():
    """Load processed data"""
    print("Loading processed data...")
    
    train_df = pd.read_csv(os.path.join(PROCESSED_DATA_DIR, 'train.csv'))
    val_df = pd.read_csv(os.path.join(PROCESSED_DATA_DIR, 'val.csv'))
    test_df = pd.read_csv(os.path.join(PROCESSED_DATA_DIR, 'test.csv'))
    
    with open(os.path.join(PROCESSED_DATA_DIR, 'label_mappings.json'), 'r') as f:
        mappings = json.load(f)
    
    return train_df, val_df, test_df, mappings


def analyze_class_distribution(train_df, mappings):
    """Analyze emotion class distribution"""
    print("\n" + "="*70)
    print("Emotion Class Distribution")
    print("="*70)
    
    id_to_emotion = mappings['id_to_emotion']
    train_df['emotion'] = train_df['labels'].map(lambda x: id_to_emotion[str(x)])
    
    # Count by class
    class_counts = train_df['emotion'].value_counts()
    print(class_counts)
    
    # Visualize
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    
    # Bar plot
    class_counts.plot(kind='bar', ax=axes[0], color='skyblue')
    axes[0].set_title('Emotion Class Distribution (Training Set)')
    axes[0].set_xlabel('Emotion')
    axes[0].set_ylabel('Count')
    axes[0].tick_params(axis='x', rotation=45)
    
    # Pie chart
    axes[1].pie(class_counts.values, labels=class_counts.index, autopct='%1.1f%%')
    axes[1].set_title('Emotion Class Distribution (Training Set)')
    
    plt.tight_layout()
    plt.savefig('eda_class_distribution.png', dpi=300, bbox_inches='tight')
    print("   Saved: eda_class_distribution.png")
    plt.close()


def analyze_lyric_length(train_df):
    """Analyze lyric length distribution"""
    print("\n" + "="*70)
    print("Lyric Length Analysis")
    print("="*70)
    
    train_df['lyric_length'] = train_df['lyrics'].str.len()
    train_df['word_count'] = train_df['lyrics'].str.split().str.len()
    
    print(f"Character count statistics:")
    print(train_df['lyric_length'].describe())
    print(f"\nWord count statistics:")
    print(train_df['word_count'].describe())
    
    # Visualize
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    
    axes[0].hist(train_df['lyric_length'], bins=50, color='lightgreen', edgecolor='black')
    axes[0].set_title('Distribution of Lyric Length (Characters)')
    axes[0].set_xlabel('Number of Characters')
    axes[0].set_ylabel('Frequency')
    
    axes[1].hist(train_df['word_count'], bins=50, color='lightcoral', edgecolor='black')
    axes[1].set_title('Distribution of Word Count')
    axes[1].set_xlabel('Number of Words')
    axes[1].set_ylabel('Frequency')
    
    plt.tight_layout()
    plt.savefig('eda_lyric_length.png', dpi=300, bbox_inches='tight')
    print("   Saved: eda_lyric_length.png")
    plt.close()


def analyze_audio_features(train_df):
    """Analyze audio feature distributions"""
    print("\n" + "="*70)
    print("Audio Feature Analysis")
    print("="*70)
    
    # Load audio features
    audio_features = np.load(os.path.join(PROCESSED_DATA_DIR, 'train_audio.npy'))
    
    from config import AUDIO_FEATURES
    audio_df = pd.DataFrame(audio_features, columns=AUDIO_FEATURES[:audio_features.shape[1]])
    
    print("Audio feature statistics:")
    print(audio_df.describe())
    
    # Visualize distributions
    n_features = len(audio_df.columns)
    n_cols = 3
    n_rows = (n_features + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 4*n_rows))
    axes = axes.flatten()
    
    for i, col in enumerate(audio_df.columns):
        axes[i].hist(audio_df[col], bins=30, color='steelblue', edgecolor='black')
        axes[i].set_title(f'{col} Distribution')
        axes[i].set_xlabel(col)
        axes[i].set_ylabel('Frequency')
    
    # Hide extra subplots
    for i in range(n_features, len(axes)):
        axes[i].axis('off')
    
    plt.tight_layout()
    plt.savefig('eda_audio_features.png', dpi=300, bbox_inches='tight')
    print("   Saved: eda_audio_features.png")
    plt.close()
    
    # Correlation heatmap
    plt.figure(figsize=(10, 8))
    correlation = audio_df.corr()
    sns.heatmap(correlation, annot=True, fmt='.2f', cmap='coolwarm', center=0)
    plt.title('Audio Feature Correlation Matrix')
    plt.tight_layout()
    plt.savefig('eda_audio_correlation.png', dpi=300, bbox_inches='tight')
    print("   Saved: eda_audio_correlation.png")
    plt.close()


def analyze_split_distribution(train_df, val_df, test_df, mappings):
    """Analyze class distribution across splits"""
    print("\n" + "="*70)
    print("Class Distribution Across Splits")
    print("="*70)
    
    id_to_emotion = mappings['id_to_emotion']
    
    splits_data = []
    for split_name, df in [('Train', train_df), ('Val', val_df), ('Test', test_df)]:
        for label in df['labels']:
            splits_data.append({
                'Split': split_name,
                'Emotion': id_to_emotion[str(label)]
            })
    
    splits_df = pd.DataFrame(splits_data)
    
    # Create grouped bar chart
    emotion_counts = splits_df.groupby(['Split', 'Emotion']).size().unstack(fill_value=0)
    
    emotion_counts.plot(kind='bar', figsize=(12, 6))
    plt.title('Emotion Distribution Across Train/Val/Test Splits')
    plt.xlabel('Split')
    plt.ylabel('Count')
    plt.legend(title='Emotion', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig('eda_split_distribution.png', dpi=300, bbox_inches='tight')
    print("   Saved: eda_split_distribution.png")
    plt.close()


def sample_lyrics(train_df, mappings, n_samples=3):
    """Show sample lyrics for each emotion"""
    print("\n" + "="*70)
    print("Sample Lyrics by Emotion")
    print("="*70)
    
    id_to_emotion = mappings['id_to_emotion']
    train_df['emotion'] = train_df['labels'].map(lambda x: id_to_emotion[str(x)])
    
    for emotion in train_df['emotion'].unique():
        print(f"\n{'='*70}")
        print(f"Emotion: {emotion.upper()}")
        print('='*70)
        
        samples = train_df[train_df['emotion'] == emotion]['lyrics'].sample(min(n_samples, len(train_df[train_df['emotion'] == emotion])))
        
        for i, lyric in enumerate(samples, 1):
            print(f"\nSample {i}:")
            print(lyric[:300] + "..." if len(lyric) > 300 else lyric)


def main():
    """Run all EDA analyses"""
    print("="*70)
    print("LyricNet - Exploratory Data Analysis")
    print("="*70)
    
    # Load data
    train_df, val_df, test_df, mappings = load_data()
    
    print(f"\nDataset sizes:")
    print(f"   Train: {len(train_df):,} samples")
    print(f"   Val:   {len(val_df):,} samples")
    print(f"   Test:  {len(test_df):,} samples")
    print(f"   Total: {len(train_df) + len(val_df) + len(test_df):,} samples")
    
    # Run analyses
    analyze_class_distribution(train_df, mappings)
    analyze_lyric_length(train_df)
    analyze_audio_features(train_df)
    analyze_split_distribution(train_df, val_df, test_df, mappings)
    sample_lyrics(train_df, mappings)
    
    print("\n" + "="*70)
    print("✨ EDA Complete! Check the generated PNG files.")
    print("="*70)


if __name__ == "__main__":
    main()

