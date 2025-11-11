"""
PyTorch Dataset and DataLoader for LyricNet.
"""

import os
import json
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer
import sys
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from config import (
    PROCESSED_DATA_DIR, MAX_LYRIC_LENGTH, 
    BERT_MODEL_NAME, BATCH_SIZE
)


class LyricDataset(Dataset):
    """PyTorch Dataset for lyrics and audio features."""
    
    def __init__(self, split='train', tokenizer=None):
        """
        Initialize dataset.
        
        Args:
            split: Data split - 'train', 'val', or 'test'
            tokenizer: BERT tokenizer instance (creates new one if None)
        """
        self.split = split
        
        # Load tokenizer
        if tokenizer is None:
            self.tokenizer = BertTokenizer.from_pretrained(BERT_MODEL_NAME)
        else:
            self.tokenizer = tokenizer
        
        # Load data
        csv_path = os.path.join(PROCESSED_DATA_DIR, f'{split}.csv')
        audio_path = os.path.join(PROCESSED_DATA_DIR, f'{split}_audio.npy')
        
        if not os.path.exists(csv_path):
            raise FileNotFoundError(
                f"Data file not found: {csv_path}\n"
                f"Please run data/preprocess.py first"
            )
        
        self.data = pd.read_csv(csv_path)
        self.audio_features = np.load(audio_path)
        
        print(f"Loaded {split} set: {len(self.data)} samples")
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        """
        Get a single data sample.
        
        Returns:
            Dictionary containing:
                - input_ids: BERT token IDs
                - attention_mask: BERT attention mask
                - audio_features: Spotify audio features
                - label: Emotion class ID
        """
        row = self.data.iloc[idx]
        
        # Tokenize lyrics
        lyrics = str(row['lyrics'])
        encoding = self.tokenizer.encode_plus(
            lyrics,
            add_special_tokens=True,
            max_length=MAX_LYRIC_LENGTH,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )
        
        # Get audio features
        audio = torch.FloatTensor(self.audio_features[idx])
        
        # Get label
        label = torch.LongTensor([row['labels']])[0]
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'audio_features': audio,
            'label': label
        }


def get_data_loaders(batch_size=BATCH_SIZE, num_workers=0):
    """
    Create train, validation, and test data loaders.
    
    Args:
        batch_size: Batch size for training
        num_workers: Number of workers for data loading (use 0 for MPS/Mac compatibility)
    
    Returns:
        Dictionary with 'train', 'val', 'test' DataLoaders and 'tokenizer'
    """
    # Create tokenizer (shared across all splits)
    tokenizer = BertTokenizer.from_pretrained(BERT_MODEL_NAME)
    
    # Create datasets
    train_dataset = LyricDataset('train', tokenizer)
    val_dataset = LyricDataset('val', tokenizer)
    test_dataset = LyricDataset('test', tokenizer)
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers
    )
    
    return {
        'train': train_loader,
        'val': val_loader,
        'test': test_loader,
        'tokenizer': tokenizer
    }


def load_label_mappings():
    """Load emotion label mappings from processed data directory."""
    mapping_path = os.path.join(PROCESSED_DATA_DIR, 'label_mappings.json')
    
    if not os.path.exists(mapping_path):
        raise FileNotFoundError(
            f"Label mappings not found: {mapping_path}\n"
            f"Please run data/preprocess.py first"
        )
    
    with open(mapping_path, 'r') as f:
        mappings = json.load(f)
    
    return mappings


if __name__ == "__main__":
    """Test data loading functionality."""
    print("Testing data loader...")
    
    # Load label mappings
    mappings = load_label_mappings()
    print(f"\nEmotion classes ({mappings['num_classes']}):")
    for emotion_id, emotion_name in mappings['id_to_emotion'].items():
        print(f"  {emotion_id}: {emotion_name}")
    
    # Create data loaders
    loaders = get_data_loaders(batch_size=4)
    
    # Test train loader
    print("\nTesting train loader...")
    batch = next(iter(loaders['train']))
    
    print(f"  Input IDs shape: {batch['input_ids'].shape}")
    print(f"  Attention mask shape: {batch['attention_mask'].shape}")
    print(f"  Audio features shape: {batch['audio_features'].shape}")
    print(f"  Labels shape: {batch['label'].shape}")
    print(f"  Sample labels: {batch['label'].tolist()}")
    
    print("\nData loader test passed successfully!")

