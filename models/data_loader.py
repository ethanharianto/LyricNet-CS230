"""
PyTorch Dataset and DataLoader for LyricNet.
"""

import os
import json
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer
import sys
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from config import (
    PROCESSED_DATA_DIR, MAX_LYRIC_LENGTH, 
    BERT_MODEL_NAME, BATCH_SIZE,
    USE_WEIGHTED_SAMPLER, SAMPLER_ALPHA
)


class LyricDataset(Dataset):
    """PyTorch Dataset for lyrics and audio features."""
    
    def __init__(self, split='train', tokenizer=None):
        self.split = split
        self.tokenizer = AutoTokenizer.from_pretrained(BERT_MODEL_NAME)

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
        
        if split == 'train':
            self._compute_weights()
    
    def _compute_weights(self):
        labels = self.data['labels'].values
        class_counts = np.bincount(labels)
        
        # Calculate weights proportional to N^(alpha-1)
        exponent = SAMPLER_ALPHA - 1.0
        class_weights = np.power(class_counts, exponent)
        
        self.sample_weights = torch.from_numpy(class_weights[labels]).float()
        print(f"Computed sample weights (alpha={SAMPLER_ALPHA}) for {len(self.sample_weights)} samples")
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        
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
        
        audio = torch.FloatTensor(self.audio_features[idx])
        label = torch.LongTensor([row['labels']])[0]
        
        return {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'audio_features': audio,
            'label': label
        }


def get_data_loaders(batch_size=BATCH_SIZE, num_workers=0):
    tokenizer = AutoTokenizer.from_pretrained(BERT_MODEL_NAME)
    
    train_dataset = LyricDataset('train', tokenizer)
    val_dataset = LyricDataset('val', tokenizer)
    test_dataset = LyricDataset('test', tokenizer)
    
    sampler = None
    shuffle = True
    
    if USE_WEIGHTED_SAMPLER:
        print(f"Using WeightedRandomSampler with alpha={SAMPLER_ALPHA}")
        sampler = torch.utils.data.WeightedRandomSampler(
            weights=train_dataset.sample_weights,
            num_samples=len(train_dataset),
            replacement=True
        )
        shuffle = False
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        sampler=sampler,
        shuffle=shuffle,
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
    mapping_path = os.path.join(PROCESSED_DATA_DIR, 'label_mappings.json')
    
    if not os.path.exists(mapping_path):
        raise FileNotFoundError(
            f"Label mappings not found: {mapping_path}\n"
            f"Please run data/preprocess.py first"
        )
    
    with open(mapping_path, 'r') as f:
        mappings = json.load(f)
    
    return mappings

