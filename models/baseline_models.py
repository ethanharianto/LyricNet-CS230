"""
Baseline models for LyricNet:
1. Lyrics-only model (BERT-based)
2. Audio-only model (MLP-based)
"""

import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer
import sys
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from config import (
    BERT_MODEL_NAME, BERT_HIDDEN_SIZE, FREEZE_BERT,
    AUDIO_FEATURE_DIM, FUSION_HIDDEN_DIMS, DROPOUT_RATE,
    LYRIC_POOLING_STRATEGIES, WEIGHT_DECAY
)
from models.modules import LyricPooling, build_linear


class LyricsOnlyModel(nn.Module):
    """BERT-based classifier using only lyrics."""
    
    def __init__(self, num_classes, freeze_bert=FREEZE_BERT):
        super(LyricsOnlyModel, self).__init__()
        
        self.bert = AutoModel.from_pretrained(BERT_MODEL_NAME)
        tokenizer = AutoTokenizer.from_pretrained(BERT_MODEL_NAME)
        
        if freeze_bert:
            for param in self.bert.parameters():
                param.requires_grad = False
            
            # Unfreeze specific layers
            for param in self.bert.encoder.layer[6].parameters():
                param.requires_grad = True
            for param in self.bert.pooler.parameters():
                param.requires_grad = True
            self.bert.eval()
        else:
            self.bert.gradient_checkpointing_enable()
        
        self.pooling = LyricPooling(BERT_HIDDEN_SIZE, LYRIC_POOLING_STRATEGIES)
        pooled_dim = self.pooling.output_dim

        self.classifier = nn.Sequential(
            nn.Linear(pooled_dim, 1024),
            nn.GELU(),
            nn.Dropout(DROPOUT_RATE),
            nn.Linear(1024, 512),
            nn.GELU(),
            nn.Dropout(DROPOUT_RATE),
            nn.Linear(512, 256),
            nn.GELU(),
            nn.Dropout(DROPOUT_RATE),
            nn.Linear(256, num_classes),
        )
    
    def get_optimizer_parameters(self, weight_decay):
        no_decay = ['bias', 'LayerNorm.weight', 'layer_norm.weight']
        optimizer_grouped_parameters = [
            {
                'params': [p for n, p in self.named_parameters() 
                          if not any(nd in n for nd in no_decay) and p.requires_grad],
                'weight_decay': weight_decay,
            },
            {
                'params': [p for n, p in self.named_parameters() 
                          if any(nd in n for nd in no_decay) and p.requires_grad],
                'weight_decay': 0,
            },
        ]
        return optimizer_grouped_parameters

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True
        )

        cls_embedding = getattr(outputs, 'pooler_output', None)
        if cls_embedding is None:
            cls_embedding = outputs.last_hidden_state[:, 0, :]
        
        lyrics_embedding = self.pooling(
            outputs.last_hidden_state,
            attention_mask,
            cls_embedding
        )
        
        logits = self.classifier(lyrics_embedding)
        
        return logits


class AudioOnlyModel(nn.Module):
    """MLP-based classifier using only audio features."""
    
    def __init__(self, num_classes, input_dim=AUDIO_FEATURE_DIM, 
                 hidden_dims=FUSION_HIDDEN_DIMS):
        super(AudioOnlyModel, self).__init__()
        
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.append(build_linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(DROPOUT_RATE))
            prev_dim = hidden_dim
        
        layers.append(build_linear(prev_dim, num_classes))
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, audio_features):
        return self.network(audio_features)
