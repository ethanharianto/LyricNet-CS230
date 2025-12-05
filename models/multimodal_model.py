"""
Multimodal fusion model for LyricNet.
Combines lyrics (BERT) and audio features.
"""

import torch
import torch.nn as nn
from transformers import AutoModel
import sys
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from config import (
    BERT_MODEL_NAME, BERT_HIDDEN_SIZE, FREEZE_BERT,
    AUDIO_FEATURE_DIM, FUSION_HIDDEN_DIMS, DROPOUT_RATE,
    FUSION_ATTENTION_LAYERS, FUSION_ATTENTION_HEADS,
    LYRIC_POOLING_STRATEGIES
)
from models.modules import LyricPooling, build_linear


class MultimodalFusionModel(nn.Module):
    """
    Multimodal model that fuses lyrics (BERT) and audio features.
    """
    
    def __init__(self, num_classes, freeze_bert=FREEZE_BERT):
        super(MultimodalFusionModel, self).__init__()
        
        # Lyrics Branch (BERT)
        self.bert = AutoModel.from_pretrained(BERT_MODEL_NAME)
        
        if freeze_bert:
            for param in self.bert.parameters():
                param.requires_grad = False
        
        self.pooling = LyricPooling(BERT_HIDDEN_SIZE, LYRIC_POOLING_STRATEGIES)
        lyric_input_dim = self.pooling.output_dim
        
        self.lyrics_projection = nn.Sequential(
            build_linear(lyric_input_dim, BERT_HIDDEN_SIZE),
            nn.ReLU(),
            nn.Dropout(DROPOUT_RATE)
        )
        
        # Audio Branch (MLP)
        self.audio_encoder = nn.Sequential(
            build_linear(AUDIO_FEATURE_DIM, 64),
            nn.ReLU(),
            nn.Dropout(DROPOUT_RATE),
            build_linear(64, 128),
            nn.ReLU(),
            nn.Dropout(DROPOUT_RATE),
            build_linear(128, BERT_HIDDEN_SIZE),
            nn.ReLU(),
            nn.Dropout(DROPOUT_RATE)
        )
        
        # Cross-modal attention
        self.use_attention = FUSION_ATTENTION_LAYERS > 0
        self.attention_layers = nn.ModuleList()
        self.attention_norms = nn.ModuleList()
        if self.use_attention:
            for _ in range(FUSION_ATTENTION_LAYERS):
                self.attention_layers.append(
                    nn.MultiheadAttention(
                        embed_dim=BERT_HIDDEN_SIZE,
                        num_heads=FUSION_ATTENTION_HEADS,
                        dropout=DROPOUT_RATE,
                        batch_first=True
                    )
                )
                self.attention_norms.append(nn.LayerNorm(BERT_HIDDEN_SIZE))

        # Fusion Network
        fusion_input_dim = BERT_HIDDEN_SIZE * 2
        
        fusion_layers = []
        prev_dim = fusion_input_dim
        
        for hidden_dim in FUSION_HIDDEN_DIMS:
            fusion_layers.append(build_linear(prev_dim, hidden_dim))
            fusion_layers.append(nn.ReLU())
            fusion_layers.append(nn.Dropout(DROPOUT_RATE))
            prev_dim = hidden_dim
        
        fusion_layers.append(build_linear(prev_dim, num_classes))
        
        self.fusion_network = nn.Sequential(*fusion_layers)
    
    def forward(self, input_ids, attention_mask, audio_features):
        # Lyrics encoding
        bert_outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True
        )
        cls_embedding = getattr(bert_outputs, 'pooler_output', None)
        if cls_embedding is None:
            cls_embedding = bert_outputs.last_hidden_state[:, 0, :]
        lyrics_tokens = bert_outputs.last_hidden_state
        lyrics_embedding = self.pooling(
            lyrics_tokens,
            attention_mask,
            cls_embedding
        )
        lyrics_embedding = self.lyrics_projection(lyrics_embedding)
        
        # Audio encoding
        audio_embedding = self.audio_encoder(audio_features)
        
        # Fusion
        if self.use_attention:
            stacked = torch.stack([lyrics_embedding, audio_embedding], dim=1)
            for attn, norm in zip(self.attention_layers, self.attention_norms):
                attn_output, _ = attn(stacked, stacked, stacked)
                stacked = norm(stacked + attn_output)
            fused = stacked.reshape(stacked.size(0), -1)
        else:
            fused = torch.cat([lyrics_embedding, audio_embedding], dim=1)
        
        logits = self.fusion_network(fused)
        
        return logits
    
    def get_embeddings(self, input_ids, attention_mask, audio_features):
        # Get lyrics embedding
        bert_outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask, return_dict=True)
        cls_embedding = getattr(bert_outputs, 'pooler_output', None)
        if cls_embedding is None:
            cls_embedding = bert_outputs.last_hidden_state[:, 0, :]
        lyrics_embedding = self.pooling(
            bert_outputs.last_hidden_state,
            attention_mask,
            cls_embedding
        )
        lyrics_embedding = self.lyrics_projection(lyrics_embedding)
        
        # Get audio embedding
        audio_embedding = self.audio_encoder(audio_features)
        
        if self.use_attention:
            stacked = torch.stack([lyrics_embedding, audio_embedding], dim=1)
            for attn, norm in zip(self.attention_layers, self.attention_norms):
                attn_output, _ = attn(stacked, stacked, stacked)
                stacked = norm(stacked + attn_output)
            fused = stacked.reshape(stacked.size(0), -1)
        else:
            fused = torch.cat([lyrics_embedding, audio_embedding], dim=1)
        
        # Pass through fusion layers, excluding final classification layer
        for i, layer in enumerate(self.fusion_network[:-1]):
            fused = layer(fused)
        
        return fused
