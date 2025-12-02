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
    
    Architecture:
    1. BERT encoder for lyrics - produces 768-dim embedding
    2. MLP encoder for audio - produces 768-dim embedding
    3. Concatenate both embeddings - produces 1536-dim vector
    4. MLP fusion network - produces emotion prediction
    """
    
    def __init__(self, num_classes, freeze_bert=FREEZE_BERT):
        """
        Initialize multimodal fusion model.
        
        Args:
            num_classes: Number of emotion classes
            freeze_bert: If True, freeze BERT weights
        """
        super(MultimodalFusionModel, self).__init__()
        
        # ====================================================================
        # Lyrics Branch (BERT)
        # ====================================================================
        self.bert = AutoModel.from_pretrained(BERT_MODEL_NAME)
        
        if freeze_bert:
            for param in self.bert.parameters():
                param.requires_grad = False
            print("   BERT weights frozen")
        
        self.pooling = LyricPooling(BERT_HIDDEN_SIZE, LYRIC_POOLING_STRATEGIES)
        lyric_input_dim = self.pooling.output_dim
        
        # Project pooled lyrics into shared embedding space
        self.lyrics_projection = nn.Sequential(
            build_linear(lyric_input_dim, BERT_HIDDEN_SIZE),
            nn.ReLU(),
            nn.Dropout(DROPOUT_RATE)
        )
        
        # ====================================================================
        # Audio Branch (MLP)
        # ====================================================================
        # Project audio features to same dimension as BERT
        self.audio_encoder = nn.Sequential(
            build_linear(AUDIO_FEATURE_DIM, 64),
            nn.ReLU(),
            nn.Dropout(DROPOUT_RATE),
            build_linear(64, 128),
            nn.ReLU(),
            nn.Dropout(DROPOUT_RATE),
            build_linear(128, BERT_HIDDEN_SIZE),  # Project to match BERT dimension
            nn.ReLU(),
            nn.Dropout(DROPOUT_RATE)
        )
        
        # ====================================================================
        # Cross-modal attention (optional)
        # ====================================================================
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

        # ====================================================================
        # Fusion Network
        # ====================================================================
        # Concatenated dimension: 768 (lyrics) + 768 (audio) = 1536
        fusion_input_dim = BERT_HIDDEN_SIZE * 2
        
        fusion_layers = []
        prev_dim = fusion_input_dim
        
        for hidden_dim in FUSION_HIDDEN_DIMS:
            fusion_layers.append(build_linear(prev_dim, hidden_dim))
            fusion_layers.append(nn.ReLU())
            fusion_layers.append(nn.Dropout(DROPOUT_RATE))
            prev_dim = hidden_dim
        
        # Final classification layer
        fusion_layers.append(build_linear(prev_dim, num_classes))
        
        self.fusion_network = nn.Sequential(*fusion_layers)
    
    def forward(self, input_ids, attention_mask, audio_features):
        """
        Forward pass through the model.
        
        Args:
            input_ids: BERT token IDs [batch_size, seq_length]
            attention_mask: BERT attention mask [batch_size, seq_length]
            audio_features: Spotify audio features [batch_size, audio_feature_dim]
        
        Returns:
            logits: Class logits [batch_size, num_classes]
        """
        # ====================================================================
        # Lyrics encoding
        # ====================================================================
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
        
        # ====================================================================
        # Audio encoding
        # ====================================================================
        audio_embedding = self.audio_encoder(audio_features)  # [batch_size, 768]
        
        # ====================================================================
        # Fusion
        # ====================================================================
        if self.use_attention:
            stacked = torch.stack([lyrics_embedding, audio_embedding], dim=1)  # [batch, 2, 768]
            for attn, norm in zip(self.attention_layers, self.attention_norms):
                attn_output, _ = attn(stacked, stacked, stacked)
                stacked = norm(stacked + attn_output)
            fused = stacked.reshape(stacked.size(0), -1)
        else:
            fused = torch.cat([lyrics_embedding, audio_embedding], dim=1)  # [batch_size, 1536]
        
        # Pass through fusion network
        logits = self.fusion_network(fused)  # [batch_size, num_classes]
        
        return logits
    
    def get_embeddings(self, input_ids, attention_mask, audio_features):
        """
        Get the fused embeddings before classification.
        Useful for visualization and clustering analysis.
        
        Returns:
            embeddings: Fused embeddings [batch_size, last_hidden_dim]
        """
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


if __name__ == "__main__":
    """Test multimodal model."""
    print("Testing Multimodal Fusion Model...\n")
    
    num_classes = 5
    batch_size = 4
    seq_length = 512
    audio_dim = 11
    
    # Create model
    model = MultimodalFusionModel(num_classes, freeze_bert=True)
    
    # Create dummy inputs
    input_ids = torch.randint(0, 30000, (batch_size, seq_length))
    attention_mask = torch.ones(batch_size, seq_length)
    audio_features = torch.randn(batch_size, audio_dim)
    
    # Forward pass
    print("1. Testing forward pass")
    logits = model(input_ids, attention_mask, audio_features)
    print(f"   Input shapes:")
    print(f"     - Lyrics: {input_ids.shape}")
    print(f"     - Audio: {audio_features.shape}")
    print(f"   Output shape: {logits.shape}")
    print(f"   Expected: ({batch_size}, {num_classes})")
    assert logits.shape == (batch_size, num_classes), "Shape mismatch!"
    print("   Forward pass working successfully!\n")
    
    # Test embedding extraction
    print("2. Testing embedding extraction")
    embeddings = model.get_embeddings(input_ids, attention_mask, audio_features)
    print(f"   Embedding shape: {embeddings.shape}")
    print("   Embedding extraction working successfully!\n")
    
    # Count parameters
    print("3. Model Statistics")
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"   Total parameters: {total_params:,}")
    print(f"   Trainable parameters: {trainable_params:,}")
    print(f"   Frozen parameters: {total_params - trainable_params:,}")
    
    print("\nMultimodal model working successfully!")
