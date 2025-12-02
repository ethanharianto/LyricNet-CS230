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
    LYRIC_POOLING_STRATEGIES
)
from models.modules import LyricPooling, build_linear


class LyricsOnlyModel(nn.Module):
    """BERT-based classifier using only lyrics."""
    
    def __init__(self, num_classes, freeze_bert=FREEZE_BERT):
        """
        Initialize lyrics-only model.
        
        Args:
            num_classes: Number of emotion classes
            freeze_bert: If True, freeze BERT weights for faster training
        """
        super(LyricsOnlyModel, self).__init__()
        
        # Load pre-trained BERT
        self.bert = AutoModel.from_pretrained(BERT_MODEL_NAME)
        tokenizer = AutoTokenizer.from_pretrained(BERT_MODEL_NAME)
        
        # Freeze BERT if specified for faster training
        if freeze_bert:
            for param in self.bert.parameters():
                param.requires_grad = False
            print("   BERT weights frozen")
            self.bert.eval()
        else:
            # Enable gradient checkpointing to save memory when training BERT
            self.bert.gradient_checkpointing_enable()
        
        
        self.pooling = LyricPooling(BERT_HIDDEN_SIZE, LYRIC_POOLING_STRATEGIES)
        pooled_dim = self.pooling.output_dim

        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(pooled_dim, 1024),
            nn.GELU(),
            #nn.Dropout(0.2),
            nn.Linear(1024, 512),
            nn.GELU(),
            #nn.Dropout(0.2),
            nn.Linear(512, 256),
            nn.GELU(),
            #nn.Dropout(0.2),
            nn.Linear(256, num_classes),
        )
    
    def forward(self, input_ids, attention_mask):
        """
        Forward pass through the model.
        
        Args:
            input_ids: Token IDs [batch_size, seq_length]
            attention_mask: Attention mask [batch_size, seq_length]
        
        Returns:
            logits: Class logits [batch_size, num_classes]
        """
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
        """
        Initialize audio-only model.
        
        Args:
            num_classes: Number of emotion classes
            input_dim: Dimension of audio features
            hidden_dims: List of hidden layer dimensions
        """
        super(AudioOnlyModel, self).__init__()
        
        layers = []
        prev_dim = input_dim
        
        # Build hidden layers
        for hidden_dim in hidden_dims:
            layers.append(build_linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(DROPOUT_RATE))
            prev_dim = hidden_dim
        
        # Output layer
        layers.append(build_linear(prev_dim, num_classes))
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, audio_features):
        """
        Forward pass through the model.
        
        Args:
            audio_features: Audio features [batch_size, audio_feature_dim]
        
        Returns:
            logits: Class logits [batch_size, num_classes]
        """
        return self.network(audio_features)


# if __name__ == "__main__":
#     """Test baseline models."""
#     print("Testing baseline models...\n")
    
#     num_classes = 5
#     batch_size = 4
#     seq_length = 512
#     audio_dim = 11
    
#     # Test Lyrics-Only Model
#     print("1. Testing Lyrics-Only Model")
#     lyrics_model = LyricsOnlyModel(num_classes)
    
#     # Create dummy inputs
#     input_ids = torch.randint(0, 30000, (batch_size, seq_length))
#     attention_mask = torch.ones(batch_size, seq_length)
    
#     logits = lyrics_model(input_ids, attention_mask)
#     print(f"   Input shape: {input_ids.shape}")
#     print(f"   Output shape: {logits.shape}")
#     print(f"   Expected: ({batch_size}, {num_classes})")
#     assert logits.shape == (batch_size, num_classes), "Shape mismatch!"
#     print("   Lyrics-Only Model working successfully!\n")
    
#     # Test Audio-Only Model
#     print("2. Testing Audio-Only Model")
#     audio_model = AudioOnlyModel(num_classes, input_dim=audio_dim)
    
#     # Create dummy audio features
#     audio_features = torch.randn(batch_size, audio_dim)
    
#     logits = audio_model(audio_features)
#     print(f"   Input shape: {audio_features.shape}")
#     print(f"   Output shape: {logits.shape}")
#     print(f"   Expected: ({batch_size}, {num_classes})")
#     assert logits.shape == (batch_size, num_classes), "Shape mismatch!"
#     print("   Audio-Only Model working successfully!\n")
    
#     # Count parameters
#     print("3. Model Statistics")
#     lyrics_params = sum(p.numel() for p in lyrics_model.parameters() if p.requires_grad)
#     audio_params = sum(p.numel() for p in audio_model.parameters() if p.requires_grad)
    
#     print(f"   Lyrics-Only trainable parameters: {lyrics_params:,}")
#     print(f"   Audio-Only trainable parameters: {audio_params:,}")
    
#     print("\nAll baseline models working successfully!")
