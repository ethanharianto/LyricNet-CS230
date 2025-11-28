"""
Utility modules for LyricNet models: pooling and mixout-enabled linear layers.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from config import (
    MIXOUT_PROB,
    USE_MIXOUT,
    POOLING_ATTENTION_HIDDEN,
)


class MixLinear(nn.Linear):
    """Linear layer with Mixout regularization applied to weights."""

    def __init__(self, in_features, out_features, bias=True, mixout_p=MIXOUT_PROB):
        super().__init__(in_features, out_features, bias=bias)
        self.mixout_p = mixout_p
        self.register_buffer("target_weight", self.weight.data.clone())

    def forward(self, input):
        weight = self.weight
        if self.training and self.mixout_p > 0:
            mask = torch.empty_like(weight).bernoulli_(1 - self.mixout_p)
            weight = mask * weight + (1 - mask) * self.target_weight
        return F.linear(input, weight, self.bias)


def build_linear(in_features, out_features, bias=True):
    """Factory that returns a linear layer with optional Mixout regularization."""
    if USE_MIXOUT and MIXOUT_PROB > 0:
        return MixLinear(in_features, out_features, bias=bias, mixout_p=MIXOUT_PROB)
    return nn.Linear(in_features, out_features, bias=bias)


class AttentionPooling(nn.Module):
    """Self-attention pooling across token representations."""

    def __init__(self, hidden_size, attention_hidden=POOLING_ATTENTION_HIDDEN):
        super().__init__()
        self.projection = nn.Sequential(
            nn.Linear(hidden_size, attention_hidden),
            nn.Tanh(),
            nn.Linear(attention_hidden, 1)
        )

    def forward(self, hidden_states, attention_mask):
        attn_scores = self.projection(hidden_states).squeeze(-1)
        attn_scores = attn_scores.masked_fill(attention_mask == 0, -1e9)
        attn_weights = torch.softmax(attn_scores, dim=1)
        return torch.sum(hidden_states * attn_weights.unsqueeze(-1), dim=1)


class LyricPooling(nn.Module):
    """Combine multiple pooling strategies into a single lyric embedding."""

    def __init__(self, hidden_size, strategies):
        super().__init__()
        if isinstance(strategies, str):
            strategies = [strategies]
        normalized = []
        for strategy in strategies:
            key = strategy.lower()
            if key not in normalized:
                normalized.append(key)
        if not normalized:
            raise ValueError("At least one pooling strategy must be specified")
        self.strategies = normalized
        self.attention_pool = AttentionPooling(hidden_size) if 'attention' in self.strategies else None
        self.hidden_size = hidden_size
        self.output_dim = hidden_size * len(self.strategies)

    def forward(self, last_hidden_state, attention_mask, cls_embedding):
        features = []
        mask = attention_mask.unsqueeze(-1).float()
        if 'cls' in self.strategies:
            features.append(cls_embedding)
        if 'mean' in self.strategies:
            summed = torch.sum(last_hidden_state * mask, dim=1)
            denom = mask.sum(dim=1).clamp(min=1e-6)
            features.append(summed / denom)
        if 'max' in self.strategies:
            masked = last_hidden_state.masked_fill(mask.squeeze(-1) == 0, -1e9)
            features.append(masked.max(dim=1).values)
        if 'attention' in self.strategies and self.attention_pool is not None:
            features.append(self.attention_pool(last_hidden_state, attention_mask))
        return torch.cat(features, dim=1)
