"""Permutation-equivariant graph attention with learned relation bias."""

from __future__ import annotations

import math
from typing import cast

import torch
from torch import Tensor, nn


class EdgeBiasedSelfAttention(nn.Module):
    """Dense self-attention for bounded graphs with per-edge head biases."""

    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        edge_types: int,
        dropout: float,
    ) -> None:
        super().__init__()
        if hidden_dim % num_heads:
            raise ValueError("hidden_dim must be divisible by num_heads")
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.qkv = nn.Linear(hidden_dim, hidden_dim * 3)
        self.edge_bias = nn.Embedding(edge_types, num_heads)
        self.output = nn.Linear(hidden_dim, hidden_dim)
        self.attention_dropout = nn.Dropout(dropout)
        self.output_dropout = nn.Dropout(dropout)

    def forward(self, values: Tensor, edge_types: Tensor, node_mask: Tensor) -> Tensor:
        batch, nodes, _ = values.shape
        qkv = self.qkv(values).view(batch, nodes, 3, self.num_heads, self.head_dim)
        queries, keys, payload = qkv.unbind(dim=2)
        queries = queries.transpose(1, 2)
        keys = keys.transpose(1, 2)
        payload = payload.transpose(1, 2)

        logits = torch.matmul(queries, keys.transpose(-2, -1)) / math.sqrt(self.head_dim)
        relation_bias = self.edge_bias(edge_types).permute(0, 3, 1, 2)
        logits = logits + relation_bias
        logits = logits.masked_fill(~node_mask[:, None, None, :], torch.finfo(logits.dtype).min)
        attention = torch.softmax(logits, dim=-1)
        attention = self.attention_dropout(attention)
        attended = torch.matmul(attention, payload)
        attended = attended.transpose(1, 2).contiguous().view(batch, nodes, self.hidden_dim)
        result = self.output_dropout(self.output(attended))
        return cast(Tensor, result * node_mask.unsqueeze(-1))


class GraphTransformerBlock(nn.Module):
    """Pre-normalized relation-aware transformer block."""

    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        edge_types: int,
        feedforward_multiplier: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.attention_norm = nn.LayerNorm(hidden_dim)
        self.attention = EdgeBiasedSelfAttention(
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            edge_types=edge_types,
            dropout=dropout,
        )
        self.feedforward_norm = nn.LayerNorm(hidden_dim)
        inner_dim = hidden_dim * feedforward_multiplier
        self.feedforward = nn.Sequential(
            nn.Linear(hidden_dim, inner_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(inner_dim, hidden_dim),
            nn.Dropout(dropout),
        )

    def forward(self, values: Tensor, edge_types: Tensor, node_mask: Tensor) -> Tensor:
        values = values + self.attention(self.attention_norm(values), edge_types, node_mask)
        values = values + self.feedforward(self.feedforward_norm(values))
        return values * node_mask.unsqueeze(-1)
