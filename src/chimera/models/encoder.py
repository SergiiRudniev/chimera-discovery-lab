"""Typed graph encoder for non-linguistic business state."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn

from chimera.config import ModelConfig
from chimera.data.contracts import GraphBatch
from chimera.models.attention import GraphTransformerBlock


@dataclass(frozen=True)
class EncoderOutput:
    memory: Tensor
    memory_mask: Tensor
    node_states: Tensor
    graph_state: Tensor


class BusinessGraphEncoder(nn.Module):
    """Encode a bounded typed graph without tokenized natural language."""

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config
        self.node_type_embedding = nn.Embedding(config.node_types, config.hidden_dim, padding_idx=0)
        self.feature_projection = nn.Sequential(
            nn.LayerNorm(config.node_numeric_features),
            nn.Linear(config.node_numeric_features, config.hidden_dim),
            nn.GELU(),
            nn.Linear(config.hidden_dim, config.hidden_dim),
        )
        self.global_token = nn.Parameter(torch.empty(1, 1, config.hidden_dim))
        self.input_norm = nn.LayerNorm(config.hidden_dim)
        self.input_dropout = nn.Dropout(config.dropout)
        self.blocks = nn.ModuleList(
            [
                GraphTransformerBlock(
                    hidden_dim=config.hidden_dim,
                    num_heads=config.num_heads,
                    edge_types=config.edge_types,
                    feedforward_multiplier=config.feedforward_multiplier,
                    dropout=config.dropout,
                )
                for _ in range(config.encoder_layers)
            ]
        )
        self.output_norm = nn.LayerNorm(config.hidden_dim)
        nn.init.normal_(self.global_token, std=0.02)

    def forward(self, graph: GraphBatch) -> EncoderOutput:
        graph.validate(feature_dim=self.config.node_numeric_features)
        batch, nodes = graph.node_types.shape
        node_values = self.node_type_embedding(graph.node_types)
        node_values = node_values + self.feature_projection(graph.node_features)
        global_values = self.global_token.expand(batch, -1, -1)
        values = self.input_dropout(self.input_norm(torch.cat((global_values, node_values), dim=1)))

        memory_mask = torch.cat(
            (
                torch.ones(batch, 1, dtype=torch.bool, device=graph.node_mask.device),
                graph.node_mask,
            ),
            dim=1,
        )
        relation_ids = torch.zeros(
            batch,
            nodes + 1,
            nodes + 1,
            dtype=torch.long,
            device=graph.edge_types.device,
        )
        relation_ids[:, 1:, 1:] = graph.edge_types
        for block in self.blocks:
            values = block(values, relation_ids, memory_mask)
        values = self.output_norm(values)
        return EncoderOutput(
            memory=values,
            memory_mask=memory_mask,
            node_states=values[:, 1:],
            graph_state=values[:, 0],
        )
