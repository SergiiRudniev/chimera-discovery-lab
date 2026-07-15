"""Chimera Venture: structured graph-to-edit ideation model."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import cast

import torch
from torch import Tensor, nn

from chimera.config import ModelConfig
from chimera.data.contracts import EditBatch, GraphBatch
from chimera.models.encoder import BusinessGraphEncoder, EncoderOutput
from chimera.models.world import LatentWorldModel


@dataclass(frozen=True)
class ChimeraOutput:
    operation_logits: Tensor
    source_logits: Tensor
    target_logits: Tensor
    node_type_logits: Tensor
    edge_type_logits: Tensor
    score_logits: Tensor
    predicted_next_state: Tensor
    graph_state: Tensor
    decoder_states: Tensor


class EditProgramDecoder(nn.Module):
    """Autoregressive decoder over graph-edit symbols, never text tokens."""

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config
        hidden = config.hidden_dim
        self.operation_embedding = nn.Embedding(config.edit_operations, hidden)
        self.source_embedding = nn.Embedding(config.max_nodes, hidden)
        self.target_embedding = nn.Embedding(config.max_nodes, hidden)
        self.node_type_embedding = nn.Embedding(config.node_types, hidden)
        self.edge_type_embedding = nn.Embedding(config.edge_types, hidden)
        self.step_embedding = nn.Embedding(config.max_edits, hidden)
        self.bos = nn.Parameter(torch.empty(1, 1, hidden))
        layer = nn.TransformerDecoderLayer(
            d_model=hidden,
            nhead=config.num_heads,
            dim_feedforward=hidden * config.feedforward_multiplier,
            dropout=config.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.decoder = nn.TransformerDecoder(layer, num_layers=config.decoder_layers)
        self.output_norm = nn.LayerNorm(hidden)
        nn.init.normal_(self.bos, std=0.02)

    def forward(self, encoded: EncoderOutput, edits: EditBatch) -> Tensor:
        edits.validate(
            batch_size=encoded.graph_state.shape[0], max_nodes=self.config.max_nodes
        )
        batch, steps = edits.operations.shape
        if steps > self.config.max_edits:
            raise ValueError("edit program exceeds configured max_edits")
        embedded = (
            self.operation_embedding(edits.operations)
            + self.source_embedding(edits.source_nodes)
            + self.target_embedding(edits.target_nodes)
            + self.node_type_embedding(edits.node_types)
            + self.edge_type_embedding(edits.edge_types)
        )
        shifted = torch.empty_like(embedded)
        shifted[:, :1] = self.bos.expand(batch, -1, -1)
        if steps > 1:
            shifted[:, 1:] = embedded[:, :-1]
        positions = torch.arange(steps, device=embedded.device)
        shifted = shifted + self.step_embedding(positions).unsqueeze(0)
        causal_mask = torch.triu(
            torch.ones(steps, steps, dtype=torch.bool, device=embedded.device), diagonal=1
        )
        states = self.decoder(
            tgt=shifted,
            memory=encoded.memory,
            tgt_mask=causal_mask,
            tgt_key_padding_mask=~edits.step_mask,
            memory_key_padding_mask=~encoded.memory_mask,
        )
        return cast(Tensor, self.output_norm(states) * edits.step_mask.unsqueeze(-1))


class ChimeraVenture(nn.Module):
    """Generate graph-edit programs and predict their latent consequences."""

    def __init__(self, config: ModelConfig | None = None) -> None:
        super().__init__()
        self.config = config or ModelConfig()
        self.encoder = BusinessGraphEncoder(self.config)
        self.edit_decoder = EditProgramDecoder(self.config)
        hidden = self.config.hidden_dim
        self.operation_head = nn.Linear(hidden, self.config.edit_operations)
        self.node_type_head = nn.Linear(hidden, self.config.node_types)
        self.edge_type_head = nn.Linear(hidden, self.config.edge_types)
        self.source_query = nn.Linear(hidden, hidden, bias=False)
        self.target_query = nn.Linear(hidden, hidden, bias=False)
        self.score_head = nn.Sequential(
            nn.LayerNorm(hidden),
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Linear(hidden, self.config.score_dimensions),
        )
        self.world_model = LatentWorldModel(self.config)

    def forward(self, graph: GraphBatch, edits: EditBatch) -> ChimeraOutput:
        if graph.max_nodes != self.config.max_nodes:
            raise ValueError(
                f"graph capacity {graph.max_nodes} does not match configured "
                f"max_nodes {self.config.max_nodes}"
            )
        encoded = self.encoder(graph)
        decoded = self.edit_decoder(encoded, edits)
        source_logits = self._pointer_logits(self.source_query(decoded), encoded, graph.node_mask)
        target_logits = self._pointer_logits(self.target_query(decoded), encoded, graph.node_mask)
        step_weights = edits.step_mask.unsqueeze(-1)
        action_state = (decoded * step_weights).sum(dim=1) / step_weights.sum(dim=1).clamp_min(1)
        predicted_next_state = self.world_model(encoded.graph_state, action_state)
        return ChimeraOutput(
            operation_logits=self.operation_head(decoded),
            source_logits=source_logits,
            target_logits=target_logits,
            node_type_logits=self.node_type_head(decoded),
            edge_type_logits=self.edge_type_head(decoded),
            score_logits=self.score_head(predicted_next_state),
            predicted_next_state=predicted_next_state,
            graph_state=encoded.graph_state,
            decoder_states=decoded,
        )

    def trainable_parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters() if parameter.requires_grad)

    @staticmethod
    def _pointer_logits(query: Tensor, encoded: EncoderOutput, node_mask: Tensor) -> Tensor:
        logits = torch.einsum("btd,bnd->btn", query, encoded.node_states)
        logits = logits / math.sqrt(query.shape[-1])
        return logits.masked_fill(~node_mask[:, None, :], torch.finfo(logits.dtype).min)
