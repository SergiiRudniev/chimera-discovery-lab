"""Object-centric causal-dynamics core for Chimera Meta-World W0."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import cast

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from chimera.meta_world.batch import MetaWorldBatch
from chimera.meta_world.config import MetaWorldModelConfig


@dataclass(frozen=True)
class MetaWorldOutput:
    """Numerical W0 proposal and uncertainty estimates."""

    next_state_mean: Tensor
    next_state_log_variance: Tensor
    effect_mean: Tensor
    effect_log_variance: Tensor
    proposal_embedding: Tensor
    final_slot_states: Tensor
    transition_state: Tensor


class ContinuousRelationAttention(nn.Module):
    """Permutation-equivariant slot attention with continuous relation bias."""

    def __init__(self, config: MetaWorldModelConfig) -> None:
        super().__init__()
        hidden = config.hidden_dim
        self.num_heads = config.num_heads
        self.head_dim = hidden // config.num_heads
        self.qkv = nn.Linear(hidden, hidden * 3)
        self.relation_bias = nn.Sequential(
            nn.Linear(config.relation_features, config.num_heads),
            nn.Tanh(),
        )
        self.output = nn.Linear(hidden, hidden)
        self.attention_dropout = nn.Dropout(config.dropout)
        self.output_dropout = nn.Dropout(config.dropout)

    def forward(self, values: Tensor, relations: Tensor, slot_mask: Tensor) -> Tensor:
        batch, slots, hidden = values.shape
        qkv = self.qkv(values).reshape(
            batch, slots, 3, self.num_heads, self.head_dim
        )
        query, key, value = qkv.unbind(dim=2)
        query = query.transpose(1, 2)
        key = key.transpose(1, 2)
        value = value.transpose(1, 2)
        scores = torch.matmul(query, key.transpose(-1, -2)) / math.sqrt(self.head_dim)
        bias = self.relation_bias(relations).permute(0, 3, 1, 2)
        scores = scores + bias
        scores = scores.masked_fill(
            ~slot_mask[:, None, None, :], torch.finfo(scores.dtype).min
        )
        weights = self.attention_dropout(torch.softmax(scores, dim=-1))
        attended = torch.matmul(weights, value).transpose(1, 2).reshape(batch, slots, hidden)
        attended = self.output_dropout(self.output(attended))
        return cast(Tensor, attended * slot_mask.unsqueeze(-1).to(attended.dtype))


class SpatialBlock(nn.Module):
    """Pre-normalized relational attention and feed-forward block."""

    def __init__(self, config: MetaWorldModelConfig) -> None:
        super().__init__()
        hidden = config.hidden_dim
        inner = hidden * config.feedforward_multiplier
        self.attention_norm = nn.LayerNorm(hidden)
        self.attention = ContinuousRelationAttention(config)
        self.feedforward_norm = nn.LayerNorm(hidden)
        self.feedforward = nn.Sequential(
            nn.Linear(hidden, inner),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(inner, hidden),
            nn.Dropout(config.dropout),
        )

    def forward(self, values: Tensor, relations: Tensor, slot_mask: Tensor) -> Tensor:
        values = values + self.attention(self.attention_norm(values), relations, slot_mask)
        values = values + self.feedforward(self.feedforward_norm(values))
        return values * slot_mask.unsqueeze(-1).to(values.dtype)


class TransitionBlock(nn.Module):
    """Residual intervention-conditioned latent dynamics block."""

    def __init__(self, config: MetaWorldModelConfig) -> None:
        super().__init__()
        hidden = config.hidden_dim
        inner = hidden * config.feedforward_multiplier
        self.norm = nn.LayerNorm(hidden)
        self.mlp = nn.Sequential(
            nn.Linear(hidden, inner),
            nn.SiLU(),
            nn.Dropout(config.dropout),
            nn.Linear(inner, hidden),
            nn.Dropout(config.dropout),
        )

    def forward(self, values: Tensor) -> Tensor:
        return cast(Tensor, values + self.mlp(self.norm(values)))


class ChimeraMetaWorld(nn.Module):
    """W0 dynamic hypergraph prototype with no language inputs or text heads."""

    def __init__(self, config: MetaWorldModelConfig) -> None:
        super().__init__()
        self.config = config
        hidden = config.hidden_dim
        adapter_input = config.observation_features * 2
        self.domain_adapters = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(adapter_input, hidden),
                    nn.SiLU(),
                    nn.Linear(hidden, hidden),
                )
                for _ in range(config.domain_count)
            ]
        )
        self.spatial_blocks = nn.ModuleList(
            [SpatialBlock(config) for _ in range(config.spatial_layers)]
        )
        self.spatial_norm = nn.LayerNorm(hidden)
        self.time_embedding = nn.Parameter(torch.empty(1, config.context_steps, hidden))
        temporal_layer = nn.TransformerEncoderLayer(
            d_model=hidden,
            nhead=config.num_heads,
            dim_feedforward=hidden * config.feedforward_multiplier,
            dropout=config.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.temporal_encoder = nn.TransformerEncoder(
            temporal_layer,
            num_layers=config.temporal_layers,
            norm=nn.LayerNorm(hidden),
            enable_nested_tensor=False,
        )
        self.intervention_type_embedding = nn.Embedding(config.intervention_types, hidden)
        self.intervention_parameter_encoder = nn.Sequential(
            nn.Linear(config.intervention_parameters, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
        )
        self.intervention_fusion = nn.Sequential(
            nn.LayerNorm(hidden * 4),
            nn.Linear(hidden * 4, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
        )
        self.transition_input = nn.Sequential(
            nn.LayerNorm(hidden * 2),
            nn.Linear(hidden * 2, hidden),
        )
        self.transition_blocks = nn.ModuleList(
            [TransitionBlock(config) for _ in range(config.transition_layers)]
        )
        self.transition_norm = nn.LayerNorm(hidden)
        self.slot_condition = nn.Linear(hidden, hidden)
        self.slot_output_norm = nn.LayerNorm(hidden)
        self.domain_decoders = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(hidden, hidden),
                    nn.SiLU(),
                    nn.Linear(hidden, config.observation_features * 2),
                )
                for _ in range(config.domain_count)
            ]
        )
        self.effect_head = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.SiLU(),
            nn.Linear(hidden, config.effect_dimensions * 2),
        )
        self.proposal_projection = nn.Linear(hidden, hidden)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.normal_(self.time_embedding, mean=0.0, std=0.02)

    @staticmethod
    def _gather_time(values: Tensor, steps: Tensor) -> Tensor:
        index = steps.view(steps.shape[0], 1, *([1] * (values.ndim - 2)))
        index = index.expand(steps.shape[0], 1, *values.shape[2:])
        return values.gather(1, index).squeeze(1)

    @staticmethod
    def _gather_slots(values: Tensor, pointers: Tensor) -> Tensor:
        index = pointers[:, None, None].expand(values.shape[0], 1, values.shape[-1])
        return values.gather(1, index).squeeze(1)

    @staticmethod
    def _apply_domain_modules(
        modules: nn.ModuleList, values: Tensor, domain_ids: Tensor
    ) -> Tensor:
        output = modules[0](values)
        selection_shape = (domain_ids.shape[0],) + (1,) * (values.ndim - 1)
        for domain_index, module in enumerate(modules[1:], start=1):
            candidate = module(values)
            selected = (domain_ids == domain_index).view(selection_shape)
            output = torch.where(selected, candidate, output)
        return cast(Tensor, output)

    def forward(self, batch: MetaWorldBatch) -> MetaWorldOutput:
        batch.validate()
        config = self.config
        if batch.observations.shape[1:] != (
            config.context_steps,
            config.max_slots,
            config.observation_features,
        ):
            raise ValueError("batch observations do not match the W0 model contract")
        if batch.relations.shape[-1] != config.relation_features:
            raise ValueError("batch relation feature count does not match the model")
        if batch.intervention_parameters.shape[-1] != config.intervention_parameters:
            raise ValueError("batch intervention parameter count does not match the model")
        if torch.any(batch.domain_ids < 0) or torch.any(batch.domain_ids >= config.domain_count):
            raise ValueError("domain_ids are outside the configured range")

        adapter_input = torch.cat(
            [batch.observations, batch.observation_mask.to(batch.observations.dtype)], dim=-1
        )
        encoded = self._apply_domain_modules(
            self.domain_adapters, adapter_input, batch.domain_ids
        )
        encoded = encoded * batch.slot_mask.unsqueeze(-1).to(encoded.dtype)

        batch_size = batch.batch_size
        flat_values = encoded.reshape(
            batch_size * config.context_steps, config.max_slots, config.hidden_dim
        )
        flat_relations = batch.relations.reshape(
            batch_size * config.context_steps,
            config.max_slots,
            config.max_slots,
            config.relation_features,
        )
        flat_slot_mask = batch.slot_mask.reshape(
            batch_size * config.context_steps, config.max_slots
        )
        for block in self.spatial_blocks:
            flat_values = block(flat_values, flat_relations, flat_slot_mask)
        flat_values = self.spatial_norm(flat_values)
        slot_states = flat_values.reshape(
            batch_size,
            config.context_steps,
            config.max_slots,
            config.hidden_dim,
        )
        slot_weight = batch.slot_mask.unsqueeze(-1).to(slot_states.dtype)
        graph_states = (slot_states * slot_weight).sum(dim=2) / slot_weight.sum(dim=2).clamp_min(1)
        graph_states = graph_states + self.time_embedding[:, : config.context_steps]
        causal_mask = torch.triu(
            torch.ones(
                config.context_steps,
                config.context_steps,
                dtype=torch.bool,
                device=graph_states.device,
            ),
            diagonal=1,
        )
        temporal_states = self.temporal_encoder(
            graph_states,
            mask=causal_mask,
            src_key_padding_mask=~batch.time_mask,
        )
        final_steps = batch.time_mask.sum(dim=1) - 1
        final_graph = self._gather_time(temporal_states, final_steps)
        final_slots = self._gather_time(slot_states, final_steps)
        final_slot_mask = self._gather_time(batch.slot_mask, final_steps)
        final_observations = self._gather_time(batch.observations, final_steps)

        source = self._gather_slots(final_slots, batch.source_slots)
        target = self._gather_slots(final_slots, batch.target_slots)
        intervention = self.intervention_fusion(
            torch.cat(
                [
                    self.intervention_type_embedding(batch.intervention_types),
                    source,
                    target,
                    self.intervention_parameter_encoder(batch.intervention_parameters),
                ],
                dim=-1,
            )
        )
        transition = self.transition_input(torch.cat([final_graph, intervention], dim=-1))
        for block in self.transition_blocks:
            transition = block(transition)
        transition = self.transition_norm(transition)

        conditioned_slots = self.slot_output_norm(
            final_slots + self.slot_condition(transition).unsqueeze(1)
        )
        decoded = self._apply_domain_modules(
            self.domain_decoders,
            conditioned_slots,
            batch.domain_ids,
        )
        delta_mean, raw_state_log_variance = decoded.chunk(2, dim=-1)
        next_state_mean = final_observations + delta_mean
        next_state_log_variance = raw_state_log_variance.clamp(
            min=config.log_variance_min, max=config.log_variance_max
        )
        final_mask = final_slot_mask.unsqueeze(-1).to(next_state_mean.dtype)
        next_state_mean = next_state_mean * final_mask
        next_state_log_variance = next_state_log_variance * final_mask

        raw_effect = self.effect_head(transition)
        effect_mean, raw_effect_log_variance = raw_effect.chunk(2, dim=-1)
        effect_log_variance = raw_effect_log_variance.clamp(
            min=config.log_variance_min, max=config.log_variance_max
        )
        proposal_embedding = F.normalize(self.proposal_projection(transition), dim=-1)
        return MetaWorldOutput(
            next_state_mean=next_state_mean,
            next_state_log_variance=next_state_log_variance,
            effect_mean=effect_mean,
            effect_log_variance=effect_log_variance,
            proposal_embedding=proposal_embedding,
            final_slot_states=final_slots,
            transition_state=transition,
        )

    def trainable_parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters() if parameter.requires_grad)
