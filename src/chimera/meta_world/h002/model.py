"""Relational sequence model and matched temporal baseline for H002."""

from __future__ import annotations

from typing import cast

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from chimera.meta_world.batch import MetaWorldBatch
from chimera.meta_world.config import MetaWorldModelConfig
from chimera.meta_world.model import MetaWorldOutput, SpatialBlock, TransitionBlock


class RelationalSequenceWorldModel(nn.Module):
    """Track object states through time before applying a relational intervention."""

    def __init__(self, config: MetaWorldModelConfig) -> None:
        super().__init__()
        self.config = config
        hidden = config.hidden_dim
        self.slot_encoder = nn.Sequential(
            nn.Linear(config.observation_features * 2, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
        )
        self.history_action_encoder = nn.Sequential(
            nn.Linear(config.intervention_parameters + 1, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
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
        self.slot_temporal_encoder = nn.TransformerEncoder(
            temporal_layer,
            num_layers=config.temporal_layers,
            norm=nn.LayerNorm(hidden),
            enable_nested_tensor=False,
        )
        self.intervention_type_embedding = nn.Embedding(
            config.intervention_types,
            hidden,
        )
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
        self.role_encoder = nn.Sequential(
            nn.Linear(2, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
        )
        self.global_condition = nn.Linear(hidden, hidden)
        self.intervention_spatial = SpatialBlock(config)
        self.slot_output_norm = nn.LayerNorm(hidden)
        self.state_head = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.SiLU(),
            nn.Linear(hidden, config.observation_features * 2),
        )
        self.effect_head = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.SiLU(),
            nn.Linear(hidden, config.effect_dimensions * 2),
        )
        self.mechanism_projection = nn.Linear(hidden, hidden)
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

    def _validate_contract(self, batch: MetaWorldBatch) -> None:
        config = self.config
        if batch.observations.shape[1:] != (
            config.context_steps,
            config.max_slots,
            config.observation_features,
        ):
            raise ValueError("batch observations do not match the H002 relational model")
        if batch.relations.shape[-1] != config.relation_features:
            raise ValueError("batch relation feature count does not match the model")
        if batch.intervention_parameters.shape[-1] != config.intervention_parameters:
            raise ValueError("batch intervention parameter count does not match the model")
        if torch.any(batch.domain_ids != 0):
            raise ValueError("H002 relational model forbids service-domain IDs")
        if batch.action_history is None or batch.action_target_history is None:
            raise ValueError("H002 relational model requires causal action histories")
        if batch.action_history.shape[-1] != config.intervention_parameters:
            raise ValueError("action history parameter count does not match the model")

    def forward(self, batch: MetaWorldBatch) -> MetaWorldOutput:
        batch.validate()
        self._validate_contract(batch)
        config = self.config
        batch_size = batch.batch_size
        slot_input = torch.cat(
            [batch.observations, batch.observation_mask.to(batch.observations.dtype)],
            dim=-1,
        )
        slot_states = self.slot_encoder(slot_input)
        assert batch.action_history is not None
        assert batch.action_target_history is not None
        history_parameters = batch.action_history[:, :, None, :].expand(
            -1,
            -1,
            config.max_slots,
            -1,
        )
        history_context = torch.cat(
            [history_parameters, batch.action_target_history.unsqueeze(-1)],
            dim=-1,
        )
        slot_states = slot_states + self.history_action_encoder(history_context)
        slot_states = slot_states * batch.slot_mask.unsqueeze(-1).to(slot_states.dtype)
        flat_states = slot_states.reshape(
            batch_size * config.context_steps,
            config.max_slots,
            config.hidden_dim,
        )
        flat_relations = batch.relations.reshape(
            batch_size * config.context_steps,
            config.max_slots,
            config.max_slots,
            config.relation_features,
        )
        flat_mask = batch.slot_mask.reshape(
            batch_size * config.context_steps,
            config.max_slots,
        )
        for block in self.spatial_blocks:
            flat_states = block(flat_states, flat_relations, flat_mask)
        spatial_states = self.spatial_norm(flat_states).reshape(
            batch_size,
            config.context_steps,
            config.max_slots,
            config.hidden_dim,
        )
        active = batch.slot_mask.unsqueeze(-1).to(spatial_states.dtype)
        spatial_states = spatial_states * active
        temporal_input = spatial_states + self.time_embedding[:, :, None, :] * active
        temporal_input = temporal_input.permute(0, 2, 1, 3).reshape(
            batch_size * config.max_slots,
            config.context_steps,
            config.hidden_dim,
        )
        causal_mask = torch.triu(
            torch.ones(
                config.context_steps,
                config.context_steps,
                dtype=torch.bool,
                device=temporal_input.device,
            ),
            diagonal=1,
        )
        temporal_states = self.slot_temporal_encoder(
            temporal_input,
            mask=causal_mask,
        ).reshape(
            batch_size,
            config.max_slots,
            config.context_steps,
            config.hidden_dim,
        )
        temporal_states = temporal_states.permute(0, 2, 1, 3) * active
        final_steps = batch.time_mask.sum(dim=1) - 1
        final_slots = self._gather_time(temporal_states, final_steps)
        final_slot_mask = self._gather_time(batch.slot_mask, final_steps)
        final_relations = self._gather_time(batch.relations, final_steps)
        final_observations = self._gather_time(batch.observations, final_steps)
        slot_weight = final_slot_mask.unsqueeze(-1).to(final_slots.dtype)
        final_graph = (final_slots * slot_weight).sum(dim=1) / slot_weight.sum(
            dim=1
        ).clamp_min(1)

        source = self._gather_slots(final_slots, batch.source_slots)
        target = self._gather_slots(final_slots, batch.target_slots)
        intervention = self.intervention_fusion(
            torch.cat(
                [
                    self.intervention_type_embedding(batch.intervention_types),
                    source,
                    target,
                    self.intervention_parameter_encoder(
                        batch.intervention_parameters
                    ),
                ],
                dim=-1,
            )
        )
        transition = self.transition_input(
            torch.cat([final_graph, intervention], dim=-1)
        )
        for block in self.transition_blocks:
            transition = block(transition)
        transition = self.transition_norm(transition)

        slot_indices = torch.arange(config.max_slots, device=final_slots.device)[None]
        roles = torch.stack(
            [
                slot_indices == batch.source_slots[:, None],
                slot_indices == batch.target_slots[:, None],
            ],
            dim=-1,
        ).to(final_slots.dtype)
        conditioned_slots = (
            final_slots
            + self.role_encoder(roles)
            + self.global_condition(transition).unsqueeze(1)
        )
        conditioned_slots = self.intervention_spatial(
            conditioned_slots,
            final_relations,
            final_slot_mask,
        )
        conditioned_slots = self.slot_output_norm(conditioned_slots)
        decoded = self.state_head(conditioned_slots)
        delta_mean, raw_state_log_variance = decoded.chunk(2, dim=-1)
        final_mask = final_slot_mask.unsqueeze(-1).to(delta_mean.dtype)
        next_state_mean = (final_observations + delta_mean) * final_mask
        next_state_log_variance = raw_state_log_variance.clamp(
            min=config.log_variance_min,
            max=config.log_variance_max,
        ) * final_mask
        raw_effect = self.effect_head(transition)
        effect_mean, raw_effect_log_variance = raw_effect.chunk(2, dim=-1)
        return MetaWorldOutput(
            next_state_mean=next_state_mean,
            next_state_log_variance=next_state_log_variance,
            effect_mean=effect_mean,
            effect_log_variance=raw_effect_log_variance.clamp(
                min=config.log_variance_min,
                max=config.log_variance_max,
            ),
            proposal_embedding=F.normalize(
                self.mechanism_projection(final_graph),
                dim=-1,
            ),
            final_slot_states=final_slots,
            transition_state=cast(Tensor, transition),
        )

    def trainable_parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters() if parameter.requires_grad)


class TemporalWorldBaseline(nn.Module):
    """Predict dynamics from pooled temporal states without reading relations."""

    def __init__(self, config: MetaWorldModelConfig) -> None:
        super().__init__()
        self.config = config
        hidden = config.hidden_dim
        self.slot_encoder = nn.Sequential(
            nn.Linear(config.observation_features * 2, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
        )
        self.history_action_encoder = nn.Sequential(
            nn.Linear(config.intervention_parameters + 1, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
        )
        self.temporal = nn.GRU(
            input_size=hidden,
            hidden_size=hidden,
            num_layers=max(config.temporal_layers, 1),
            dropout=config.dropout if config.temporal_layers > 1 else 0.0,
            batch_first=True,
        )
        self.action_encoder = nn.Sequential(
            nn.Linear(config.intervention_parameters, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
        )
        self.transition = nn.Sequential(
            nn.LayerNorm(hidden * 4),
            nn.Linear(hidden * 4, hidden * 2),
            nn.SiLU(),
            nn.Dropout(config.dropout),
            nn.Linear(hidden * 2, hidden),
            nn.LayerNorm(hidden),
        )
        self.slot_condition = nn.Linear(hidden, hidden)
        self.state_head = nn.Sequential(
            nn.LayerNorm(hidden),
            nn.Linear(hidden, config.observation_features * 2),
        )
        self.effect_head = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.SiLU(),
            nn.Linear(hidden, config.effect_dimensions * 2),
        )
        self.proposal_projection = nn.Linear(hidden, hidden)

    @staticmethod
    def _gather_time(values: Tensor, steps: Tensor) -> Tensor:
        index = steps.view(steps.shape[0], 1, *([1] * (values.ndim - 2)))
        index = index.expand(steps.shape[0], 1, *values.shape[2:])
        return values.gather(1, index).squeeze(1)

    @staticmethod
    def _gather_slots(values: Tensor, pointers: Tensor) -> Tensor:
        index = pointers[:, None, None].expand(values.shape[0], 1, values.shape[-1])
        return values.gather(1, index).squeeze(1)

    def forward(self, batch: MetaWorldBatch) -> MetaWorldOutput:
        batch.validate()
        config = self.config
        if batch.observations.shape[1:] != (
            config.context_steps,
            config.max_slots,
            config.observation_features,
        ):
            raise ValueError("batch observations do not match the temporal baseline")
        if batch.action_history is None or batch.action_target_history is None:
            raise ValueError("temporal baseline requires causal action histories")
        adapter_input = torch.cat(
            [batch.observations, batch.observation_mask.to(batch.observations.dtype)],
            dim=-1,
        )
        slot_states = self.slot_encoder(adapter_input)
        history_parameters = batch.action_history[:, :, None, :].expand(
            -1,
            -1,
            config.max_slots,
            -1,
        )
        history_context = torch.cat(
            [history_parameters, batch.action_target_history.unsqueeze(-1)],
            dim=-1,
        )
        slot_states = slot_states + self.history_action_encoder(history_context)
        slot_states = slot_states * batch.slot_mask.unsqueeze(-1).to(slot_states.dtype)
        weights = batch.slot_mask.unsqueeze(-1).to(slot_states.dtype)
        graph_states = (slot_states * weights).sum(dim=2) / weights.sum(dim=2).clamp_min(1)
        graph_states = graph_states * batch.time_mask.unsqueeze(-1).to(graph_states.dtype)
        temporal_states, _ = self.temporal(graph_states)
        final_steps = batch.time_mask.sum(dim=1) - 1
        final_graph = self._gather_time(temporal_states, final_steps)
        final_slots = self._gather_time(slot_states, final_steps)
        final_slot_mask = self._gather_time(batch.slot_mask, final_steps)
        final_observations = self._gather_time(batch.observations, final_steps)
        source = self._gather_slots(final_slots, batch.source_slots)
        target = self._gather_slots(final_slots, batch.target_slots)
        transition = self.transition(
            torch.cat(
                [
                    final_graph,
                    source,
                    target,
                    self.action_encoder(batch.intervention_parameters),
                ],
                dim=-1,
            )
        )
        conditioned = final_slots + self.slot_condition(transition).unsqueeze(1)
        decoded = self.state_head(conditioned)
        delta_mean, raw_state_log_variance = decoded.chunk(2, dim=-1)
        mask = final_slot_mask.unsqueeze(-1).to(delta_mean.dtype)
        next_state_mean = (final_observations + delta_mean) * mask
        next_state_log_variance = raw_state_log_variance.clamp(
            min=config.log_variance_min,
            max=config.log_variance_max,
        ) * mask
        raw_effect = self.effect_head(transition)
        effect_mean, raw_effect_log_variance = raw_effect.chunk(2, dim=-1)
        return MetaWorldOutput(
            next_state_mean=next_state_mean,
            next_state_log_variance=next_state_log_variance,
            effect_mean=effect_mean,
            effect_log_variance=raw_effect_log_variance.clamp(
                min=config.log_variance_min,
                max=config.log_variance_max,
            ),
            proposal_embedding=F.normalize(
                self.proposal_projection(transition),
                dim=-1,
            ),
            final_slot_states=final_slots,
            transition_state=cast(Tensor, transition),
        )

    def trainable_parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters() if parameter.requires_grad)
