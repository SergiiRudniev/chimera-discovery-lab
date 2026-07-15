"""Non-relational temporal baseline for the H002 generated-world comparison."""

from __future__ import annotations

from typing import cast

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from chimera.meta_world.batch import MetaWorldBatch
from chimera.meta_world.config import MetaWorldModelConfig
from chimera.meta_world.model import MetaWorldOutput


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
        adapter_input = torch.cat(
            [batch.observations, batch.observation_mask.to(batch.observations.dtype)],
            dim=-1,
        )
        slot_states = self.slot_encoder(adapter_input)
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

