"""Parameter-matched direct and factorized transition models for H013."""

from __future__ import annotations

from dataclasses import replace

import torch
from torch import Tensor, nn

from chimera.meta_world.batch import MetaWorldBatch
from chimera.meta_world.config import MetaWorldModelConfig
from chimera.meta_world.h008.model import CounterfactualRelationalWorldModel
from chimera.meta_world.model import MetaWorldOutput


class _DualTransitionWorldModel(CounterfactualRelationalWorldModel):
    """Add a matched action-independent no-op transition head."""

    def __init__(self, config: MetaWorldModelConfig) -> None:
        super().__init__(config)
        self.no_op_state_head = nn.Sequential(
            nn.Linear(config.hidden_dim, config.hidden_dim),
            nn.SiLU(),
            nn.Linear(config.hidden_dim, config.observation_features * 2),
        )

    def _final_observations(self, batch: MetaWorldBatch) -> Tensor:
        final_steps = batch.time_mask.sum(dim=1) - 1
        return self._gather_time(batch.observations, final_steps)

    def _no_op_prediction(
        self,
        raw: MetaWorldOutput,
        batch: MetaWorldBatch,
    ) -> tuple[Tensor, Tensor]:
        decoded = self.no_op_state_head(raw.final_slot_states)
        no_op_delta, no_op_log_variance = decoded.chunk(2, dim=-1)
        final_observations = self._final_observations(batch)
        final_steps = batch.time_mask.sum(dim=1) - 1
        final_mask = self._gather_time(batch.slot_mask, final_steps)
        mask = final_mask.unsqueeze(-1).to(no_op_delta.dtype)
        no_op_mean = (final_observations + no_op_delta) * mask
        return (
            no_op_mean.float(),
            no_op_log_variance.clamp(
                min=self.config.log_variance_min,
                max=self.config.log_variance_max,
            ).float()
            * mask.float(),
        )


class DirectDualTransitionWorldModel(_DualTransitionWorldModel):
    """Predict factual and no-op state directly with matched heads."""

    def forward(self, batch: MetaWorldBatch) -> MetaWorldOutput:
        raw = super().forward(batch)
        no_op_mean, no_op_log_variance = self._no_op_prediction(raw, batch)
        factual_mean = raw.next_state_mean.float()
        factual_log_variance = raw.next_state_log_variance.float()
        delta_mean = factual_mean - no_op_mean
        delta_log_variance = torch.logaddexp(
            factual_log_variance,
            no_op_log_variance,
        ).clamp(
            min=self.config.log_variance_min,
            max=self.config.log_variance_max,
        )
        return replace(
            raw,
            next_state_mean=factual_mean,
            next_state_log_variance=factual_log_variance,
            counterfactual_no_op_state_mean=no_op_mean,
            counterfactual_no_op_state_log_variance=no_op_log_variance,
            intervention_state_delta_mean=delta_mean,
            intervention_state_delta_log_variance=delta_log_variance,
        )


class FactorizedCounterfactualTransitionWorldModel(_DualTransitionWorldModel):
    """Derive factual state exactly as no-op state plus intervention delta."""

    def forward(self, batch: MetaWorldBatch) -> MetaWorldOutput:
        raw = super().forward(batch)
        no_op_mean, no_op_log_variance = self._no_op_prediction(raw, batch)
        final_observations = self._final_observations(batch).float()
        intervention_delta = raw.next_state_mean.float() - final_observations
        intervention_delta_log_variance = raw.next_state_log_variance.float()
        factual_mean = no_op_mean + intervention_delta
        factual_log_variance = torch.logaddexp(
            no_op_log_variance,
            intervention_delta_log_variance,
        ).clamp(
            min=self.config.log_variance_min,
            max=self.config.log_variance_max,
        )
        return replace(
            raw,
            next_state_mean=factual_mean,
            next_state_log_variance=factual_log_variance,
            counterfactual_no_op_state_mean=no_op_mean,
            counterfactual_no_op_state_log_variance=no_op_log_variance,
            intervention_state_delta_mean=intervention_delta,
            intervention_state_delta_log_variance=intervention_delta_log_variance,
        )
