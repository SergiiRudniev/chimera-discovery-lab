"""Effect head conditioned on predicted state response or matched control."""

from __future__ import annotations

from dataclasses import replace
from enum import Enum

import torch
from torch import Tensor, nn

from chimera.meta_world.batch import MetaWorldBatch
from chimera.meta_world.config import MetaWorldModelConfig
from chimera.meta_world.h013.model import DirectDualTransitionWorldModel
from chimera.meta_world.model import MetaWorldOutput


class ResponseSource(str, Enum):
    """The only controlled variable in the matched H014 models."""

    NO_OP_SUBTRACTED = "predicted_factual_minus_predicted_no_op"
    FACTUAL_RESIDUAL = "predicted_factual_minus_final_observation"


class ResponseConditionedEffectWorldModel(DirectDualTransitionWorldModel):
    """Recompute outcomes from transition state plus a predicted response."""

    def __init__(
        self,
        config: MetaWorldModelConfig,
        *,
        response_source: ResponseSource,
    ) -> None:
        super().__init__(config)
        self.response_source = response_source
        response_features = 4 * 2
        self.response_encoder = nn.Sequential(
            nn.LayerNorm(response_features),
            nn.Linear(response_features, config.hidden_dim),
            nn.SiLU(),
            nn.Linear(config.hidden_dim, config.hidden_dim),
        )
        self.response_effect_head = nn.Sequential(
            nn.LayerNorm(config.hidden_dim * 2),
            nn.Linear(config.hidden_dim * 2, config.hidden_dim),
            nn.SiLU(),
            nn.Linear(config.hidden_dim, config.effect_dimensions * 2),
        )

    def _pool_response(self, values: Tensor, mask: Tensor) -> Tensor:
        state = values[:, :, :4].float()
        weight = mask.unsqueeze(-1).to(state.dtype)
        count = weight.sum(dim=1).clamp_min(1.0)
        mean = (state * weight).sum(dim=1) / count
        variance = ((state - mean.unsqueeze(1)).square() * weight).sum(dim=1) / count
        return torch.cat([mean, variance.clamp_min(0.0).sqrt()], dim=-1)

    def forward(self, batch: MetaWorldBatch) -> MetaWorldOutput:
        raw = super().forward(batch)
        no_op_state = raw.counterfactual_no_op_state_mean
        if no_op_state is None:
            raise RuntimeError("H014 requires the H013 no-op state prediction")
        final_observations = self._final_observations(batch).float()
        response = (
            raw.next_state_mean.float() - no_op_state.float()
            if self.response_source is ResponseSource.NO_OP_SUBTRACTED
            else raw.next_state_mean.float() - final_observations
        )
        final_steps = batch.time_mask.sum(dim=1) - 1
        final_mask = self._gather_time(batch.slot_mask, final_steps)
        pooled = self._pool_response(response, final_mask)
        response_state = self.response_encoder(
            pooled.to(raw.transition_state.dtype)
        )
        conditioned = torch.cat([raw.transition_state, response_state], dim=-1)
        effect_raw = self.response_effect_head(conditioned)
        raw_means, raw_log_variances = effect_raw.chunk(2, dim=-1)
        raw_means = raw_means.float()
        raw_log_variances = raw_log_variances.float().clamp(
            min=self.config.log_variance_min,
            max=self.config.log_variance_max,
        )
        factual_means = raw_means[:, :3]
        no_op_utility = raw_means[:, 3:4]
        effect = factual_means[:, :1] - no_op_utility
        factual_log_variances = raw_log_variances[:, :3]
        no_op_log_variance = raw_log_variances[:, 3:4]
        effect_log_variance = torch.logaddexp(
            factual_log_variances[:, :1],
            no_op_log_variance,
        ).clamp(
            min=self.config.log_variance_min,
            max=self.config.log_variance_max,
        )
        return replace(
            raw,
            effect_mean=torch.cat([factual_means, effect], dim=-1),
            effect_log_variance=torch.cat(
                [factual_log_variances, effect_log_variance], dim=-1
            ),
            counterfactual_no_op_mean=no_op_utility,
            counterfactual_no_op_log_variance=no_op_log_variance,
        )
