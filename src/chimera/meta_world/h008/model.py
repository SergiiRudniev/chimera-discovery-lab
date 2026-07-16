"""Relational world model with an algebraically constrained effect channel."""

from __future__ import annotations

from dataclasses import replace

import torch

from chimera.meta_world.batch import MetaWorldBatch
from chimera.meta_world.config import MetaWorldModelConfig
from chimera.meta_world.h002 import RelationalSequenceWorldModel
from chimera.meta_world.model import MetaWorldOutput


class DirectOutcomeRelationalWorldModel(RelationalSequenceWorldModel):
    """Keep matched outcome arithmetic in FP32 under BF16 model autocast."""

    def forward(self, batch: MetaWorldBatch) -> MetaWorldOutput:
        raw = super().forward(batch)
        return replace(
            raw,
            effect_mean=raw.effect_mean.float(),
            effect_log_variance=raw.effect_log_variance.float(),
        )


class CounterfactualRelationalWorldModel(DirectOutcomeRelationalWorldModel):
    """Interpret the fourth raw outcome channel as no-op utility."""

    def __init__(self, config: MetaWorldModelConfig) -> None:
        if config.effect_dimensions != 4:
            raise ValueError("H008 requires exactly four raw outcome channels")
        super().__init__(config)

    def forward(self, batch: MetaWorldBatch) -> MetaWorldOutput:
        raw = super().forward(batch)
        factual_means = raw.effect_mean[:, :3]
        no_op_mean = raw.effect_mean[:, 3:4]
        intervention_effect = factual_means[:, :1] - no_op_mean
        factual_log_variances = raw.effect_log_variance[:, :3]
        no_op_log_variance = raw.effect_log_variance[:, 3:4]
        intervention_effect_log_variance = torch.logaddexp(
            factual_log_variances[:, :1],
            no_op_log_variance,
        ).clamp(
            min=self.config.log_variance_min,
            max=self.config.log_variance_max,
        )
        return replace(
            raw,
            effect_mean=torch.cat(
                [factual_means, intervention_effect],
                dim=-1,
            ),
            effect_log_variance=torch.cat(
                [factual_log_variances, intervention_effect_log_variance],
                dim=-1,
            ),
            counterfactual_no_op_mean=no_op_mean,
            counterfactual_no_op_log_variance=no_op_log_variance,
        )
