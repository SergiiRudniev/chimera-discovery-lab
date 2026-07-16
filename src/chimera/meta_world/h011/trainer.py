"""H011 trainer with direct paired response consistency."""

from __future__ import annotations

import math
from typing import cast

from torch import nn

from chimera.meta_world.batch import MetaWorldBatch
from chimera.meta_world.config import MetaWorldTrainingConfig
from chimera.meta_world.h002.objectives import h002_loss
from chimera.meta_world.h002.trainer import H002Trainer
from chimera.meta_world.h011.objectives import paired_response_consistency
from chimera.meta_world.model import MetaWorldOutput


class H011Trainer(H002Trainer):
    def __init__(
        self,
        model: nn.Module,
        config: MetaWorldTrainingConfig,
        *,
        response_consistency_weight: float,
        uncertainty_consistency_fraction: float,
    ) -> None:
        super().__init__(model, config)
        self.response_consistency_weight = response_consistency_weight
        self.uncertainty_consistency_fraction = uncertainty_consistency_fraction

    def train_step(self, batch: MetaWorldBatch) -> dict[str, float]:
        self.model.train()
        device_batch = batch.to(self.device)
        self.optimizer.zero_grad(set_to_none=True)
        with self._autocast():
            output = cast(MetaWorldOutput, self.model(device_batch))
            losses = h002_loss(output, device_batch, self.config)
            consistency = paired_response_consistency(
                output,
                device_batch.mechanism_ids,
                uncertainty_fraction=self.uncertainty_consistency_fraction,
            )
            losses.update(consistency)
            losses["loss"] = losses["loss"] + (
                self.response_consistency_weight
                * consistency["response_consistency_loss"]
            )
        losses["loss"].backward()  # type: ignore[no-untyped-call]
        gradient_norm = nn.utils.clip_grad_norm_(
            self.model.parameters(), self.config.max_grad_norm
        )
        self.optimizer.step()
        self._update_ema()
        metrics = {
            name: float(value.detach().float().cpu()) for name, value in losses.items()
        }
        metrics["gradient_norm"] = float(gradient_norm.detach().float().cpu())
        if not all(math.isfinite(value) for value in metrics.values()):
            raise FloatingPointError("non-finite H011 training metric")
        return metrics
