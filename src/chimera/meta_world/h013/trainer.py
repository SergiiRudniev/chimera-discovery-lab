"""H013 trainer with matched paired-transition supervision."""

from __future__ import annotations

import math
from typing import cast

from torch import nn

from chimera.meta_world.batch import MetaWorldBatch
from chimera.meta_world.config import MetaWorldTrainingConfig
from chimera.meta_world.h002.trainer import H002Trainer
from chimera.meta_world.h013.objectives import h013_loss
from chimera.meta_world.model import MetaWorldOutput


class H013Trainer(H002Trainer):
    """Use identical factual/no-op/delta losses for both matched H013 arms."""

    def __init__(
        self,
        model: nn.Module,
        config: MetaWorldTrainingConfig,
        *,
        no_op_state_weight: float,
        intervention_delta_weight: float,
    ) -> None:
        super().__init__(model, config)
        self.no_op_state_weight = no_op_state_weight
        self.intervention_delta_weight = intervention_delta_weight

    def train_step(self, batch: MetaWorldBatch) -> dict[str, float]:
        self.model.train()
        device_batch = batch.to(self.device)
        self.optimizer.zero_grad(set_to_none=True)
        with self._autocast():
            output = cast(MetaWorldOutput, self.model(device_batch))
            losses = h013_loss(
                output,
                device_batch,
                self.config,
                no_op_state_weight=self.no_op_state_weight,
                intervention_delta_weight=self.intervention_delta_weight,
            )
        losses["loss"].backward()  # type: ignore[no-untyped-call]
        gradient_norm = nn.utils.clip_grad_norm_(
            self.model.parameters(),
            self.config.max_grad_norm,
        )
        self.optimizer.step()
        self._update_ema()
        metrics = {
            name: float(value.detach().float().cpu())
            for name, value in losses.items()
        }
        metrics["gradient_norm"] = float(gradient_norm.detach().float().cpu())
        if not all(math.isfinite(value) for value in metrics.values()):
            raise FloatingPointError("non-finite H013 training metric")
        return metrics
