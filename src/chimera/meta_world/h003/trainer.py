"""Autoregressive trainer for CHM-W-H003."""

from __future__ import annotations

import math
from dataclasses import replace
from typing import cast

import torch
from torch import nn

from chimera.meta_world.config import MetaWorldTrainingConfig
from chimera.meta_world.h002.trainer import H002Trainer
from chimera.meta_world.h002.windows import (
    GeneratedSequenceSample,
    make_transition_window,
)
from chimera.meta_world.h003.objectives import (
    MechanismMemoryQueue,
    h003_closed_loop_loss,
)
from chimera.meta_world.model import MetaWorldOutput


class H003Trainer(H002Trainer):
    """Backpropagate through model-generated state histories for four steps."""

    def __init__(
        self,
        model: nn.Module,
        config: MetaWorldTrainingConfig,
        *,
        rollout_horizon: int,
        state_features: int,
        queue_minimum_entries: int,
        queue_maximum_entries: int,
    ) -> None:
        super().__init__(model, config)
        if rollout_horizon <= 1 or state_features <= 0:
            raise ValueError("H003 requires a multi-step horizon and state features")
        self.rollout_horizon = rollout_horizon
        self.state_features = state_features
        self.queue = MechanismMemoryQueue(
            queue_minimum_entries,
            queue_maximum_entries,
        )

    def train_sequence_step(
        self,
        sample: GeneratedSequenceSample,
        *,
        prediction_step: int,
        context_steps: int,
        effect_supervision_mask: torch.Tensor | None = None,
    ) -> dict[str, float]:
        self.model.train()
        generated = sample.batch.to(self.device)
        if prediction_step < 0 or prediction_step + self.rollout_horizon >= (
            generated.observations.shape[1]
        ):
            raise ValueError("prediction step cannot fit the registered rollout horizon")
        device_sample = replace(
            sample,
            batch=generated,
            mechanism_ids=sample.mechanism_ids.to(self.device),
            mechanism_keys=sample.mechanism_keys.to(self.device),
            world_family_ids=sample.world_family_ids.to(self.device),
            renderer_profile_ids=sample.renderer_profile_ids.to(self.device),
            trajectory_indices=sample.trajectory_indices.to(self.device),
        )
        device_effect_supervision = (
            effect_supervision_mask.to(self.device)
            if effect_supervision_mask is not None
            else None
        )
        frames = list(generated.observations.unbind(dim=1))
        outputs: list[MetaWorldOutput] = []
        windows = []
        self.optimizer.zero_grad(set_to_none=True)
        with self._autocast():
            for offset in range(self.rollout_horizon):
                step = prediction_step + offset
                rolling_batch = replace(
                    generated,
                    observations=torch.stack(frames, dim=1),
                )
                rolling_sample = replace(device_sample, batch=rolling_batch)
                window = make_transition_window(
                    rolling_sample,
                    prediction_step=step,
                    context_steps=context_steps,
                )
                output = cast(MetaWorldOutput, self.model(window))
                outputs.append(output)
                windows.append(window)
                rolled_state = torch.zeros_like(output.next_state_mean)
                rolled_state[:, :, : self.state_features] = output.next_state_mean[
                    :, :, : self.state_features
                ]
                active = generated.object_mask[:, step + 1].unsqueeze(-1)
                frames[step + 1] = torch.where(
                    active,
                    rolled_state,
                    frames[step + 1],
                )
            losses, mechanism_embedding = h003_closed_loop_loss(
                outputs,
                windows,
                device_sample.mechanism_keys,
                self.config,
                self.queue,
                device_effect_supervision,
            )
        losses["loss"].backward()  # type: ignore[no-untyped-call]
        gradient_norm = nn.utils.clip_grad_norm_(
            self.model.parameters(),
            self.config.max_grad_norm,
        )
        self.optimizer.step()
        self._update_ema()
        self.queue.update(mechanism_embedding, device_sample.mechanism_keys)
        metrics = {
            name: float(value.detach().float().cpu()) for name, value in losses.items()
        }
        metrics["gradient_norm"] = float(gradient_norm.detach().float().cpu())
        metrics["mechanism_queue_entries"] = float(self.queue.size)
        if device_effect_supervision is not None:
            metrics["effect_supervision_fraction"] = float(
                device_effect_supervision.float().mean().cpu()
            )
        if not all(math.isfinite(value) for value in metrics.values()):
            raise FloatingPointError("non-finite H003 training metric")
        return metrics
