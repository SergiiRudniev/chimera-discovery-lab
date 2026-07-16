"""Closed-loop trainer with symmetric state/effect conflict projection."""

from __future__ import annotations

import math
from dataclasses import replace
from typing import cast

import torch
from torch import Tensor, nn

from chimera.meta_world.h002.windows import GeneratedSequenceSample, make_transition_window
from chimera.meta_world.h003 import H003Trainer, h003_closed_loop_loss
from chimera.meta_world.h007.pcgrad import project_task_gradients
from chimera.meta_world.model import MetaWorldOutput


class H007Trainer(H003Trainer):
    """Apply the preregistered symmetric PCGrad update before global clipping."""

    def _projected_backward(self, losses: dict[str, Tensor]) -> dict[str, float]:
        parameters = tuple(self.model.parameters())
        state_task = self.config.next_state_weight * losses["closed_loop_state_loss"]
        effect_task = self.config.effect_weight * losses["closed_loop_effect_loss"]
        auxiliary = (
            self.config.alignment_weight * losses["alignment_loss"]
            + self.config.variance_weight * losses["variance_loss"]
        )
        state_gradients = torch.autograd.grad(
            state_task,
            parameters,
            retain_graph=True,
            allow_unused=True,
        )
        effect_gradients = torch.autograd.grad(
            effect_task,
            parameters,
            retain_graph=True,
            allow_unused=True,
        )
        combined, geometry = project_task_gradients(
            state_gradients,
            effect_gradients,
        )
        auxiliary_gradients = torch.autograd.grad(
            auxiliary,
            parameters,
            allow_unused=True,
        )
        for parameter, task_gradient, auxiliary_gradient in zip(
            parameters,
            combined,
            auxiliary_gradients,
            strict=True,
        ):
            gradient = task_gradient
            if auxiliary_gradient is not None:
                gradient = (
                    auxiliary_gradient
                    if gradient is None
                    else gradient + auxiliary_gradient
                )
            parameter.grad = None if gradient is None else gradient.detach()
        return {
            "gradient_cosine": geometry.cosine,
            "gradient_conflict_applied": float(geometry.conflict_applied),
            "state_gradient_norm": geometry.state_norm,
            "effect_gradient_norm": geometry.effect_norm,
            "shared_gradient_parameter_tensors": float(
                geometry.shared_parameter_tensors
            ),
        }

    def train_sequence_step(
        self,
        sample: GeneratedSequenceSample,
        *,
        prediction_step: int,
        context_steps: int,
        effect_supervision_mask: torch.Tensor | None = None,
    ) -> dict[str, float]:
        if effect_supervision_mask is not None:
            raise ValueError("H007 uses shared effect supervision for every trajectory")
        self.model.train()
        generated = sample.batch.to(self.device)
        if prediction_step < 0 or prediction_step + self.rollout_horizon >= (
            generated.observations.shape[1]
        ):
            raise ValueError("prediction step cannot fit the H007 rollout horizon")
        device_sample = replace(
            sample,
            batch=generated,
            mechanism_ids=sample.mechanism_ids.to(self.device),
            mechanism_keys=sample.mechanism_keys.to(self.device),
            world_family_ids=sample.world_family_ids.to(self.device),
            renderer_profile_ids=sample.renderer_profile_ids.to(self.device),
            trajectory_indices=sample.trajectory_indices.to(self.device),
        )
        frames = list(generated.observations.unbind(dim=1))
        outputs: list[MetaWorldOutput] = []
        windows = []
        self.optimizer.zero_grad(set_to_none=True)
        with self._autocast():
            for offset in range(self.rollout_horizon):
                step = prediction_step + offset
                rolling_sample = replace(
                    device_sample,
                    batch=replace(
                        generated,
                        observations=torch.stack(frames, dim=1),
                    ),
                )
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
                frames[step + 1] = torch.where(
                    generated.object_mask[:, step + 1].unsqueeze(-1),
                    rolled_state,
                    frames[step + 1],
                )
            losses, mechanism_embedding = h003_closed_loop_loss(
                outputs,
                windows,
                device_sample.mechanism_keys,
                self.config,
                self.queue,
            )
        gradient_metrics = self._projected_backward(losses)
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
        metrics.update(gradient_metrics)
        metrics["gradient_norm"] = float(gradient_norm.detach().float().cpu())
        metrics["mechanism_queue_entries"] = float(self.queue.size)
        if not all(math.isfinite(value) for value in metrics.values()):
            raise FloatingPointError("non-finite H007 training metric")
        return metrics
