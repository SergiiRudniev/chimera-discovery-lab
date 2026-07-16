"""Numerical identity and no-op utility audit for H008 checkpoints."""

from __future__ import annotations

from typing import Protocol

import torch
from torch import Tensor

from chimera.meta_world.batch import MetaWorldBatch
from chimera.meta_world.h002 import GeneratedSequenceSample, make_transition_window
from chimera.meta_world.model import MetaWorldOutput


class PredictionRuntime(Protocol):
    """Minimum checkpoint prediction surface used by the H008 audit."""

    device: torch.device

    def predict(self, batch: MetaWorldBatch) -> MetaWorldOutput:
        ...


@torch.no_grad()
def evaluate_counterfactual_structure(
    runtime: PredictionRuntime,
    sample: GeneratedSequenceSample,
    *,
    context_steps: int,
) -> dict[str, float | None]:
    """Audit derived no-op utility and the explicit counterfactual identity."""

    squared_error = 0.0
    values = 0
    targets: list[Tensor] = []
    identity_residual = 0.0
    explicit_counterfactual = False
    for prediction_step in range(sample.batch.observations.shape[1] - 1):
        window = make_transition_window(
            sample,
            prediction_step=prediction_step,
            context_steps=context_steps,
        )
        output = runtime.predict(window)
        target = (
            window.effect_targets[:, 0] - window.effect_targets[:, 3]
        ).to(runtime.device)
        no_op_mean = output.counterfactual_no_op_mean
        if no_op_mean is None:
            prediction = output.effect_mean[:, 0] - output.effect_mean[:, 3]
        else:
            explicit_counterfactual = True
            prediction = no_op_mean[:, 0]
            residual = (
                output.effect_mean[:, 0]
                - output.effect_mean[:, 3]
                - prediction
            ).abs()
            identity_residual = max(
                identity_residual,
                float(residual.max().float().cpu()),
            )
        difference = prediction.float() - target.float()
        squared_error += float(difference.square().sum().cpu())
        values += int(target.numel())
        targets.append(target.detach().float())
    concatenated_targets = torch.cat(targets)
    scale = concatenated_targets.std(unbiased=False).clamp_min(1e-6)
    rmse = (squared_error / max(values, 1)) ** 0.5
    return {
        "no_op_utility_rmse": rmse,
        "no_op_utility_nrmse": rmse / float(scale.cpu()),
        "counterfactual_identity_maximum_absolute_residual": (
            identity_residual if explicit_counterfactual else None
        ),
    }
