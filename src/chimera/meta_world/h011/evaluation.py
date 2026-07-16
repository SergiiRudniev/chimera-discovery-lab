"""Paired response disagreement diagnostics for H011."""

from __future__ import annotations

from typing import cast

import torch

from chimera.meta_world.h002.model import RelationalSequenceWorldModel
from chimera.meta_world.h002.windows import GeneratedSequenceSample
from chimera.meta_world.h011.windows import make_paired_response_window
from chimera.meta_world.model import MetaWorldOutput


@torch.no_grad()
def evaluate_paired_response_disagreement(
    model: RelationalSequenceWorldModel,
    sample: GeneratedSequenceSample,
    *,
    context_steps: int,
) -> dict[str, float]:
    model.eval()
    mean_differences: list[torch.Tensor] = []
    uncertainty_differences: list[torch.Tensor] = []
    for step in range(sample.batch.observations.shape[1] - 1):
        window = make_paired_response_window(sample, step, context_steps)
        output = cast(MetaWorldOutput, model(window))
        for key in torch.unique(window.mechanism_ids):
            members = window.mechanism_ids == key
            if int(members.sum()) != 2:
                continue
            mean_values = output.effect_mean[members, -1]
            uncertainty_values = output.effect_log_variance[members, -1]
            mean_differences.append((mean_values[0] - mean_values[1]).abs())
            uncertainty_differences.append(
                (uncertainty_values[0] - uncertainty_values[1]).abs()
            )
    if not mean_differences or not uncertainty_differences:
        raise ValueError("paired response evaluation requires complete renderer pairs")
    return {
        "paired_effect_mean_disagreement": float(
            torch.stack(mean_differences).mean().cpu()
        ),
        "paired_effect_uncertainty_disagreement": float(
            torch.stack(uncertainty_differences).mean().cpu()
        ),
    }
