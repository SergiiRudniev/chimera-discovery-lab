"""Structural audits for the registered H010 mechanism path."""

from __future__ import annotations

import torch

from chimera.meta_world.batch import MetaWorldBatch
from chimera.meta_world.h002.model import RelationalSequenceWorldModel


@torch.no_grad()
def projection_prediction_delta(
    model: RelationalSequenceWorldModel,
    batch: MetaWorldBatch,
    *,
    perturbation: float = 0.01,
) -> float:
    """Measure whether mechanism-projection parameters affect prediction heads."""

    if perturbation <= 0.0:
        raise ValueError("perturbation must be positive")
    model.eval()
    before = model(batch)
    weight = model.mechanism_projection.weight.detach().clone()
    bias = model.mechanism_projection.bias.detach().clone()
    try:
        model.mechanism_projection.weight.add_(perturbation)
        model.mechanism_projection.bias.add_(perturbation)
        after = model(batch)
    finally:
        model.mechanism_projection.weight.copy_(weight)
        model.mechanism_projection.bias.copy_(bias)
    state_delta = (after.next_state_mean - before.next_state_mean).abs().amax()
    effect_delta = (after.effect_mean - before.effect_mean).abs().amax()
    return float(torch.maximum(state_delta, effect_delta).cpu())
