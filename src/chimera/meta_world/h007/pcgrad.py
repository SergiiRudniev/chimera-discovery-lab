"""Deterministic symmetric conflict projection for two task gradients."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

GradientTuple = tuple[Tensor | None, ...]


@dataclass(frozen=True)
class PCGradMetrics:
    """Raw shared-gradient geometry before any projection."""

    cosine: float
    conflict_applied: bool
    state_norm: float
    effect_norm: float
    shared_parameter_tensors: int


def project_task_gradients(
    state_gradients: GradientTuple,
    effect_gradients: GradientTuple,
) -> tuple[GradientTuple, PCGradMetrics]:
    """Project only coordinates shared by both tasks and sum the task updates."""

    if len(state_gradients) != len(effect_gradients) or not state_gradients:
        raise ValueError("task gradient tuples must have equal non-zero length")
    device = next(
        gradient.device
        for gradient in (*state_gradients, *effect_gradients)
        if gradient is not None
    )
    dot = torch.zeros((), dtype=torch.float64, device=device)
    state_square = torch.zeros_like(dot)
    effect_square = torch.zeros_like(dot)
    shared = 0
    for state_gradient, effect_gradient in zip(
        state_gradients,
        effect_gradients,
        strict=True,
    ):
        if state_gradient is None or effect_gradient is None:
            continue
        state_double = state_gradient.detach().double()
        effect_double = effect_gradient.detach().double()
        dot += (state_double * effect_double).sum()
        state_square += state_double.square().sum()
        effect_square += effect_double.square().sum()
        shared += 1
    if shared == 0 or float(state_square.cpu()) == 0.0 or float(effect_square.cpu()) == 0.0:
        raise RuntimeError("PCGrad requires non-zero shared task gradients")
    denominator = (state_square * effect_square).sqrt()
    cosine = float((dot / denominator).cpu())
    conflict = float(dot.cpu()) < 0.0
    state_coefficient = dot / effect_square if conflict else dot.new_zeros(())
    effect_coefficient = dot / state_square if conflict else dot.new_zeros(())
    combined: list[Tensor | None] = []
    for state_gradient, effect_gradient in zip(
        state_gradients,
        effect_gradients,
        strict=True,
    ):
        if state_gradient is None:
            combined.append(effect_gradient)
        elif effect_gradient is None:
            combined.append(state_gradient)
        elif conflict:
            projected_state = state_gradient - state_coefficient.to(
                state_gradient.dtype
            ) * effect_gradient
            projected_effect = effect_gradient - effect_coefficient.to(
                effect_gradient.dtype
            ) * state_gradient
            combined.append(projected_state + projected_effect)
        else:
            combined.append(state_gradient + effect_gradient)
    return (
        tuple(combined),
        PCGradMetrics(
            cosine=cosine,
            conflict_applied=conflict,
            state_norm=float(state_square.sqrt().cpu()),
            effect_norm=float(effect_square.sqrt().cpu()),
            shared_parameter_tensors=shared,
        ),
    )
