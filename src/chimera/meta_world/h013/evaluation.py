"""Frozen factual, no-op and intervention-delta metrics for H013."""

from __future__ import annotations

import math
from typing import Any

import torch
from torch import Tensor

from chimera.meta_world.h002.evaluation import evaluate_h002_model
from chimera.meta_world.h002.windows import (
    GeneratedSequenceSample,
    make_transition_window,
)
from chimera.meta_world.h013.trainer import H013Trainer


def _nrmse(
    squared_error: float,
    values: int,
    targets: list[Tensor],
) -> float:
    target = torch.cat(targets)
    scale = target.std(unbiased=False).clamp_min(1e-6)
    return float(
        math.sqrt(squared_error / max(values, 1)) / float(scale.cpu())
    )


@torch.no_grad()
def evaluate_h013_model(
    trainer: H013Trainer,
    sample: GeneratedSequenceSample,
    *,
    context_steps: int,
    rollout_horizon: int,
) -> dict[str, Any]:
    """Evaluate paired transition metrics without opening any test split."""

    base = evaluate_h002_model(
        trainer,
        sample,
        context_steps=context_steps,
        rollout_horizon=rollout_horizon,
    ).to_dict()
    generated = sample.batch
    if generated.counterfactual_no_op_observations is None:
        raise ValueError("H013 evaluation requires paired no-op transitions")
    no_op_squared_error = 0.0
    delta_squared_error = 0.0
    no_op_values = 0
    delta_values = 0
    no_op_targets: list[Tensor] = []
    delta_targets: list[Tensor] = []
    identity_residual = 0.0
    outcome_identity_residual = 0.0
    explicit_transition = False
    for prediction_step in range(generated.observations.shape[1] - 1):
        window = make_transition_window(
            sample,
            prediction_step=prediction_step,
            context_steps=context_steps,
        )
        output = trainer.predict(window)
        if output.counterfactual_no_op_mean is not None:
            outcome_identity = (
                output.effect_mean[:, :1].float()
                - output.effect_mean[:, 3:4].float()
                - output.counterfactual_no_op_mean.float()
            )
            outcome_identity_residual = max(
                outcome_identity_residual,
                float(outcome_identity.abs().max().cpu()),
            )
        no_op_target = window.counterfactual_no_op_observations
        if no_op_target is None:
            raise RuntimeError("H013 window lost its paired no-op target")
        mask = window.next_observation_mask.to(trainer.device)
        factual_target = window.next_observations.to(trainer.device).float()
        no_op_target = no_op_target.to(trainer.device).float()
        delta_target = factual_target - no_op_target
        no_op_mean = output.counterfactual_no_op_state_mean
        delta_mean = output.intervention_state_delta_mean
        if delta_mean is None:
            delta_mean = output.next_state_mean.float() - no_op_target
        delta_residual = delta_mean.float() - delta_target
        delta_squared_error += float(delta_residual.square()[mask].sum().cpu())
        delta_values += int(mask.sum())
        delta_targets.append(delta_target[mask].detach().cpu())
        if no_op_mean is None:
            continue
        explicit_transition = True
        no_op_residual = no_op_mean.float() - no_op_target
        no_op_squared_error += float(no_op_residual.square()[mask].sum().cpu())
        no_op_values += int(mask.sum())
        no_op_targets.append(no_op_target[mask].detach().cpu())
        identity = output.next_state_mean.float() - no_op_mean.float() - delta_mean.float()
        identity_residual = max(
            identity_residual,
            float(identity[mask].abs().max().cpu()),
        )
    metrics: dict[str, Any] = dict(base)
    metrics.update(
        {
            "no_op_state_nrmse": (
                _nrmse(no_op_squared_error, no_op_values, no_op_targets)
                if explicit_transition
                else None
            ),
            "intervention_state_delta_nrmse": (
                _nrmse(delta_squared_error, delta_values, delta_targets)
            ),
            "factorized_identity_maximum_absolute_residual": (
                identity_residual if explicit_transition else None
            ),
            "outcome_counterfactual_identity_maximum_absolute_residual": (
                outcome_identity_residual
            ),
        }
    )
    return metrics
