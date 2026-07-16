"""Matched factual, no-op and intervention-delta objectives for H013."""

from __future__ import annotations

import torch
from torch import Tensor

from chimera.meta_world.batch import MetaWorldBatch
from chimera.meta_world.config import MetaWorldTrainingConfig
from chimera.meta_world.h002.objectives import h002_loss
from chimera.meta_world.model import MetaWorldOutput


def _masked_gaussian_nll(
    mean: Tensor,
    log_variance: Tensor,
    target: Tensor,
    mask: Tensor,
) -> Tensor:
    residual = target.float() - mean.float()
    log_variance_float = log_variance.float()
    per_value = 0.5 * (
        log_variance_float + residual.square() * torch.exp(-log_variance_float)
    )
    weight = mask.to(per_value.dtype)
    return (per_value * weight).sum() / weight.sum().clamp_min(1)


def h013_loss(
    output: MetaWorldOutput,
    batch: MetaWorldBatch,
    config: MetaWorldTrainingConfig,
    *,
    no_op_state_weight: float,
    intervention_delta_weight: float,
) -> dict[str, Tensor]:
    """Add equal auxiliary losses to both parameter-matched transition arms."""

    losses = h002_loss(output, batch, config)
    zero = losses["loss"] * 0.0
    if no_op_state_weight == 0.0 and intervention_delta_weight == 0.0:
        losses["no_op_state_loss"] = zero
        losses["intervention_delta_loss"] = zero
        return losses
    no_op_target = batch.counterfactual_no_op_observations
    no_op_mean = output.counterfactual_no_op_state_mean
    no_op_log_variance = output.counterfactual_no_op_state_log_variance
    delta_mean = output.intervention_state_delta_mean
    delta_log_variance = output.intervention_state_delta_log_variance
    if (
        no_op_target is None
        or no_op_mean is None
        or no_op_log_variance is None
        or delta_mean is None
        or delta_log_variance is None
    ):
        raise ValueError("H013 paired transition tensors are required")
    no_op_loss = _masked_gaussian_nll(
        no_op_mean,
        no_op_log_variance,
        no_op_target,
        batch.next_observation_mask,
    )
    delta_loss = _masked_gaussian_nll(
        delta_mean,
        delta_log_variance,
        batch.next_observations - no_op_target,
        batch.next_observation_mask,
    )
    losses["loss"] = (
        losses["loss"]
        + no_op_state_weight * no_op_loss
        + intervention_delta_weight * delta_loss
    )
    losses["no_op_state_loss"] = no_op_loss
    losses["intervention_delta_loss"] = delta_loss
    return losses
