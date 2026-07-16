"""Renderer-paired intervention-response consistency objective."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor

from chimera.meta_world.model import MetaWorldOutput


def paired_response_consistency(
    output: MetaWorldOutput,
    pair_keys: Tensor,
    *,
    uncertainty_fraction: float,
) -> dict[str, Tensor]:
    """Match primary effect mean and uncertainty inside each renderer pair."""

    means = output.effect_mean[:, -1].float()
    log_variances = output.effect_log_variance[:, -1].float()
    if pair_keys.ndim != 1 or pair_keys.shape[0] != means.shape[0]:
        raise ValueError("pair_keys must have shape [batch]")
    if not 0.0 <= uncertainty_fraction <= 1.0:
        raise ValueError("uncertainty_fraction must be in [0, 1]")
    unique_keys = torch.unique(pair_keys)
    mean_losses: list[Tensor] = []
    uncertainty_losses: list[Tensor] = []
    for key in unique_keys:
        members = pair_keys == key
        if int(members.sum()) < 2:
            continue
        pair_means = means[members]
        pair_log_variances = log_variances[members]
        mean_losses.append(
            F.smooth_l1_loss(
                pair_means,
                pair_means.mean().expand_as(pair_means),
            )
        )
        uncertainty_losses.append(
            F.smooth_l1_loss(
                pair_log_variances,
                pair_log_variances.mean().expand_as(pair_log_variances),
            )
        )
    zero = means.sum() * 0.0
    mean_loss = torch.stack(mean_losses).mean() if mean_losses else zero
    uncertainty_loss = (
        torch.stack(uncertainty_losses).mean() if uncertainty_losses else zero
    )
    return {
        "response_consistency_loss": mean_loss
        + uncertainty_fraction * uncertainty_loss,
        "response_mean_consistency_loss": mean_loss,
        "response_uncertainty_consistency_loss": uncertainty_loss,
    }
