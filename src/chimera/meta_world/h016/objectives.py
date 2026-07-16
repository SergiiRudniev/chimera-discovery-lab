"""Listwise and pairwise within-state ranking losses."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor

from chimera.meta_world.h016.config import H016RankingTrainingConfig


def h016_ranking_loss(
    logits: Tensor,
    realized_effects: Tensor,
    config: H016RankingTrainingConfig,
) -> dict[str, Tensor]:
    """Match the within-state effect ordering without exposing labels to forward."""

    if logits.ndim != 1 or realized_effects.shape != logits.shape or logits.numel() < 2:
        raise ValueError("H016 ranking tensors must be aligned one-dimensional groups")
    effects = realized_effects.float()
    standardized = (effects - effects.mean()) / effects.std(unbiased=False).clamp_min(
        1e-6
    )
    target_probabilities = torch.softmax(
        standardized / config.listnet_target_temperature,
        dim=0,
    )
    listwise = -(target_probabilities * torch.log_softmax(logits.float(), dim=0)).sum()
    row, column = torch.triu_indices(
        logits.numel(),
        logits.numel(),
        offset=1,
        device=logits.device,
    )
    effect_differences = effects[row] - effects[column]
    retained = effect_differences.abs() >= config.minimum_effect_separation
    if bool(retained.any()):
        signs = effect_differences[retained].sign()
        logit_differences = logits.float()[row[retained]] - logits.float()[column[retained]]
        pairwise = F.softplus(
            -signs * logit_differences / config.pairwise_logit_temperature
        ).mean()
        retained_pairs = retained.sum().to(logits.dtype)
    else:
        pairwise = logits.float().sum() * 0.0
        retained_pairs = torch.zeros((), device=logits.device, dtype=logits.dtype)
    loss = listwise + config.pairwise_weight * pairwise
    return {
        "loss": loss,
        "listwise_loss": listwise,
        "pairwise_loss": pairwise,
        "retained_pairs": retained_pairs,
        "target_effect_standard_deviation": effects.std(unbiased=False),
    }
