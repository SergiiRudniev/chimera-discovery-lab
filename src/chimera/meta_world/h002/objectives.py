"""H002 losses with metadata-free model inputs and evaluator-only alignment labels."""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import Tensor

from chimera.meta_world.batch import MetaWorldBatch
from chimera.meta_world.config import MetaWorldTrainingConfig
from chimera.meta_world.model import MetaWorldOutput


def _masked_gaussian_nll(
    mean: Tensor,
    log_variance: Tensor,
    target: Tensor,
    mask: Tensor,
) -> Tensor:
    mean_float = mean.float()
    target_float = target.float()
    log_variance_float = log_variance.float()
    per_value = 0.5 * (
        log_variance_float
        + (target_float - mean_float).square() * torch.exp(-log_variance_float)
    )
    weight = mask.to(per_value.dtype)
    return (per_value * weight).sum() / weight.sum().clamp_min(1)


def _mechanism_alignment(
    embeddings: Tensor,
    mechanism_ids: Tensor,
    margin: float,
) -> Tensor:
    normalized = F.normalize(embeddings.float(), dim=-1)
    similarity = normalized @ normalized.transpose(0, 1)
    diagonal = torch.eye(similarity.shape[0], dtype=torch.bool, device=similarity.device)
    same_mechanism = mechanism_ids[:, None] == mechanism_ids[None, :]
    positives = same_mechanism & ~diagonal
    negatives = ~same_mechanism
    zero = similarity.sum() * 0.0
    positive_loss = (1.0 - similarity[positives]).mean() if bool(positives.any()) else zero
    negative_loss = (
        torch.relu(similarity[negatives] - margin).mean()
        if bool(negatives.any())
        else zero
    )
    embedding_scale = 1.0 / math.sqrt(normalized.shape[-1])
    standard_deviation = normalized.std(dim=0, unbiased=False)
    anti_collapse = torch.relu(embedding_scale - standard_deviation).mean()
    anti_collapse = anti_collapse / embedding_scale
    return positive_loss + negative_loss + anti_collapse


def _variance_floor(embeddings: Tensor) -> Tensor:
    standard_deviation = embeddings.float().std(dim=0, unbiased=False)
    return torch.relu(1.0 - standard_deviation).mean()


def h002_loss(
    output: MetaWorldOutput,
    batch: MetaWorldBatch,
    config: MetaWorldTrainingConfig,
) -> dict[str, Tensor]:
    """Return transition, outcome, alignment and anti-collapse loss components."""

    next_state = _masked_gaussian_nll(
        output.next_state_mean,
        output.next_state_log_variance,
        batch.next_observations,
        batch.next_observation_mask,
    )
    effect = _masked_gaussian_nll(
        output.effect_mean,
        output.effect_log_variance,
        batch.effect_targets,
        torch.cat(
            [
                torch.ones_like(batch.effect_targets[:, :-1]),
                torch.full_like(
                    batch.effect_targets[:, -1:],
                    config.primary_effect_weight,
                ),
            ],
            dim=1,
        ),
    )
    alignment = _mechanism_alignment(
        output.proposal_embedding,
        batch.mechanism_ids,
        config.alignment_margin,
    )
    variance = _variance_floor(output.transition_state)
    total = (
        config.next_state_weight * next_state
        + config.effect_weight * effect
        + config.alignment_weight * alignment
        + config.variance_weight * variance
    )
    return {
        "loss": total,
        "next_state_loss": next_state,
        "effect_loss": effect,
        "alignment_loss": alignment,
        "variance_loss": variance,
    }
