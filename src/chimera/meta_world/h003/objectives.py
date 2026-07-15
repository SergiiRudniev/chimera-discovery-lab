"""Closed-loop prediction and cross-batch discrimination losses for H003."""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import Tensor

from chimera.meta_world.batch import MetaWorldBatch
from chimera.meta_world.config import MetaWorldTrainingConfig
from chimera.meta_world.model import MetaWorldOutput


class MechanismMemoryQueue:
    """Bounded detached embedding queue used only as a hard-negative pool."""

    def __init__(self, minimum_entries: int, maximum_entries: int) -> None:
        if minimum_entries <= 0 or maximum_entries < minimum_entries:
            raise ValueError("invalid mechanism queue bounds")
        self.minimum_entries = minimum_entries
        self.maximum_entries = maximum_entries
        self._embeddings: Tensor | None = None
        self._keys: Tensor | None = None

    @property
    def size(self) -> int:
        return 0 if self._keys is None else int(self._keys.shape[0])

    def candidates(self) -> tuple[Tensor | None, Tensor | None]:
        if self.size < self.minimum_entries:
            return None, None
        return self._embeddings, self._keys

    @torch.no_grad()
    def update(self, embeddings: Tensor, mechanism_keys: Tensor) -> None:
        if embeddings.ndim != 2 or mechanism_keys.shape != (embeddings.shape[0],):
            raise ValueError("queue entries must have aligned embedding and key shapes")
        detached_embeddings = F.normalize(embeddings.detach().float(), dim=-1)
        detached_keys = mechanism_keys.detach().long()
        if self._embeddings is None or self._keys is None:
            combined_embeddings = detached_embeddings
            combined_keys = detached_keys
        else:
            combined_embeddings = torch.cat([self._embeddings, detached_embeddings], dim=0)
            combined_keys = torch.cat([self._keys, detached_keys], dim=0)
        self._embeddings = combined_embeddings[-self.maximum_entries :]
        self._keys = combined_keys[-self.maximum_entries :]


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
    weights = mask.to(per_value.dtype)
    return (per_value * weights).sum() / weights.sum().clamp_min(1)


def _cross_batch_alignment(
    embeddings: Tensor,
    mechanism_keys: Tensor,
    margin: float,
    queued_embeddings: Tensor | None,
    queued_keys: Tensor | None,
) -> Tensor:
    normalized = F.normalize(embeddings.float(), dim=-1)
    current_similarity = normalized @ normalized.transpose(0, 1)
    diagonal = torch.eye(
        current_similarity.shape[0],
        dtype=torch.bool,
        device=current_similarity.device,
    )
    same_current = mechanism_keys[:, None] == mechanism_keys[None, :]
    positives = same_current & ~diagonal
    positive_count = positives.sum(dim=1).clamp_min(1)
    positive_similarity = (current_similarity * positives).sum(dim=1) / positive_count
    positive_loss = torch.where(
        positives.any(dim=1),
        1.0 - positive_similarity,
        torch.zeros_like(positive_similarity),
    ).mean()

    candidate_embeddings = normalized
    candidate_keys = mechanism_keys
    if queued_embeddings is not None and queued_keys is not None:
        candidate_embeddings = torch.cat(
            [candidate_embeddings, queued_embeddings.to(normalized.device)],
            dim=0,
        )
        candidate_keys = torch.cat(
            [candidate_keys, queued_keys.to(mechanism_keys.device)],
            dim=0,
        )
    candidate_similarity = normalized @ candidate_embeddings.transpose(0, 1)
    negatives = mechanism_keys[:, None] != candidate_keys[None, :]
    if bool(negatives.any()):
        hardest_negative = candidate_similarity.masked_fill(
            ~negatives,
            torch.finfo(candidate_similarity.dtype).min,
        ).amax(dim=1)
        negative_loss = torch.relu(
            margin + hardest_negative - positive_similarity
        ).mean()
    else:
        negative_loss = candidate_similarity.sum() * 0.0

    embedding_scale = 1.0 / math.sqrt(normalized.shape[-1])
    standard_deviation = normalized.std(dim=0, unbiased=False)
    anti_collapse = torch.relu(embedding_scale - standard_deviation).mean()
    anti_collapse = anti_collapse / embedding_scale
    return positive_loss + negative_loss + anti_collapse


def h003_closed_loop_loss(
    outputs: list[MetaWorldOutput],
    windows: list[MetaWorldBatch],
    mechanism_keys: Tensor,
    training: MetaWorldTrainingConfig,
    queue: MechanismMemoryQueue,
) -> tuple[dict[str, Tensor], Tensor]:
    """Aggregate equally weighted autoregressive losses across the frozen horizon."""

    if not outputs or len(outputs) != len(windows):
        raise ValueError("closed-loop outputs and windows must have equal non-zero length")
    state_losses: list[Tensor] = []
    effect_losses: list[Tensor] = []
    variance_losses: list[Tensor] = []
    for output, window in zip(outputs, windows, strict=True):
        state_losses.append(
            _masked_gaussian_nll(
                output.next_state_mean,
                output.next_state_log_variance,
                window.next_observations,
                window.next_observation_mask,
            )
        )
        effect_weights = torch.cat(
            [
                torch.ones_like(window.effect_targets[:, :-1]),
                torch.full_like(
                    window.effect_targets[:, -1:],
                    training.primary_effect_weight,
                ),
            ],
            dim=1,
        )
        effect_losses.append(
            _masked_gaussian_nll(
                output.effect_mean,
                output.effect_log_variance,
                window.effect_targets,
                effect_weights,
            )
        )
        variance_losses.append(
            torch.relu(
                1.0 - output.transition_state.float().std(dim=0, unbiased=False)
            ).mean()
        )
    next_state = torch.stack(state_losses).mean()
    effect = torch.stack(effect_losses).mean()
    variance = torch.stack(variance_losses).mean()
    mean_embedding = F.normalize(
        torch.stack([output.proposal_embedding.float() for output in outputs]).mean(dim=0),
        dim=-1,
    )
    queued_embeddings, queued_keys = queue.candidates()
    alignment = _cross_batch_alignment(
        mean_embedding,
        mechanism_keys,
        training.alignment_margin,
        queued_embeddings,
        queued_keys,
    )
    total = (
        training.next_state_weight * next_state
        + training.effect_weight * effect
        + training.alignment_weight * alignment
        + training.variance_weight * variance
    )
    return (
        {
            "loss": total,
            "closed_loop_state_loss": next_state,
            "closed_loop_effect_loss": effect,
            "alignment_loss": alignment,
            "variance_loss": variance,
        },
        mean_embedding,
    )
