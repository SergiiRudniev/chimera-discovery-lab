"""Candidate scoring and ranking diagnostics for H016."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from numpy.typing import NDArray

from chimera.meta_world.batch import MetaWorldBatch
from chimera.meta_world.h015.evaluation import candidate_batch
from chimera.meta_world.h015.search import InterventionCandidate
from chimera.meta_world.h016.model import WithinStateActionRanker


@dataclass
class H016CandidatePredictor:
    """Frozen adapter for rank-logit and H015 pointwise scoring arms."""

    model: WithinStateActionRanker
    device: torch.device
    use_autocast: bool

    @torch.no_grad()
    def _forward(
        self,
        window: MetaWorldBatch,
        candidates: tuple[InterventionCandidate, ...],
    ) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
        self.model.eval()
        batch = candidate_batch(window, candidates).to(self.device)
        with torch.autocast(
            device_type=self.device.type,
            dtype=torch.bfloat16,
            enabled=self.use_autocast,
        ):
            output = self.model(batch)
        rank_logits = output.rank_logits.float()
        point_means = output.backbone.effect_mean[:, 3].float()
        point_deviations = torch.exp(
            0.5 * output.backbone.effect_log_variance[:, 3].float()
        )
        return tuple(
            value.detach().cpu().numpy().astype(np.float64)
            for value in (rank_logits, point_means, point_deviations)
        )  # type: ignore[return-value]

    def predict_rank(
        self,
        window: MetaWorldBatch,
        candidates: tuple[InterventionCandidate, ...],
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        rank_logits, _, _ = self._forward(window, candidates)
        return rank_logits, np.zeros_like(rank_logits)

    def predict_pointwise(
        self,
        window: MetaWorldBatch,
        candidates: tuple[InterventionCandidate, ...],
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        _, means, deviations = self._forward(window, candidates)
        return means, deviations


def _average_ranks(values: NDArray[np.float64]) -> NDArray[np.float64]:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(values.size, dtype=np.float64)
    position = 0
    while position < values.size:
        end = position + 1
        while end < values.size and values[order[end]] == values[order[position]]:
            end += 1
        ranks[order[position:end]] = 0.5 * (position + end - 1)
        position = end
    return ranks


def spearman_rank_correlation(
    predictions: NDArray[np.float64],
    targets: NDArray[np.float64],
) -> float:
    """Compute tie-aware Spearman correlation with a defined constant case."""

    if predictions.shape != targets.shape or predictions.ndim != 1:
        raise ValueError("H016 Spearman inputs must be aligned vectors")
    predicted_ranks = _average_ranks(predictions)
    target_ranks = _average_ranks(targets)
    if predicted_ranks.std() <= 1e-12 or target_ranks.std() <= 1e-12:
        return 0.0
    return float(np.corrcoef(predicted_ranks, target_ranks)[0, 1])


def ndcg_at_k(
    predictions: NDArray[np.float64],
    targets: NDArray[np.float64],
    *,
    k: int,
) -> float:
    """Compute NDCG from non-negative within-state shifted realized effects."""

    if predictions.shape != targets.shape or predictions.ndim != 1 or k <= 0:
        raise ValueError("H016 NDCG inputs are invalid")
    limit = min(k, predictions.size)
    relevance = targets - targets.min()
    discounts = np.log2(np.arange(2, limit + 2, dtype=np.float64))
    selected = np.argsort(predictions)[::-1][:limit]
    ideal = np.argsort(relevance)[::-1][:limit]
    dcg = float(np.sum(relevance[selected] / discounts))
    ideal_dcg = float(np.sum(relevance[ideal] / discounts))
    return 1.0 if ideal_dcg <= 1e-12 else dcg / ideal_dcg
