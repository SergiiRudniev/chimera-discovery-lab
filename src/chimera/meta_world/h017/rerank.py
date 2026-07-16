"""One-pass quality-diversity reranking over a finite H017 pool."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from chimera.meta_world.h015.search import (
    InterventionCandidate,
    ScoredCandidate,
    SearchResult,
)


def _archive_cell(candidate: InterventionCandidate) -> tuple[int, int, int]:
    return (
        candidate.source_slot,
        candidate.target_slot,
        min(int(candidate.magnitude * 4.0), 3),
    )


def one_pass_qd_rerank(
    candidates: tuple[InterventionCandidate, ...],
    rank_logits: NDArray[np.float64],
    *,
    executions: int,
) -> SearchResult:
    """Retain one candidate per cell, then execute the best distinct cells."""

    if (
        not candidates
        or rank_logits.shape != (len(candidates),)
        or executions <= 0
        or executions > len(candidates)
    ):
        raise ValueError("invalid H017 reranking request")
    if not np.isfinite(rank_logits).all():
        raise FloatingPointError("H017 rank logits are non-finite")
    archive: dict[tuple[int, int, int], ScoredCandidate] = {}
    for candidate, logit in zip(candidates, rank_logits, strict=True):
        cell = _archive_cell(candidate)
        scored = ScoredCandidate(
            candidate=candidate,
            predicted_effect_mean=float(logit),
            predicted_effect_std=0.0,
            score=float(logit),
            archive_cell=cell,
        )
        retained = archive.get(cell)
        if retained is None or scored.score > retained.score:
            archive[cell] = scored
    selected = tuple(
        sorted(archive.values(), key=lambda item: item.score, reverse=True)[:executions]
    )
    if len(selected) != executions:
        raise RuntimeError("H017 support archive did not fill the execution budget")
    return SearchResult(
        selected=selected,
        model_scores=len(candidates),
        archive_cells=len(archive),
        unique_source_target_pairs=len(
            {
                (item.candidate.source_slot, item.candidate.target_slot)
                for item in selected
            }
        ),
    )
