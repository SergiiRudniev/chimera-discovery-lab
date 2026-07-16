"""Deterministic CEM plus quality-diversity archive for H015."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import asdict, dataclass

import numpy as np
from numpy.typing import NDArray

from chimera.meta_world.h015.config import H015SearchConfig


@dataclass(frozen=True)
class InterventionCandidate:
    """One legal language-free intervention vector."""

    source_slot: int
    target_slot: int
    magnitude: float
    control: float

    def __post_init__(self) -> None:
        if self.source_slot < 0 or self.target_slot < 0:
            raise ValueError("candidate slots must be non-negative")
        if self.source_slot == self.target_slot:
            raise ValueError("candidate source and target must differ")
        if not 0.0 <= self.magnitude <= 1.0 or not -1.0 <= self.control <= 1.0:
            raise ValueError("candidate continuous values are outside legal bounds")

    def to_dict(self) -> dict[str, int | float]:
        return asdict(self)


@dataclass(frozen=True)
class ScoredCandidate:
    """Candidate plus model-only prediction and archive score."""

    candidate: InterventionCandidate
    predicted_effect_mean: float
    predicted_effect_std: float
    score: float
    archive_cell: tuple[int, int, int]

    def to_dict(self) -> dict[str, object]:
        return {
            **self.candidate.to_dict(),
            "predicted_effect_mean": self.predicted_effect_mean,
            "predicted_effect_std": self.predicted_effect_std,
            "score": self.score,
            "archive_cell": list(self.archive_cell),
        }


@dataclass(frozen=True)
class SearchResult:
    """Frozen selected archive and exact model-score accounting."""

    selected: tuple[ScoredCandidate, ...]
    model_scores: int
    archive_cells: int
    unique_source_target_pairs: int

    def to_dict(self) -> dict[str, object]:
        return {
            "selected": [item.to_dict() for item in self.selected],
            "model_scores": self.model_scores,
            "archive_cells": self.archive_cells,
            "unique_source_target_pairs": self.unique_source_target_pairs,
        }


PredictionFunction = Callable[
    [tuple[InterventionCandidate, ...]],
    tuple[NDArray[np.float64], NDArray[np.float64]],
]


def _archive_cell(candidate: InterventionCandidate) -> tuple[int, int, int]:
    quartile = min(int(candidate.magnitude * 4.0), 3)
    return candidate.source_slot, candidate.target_slot, quartile


def _sample_candidates(
    rng: np.random.Generator,
    *,
    objects: int,
    count: int,
    source_probabilities: NDArray[np.float64],
    target_probabilities: NDArray[np.float64],
    magnitude_mean: float,
    magnitude_std: float,
    control_mean: float,
    control_std: float,
) -> tuple[InterventionCandidate, ...]:
    sources = rng.choice(objects, size=count, p=source_probabilities)
    targets = rng.choice(objects, size=count, p=target_probabilities)
    targets = np.where(targets == sources, (targets + 1) % objects, targets)
    magnitudes = np.clip(rng.normal(magnitude_mean, magnitude_std, count), 0.0, 1.0)
    controls = np.clip(rng.normal(control_mean, control_std, count), -1.0, 1.0)
    return tuple(
        InterventionCandidate(
            source_slot=int(sources[index]),
            target_slot=int(targets[index]),
            magnitude=float(magnitudes[index]),
            control=float(controls[index]),
        )
        for index in range(count)
    )


def quality_diversity_search(
    *,
    objects: int,
    seed: int,
    config: H015SearchConfig,
    uncertainty_beta: float,
    predict: PredictionFunction,
) -> SearchResult:
    """Run registered CEM rounds and return eight diverse legal candidates."""

    if objects <= 1 or seed < 0 or uncertainty_beta < 0.0:
        raise ValueError("invalid H015 search inputs")
    rng = np.random.default_rng(seed)
    source_probabilities = np.full(objects, 1.0 / objects, dtype=np.float64)
    target_probabilities = source_probabilities.copy()
    magnitude_mean, magnitude_std = 0.5, 0.30
    control_mean, control_std = 0.0, 0.60
    archive: dict[tuple[int, int, int], ScoredCandidate] = {}
    model_scores = 0
    for _ in range(config.rounds):
        candidates = _sample_candidates(
            rng,
            objects=objects,
            count=config.candidates_per_round,
            source_probabilities=source_probabilities,
            target_probabilities=target_probabilities,
            magnitude_mean=magnitude_mean,
            magnitude_std=magnitude_std,
            control_mean=control_mean,
            control_std=control_std,
        )
        means, standard_deviations = predict(candidates)
        if means.shape != (len(candidates),) or standard_deviations.shape != means.shape:
            raise ValueError("H015 predictor returned an invalid shape")
        if (
            not np.isfinite(means).all()
            or not np.isfinite(standard_deviations).all()
            or np.any(standard_deviations < 0.0)
        ):
            raise FloatingPointError("H015 predictor returned invalid values")
        scores = means - uncertainty_beta * standard_deviations
        model_scores += len(candidates)
        scored = tuple(
            ScoredCandidate(
                candidate=candidate,
                predicted_effect_mean=float(means[index]),
                predicted_effect_std=float(standard_deviations[index]),
                score=float(scores[index]),
                archive_cell=_archive_cell(candidate),
            )
            for index, candidate in enumerate(candidates)
        )
        for item in scored:
            retained = archive.get(item.archive_cell)
            if retained is None or item.score > retained.score:
                archive[item.archive_cell] = item
        elite_indices = np.argsort(scores)[-config.elite_candidates_per_round :]
        elites = [candidates[int(index)] for index in elite_indices]
        source_counts = np.full(objects, 0.5, dtype=np.float64)
        target_counts = np.full(objects, 0.5, dtype=np.float64)
        for elite in elites:
            source_counts[elite.source_slot] += 1.0
            target_counts[elite.target_slot] += 1.0
        source_probabilities = source_counts / source_counts.sum()
        target_probabilities = target_counts / target_counts.sum()
        elite_magnitudes = np.asarray([item.magnitude for item in elites])
        elite_controls = np.asarray([item.control for item in elites])
        magnitude_mean = float(elite_magnitudes.mean())
        magnitude_std = max(float(elite_magnitudes.std()), 0.05)
        control_mean = float(elite_controls.mean())
        control_std = max(float(elite_controls.std()), 0.05)
    selected = tuple(
        sorted(archive.values(), key=lambda item: item.score, reverse=True)[
            : config.simulator_executions_per_state
        ]
    )
    if len(selected) != config.simulator_executions_per_state:
        raise RuntimeError("H015 archive did not fill the execution budget")
    return SearchResult(
        selected=selected,
        model_scores=model_scores,
        archive_cells=len(archive),
        unique_source_target_pairs=len(
            {
                (item.candidate.source_slot, item.candidate.target_slot)
                for item in selected
            }
        ),
    )
