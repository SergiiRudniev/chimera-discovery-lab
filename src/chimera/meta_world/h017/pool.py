"""Balanced seeded Latin-hypercube candidate generation for H017."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from chimera.meta_world.h015.search import InterventionCandidate


@dataclass(frozen=True)
class SupportPoolDiagnostics:
    """Exact pool invariants used by the development gate."""

    candidates: int
    legal_action_rate: float
    exact_continuous_boundary_rate: float
    unique_vector_rate: float
    pair_count_discrepancy: int
    magnitude_mean: float
    absolute_control_mean: float

    def to_dict(self) -> dict[str, int | float]:
        return {
            "candidates": self.candidates,
            "legal_action_rate": self.legal_action_rate,
            "exact_continuous_boundary_rate": self.exact_continuous_boundary_rate,
            "unique_vector_rate": self.unique_vector_rate,
            "pair_count_discrepancy": self.pair_count_discrepancy,
            "magnitude_mean": self.magnitude_mean,
            "absolute_control_mean": self.absolute_control_mean,
        }


def balanced_support_pool(
    *,
    objects: int,
    count: int,
    seed: int,
) -> tuple[InterventionCandidate, ...]:
    """Generate legal interior candidates with balanced ordered slot pairs."""

    if objects <= 1 or count <= 0 or seed < 0:
        raise ValueError("invalid H017 support-pool request")
    rng = np.random.default_rng(seed)
    ordered_pairs = np.asarray(
        [
            (source, target)
            for source in range(objects)
            for target in range(objects)
            if source != target
        ],
        dtype=np.int64,
    )
    pair_rows: list[NDArray[np.int64]] = []
    remaining = count
    while remaining:
        permutation = rng.permutation(len(ordered_pairs))
        take = min(remaining, len(ordered_pairs))
        pair_rows.append(ordered_pairs[permutation[:take]])
        remaining -= take
    pairs = np.concatenate(pair_rows, axis=0)
    magnitude_strata = rng.permutation(count)
    control_strata = rng.permutation(count)
    lower = np.nextafter(0.0, 1.0)
    upper = np.nextafter(1.0, 0.0)
    magnitude_jitter = rng.uniform(lower, upper, size=count)
    control_jitter = rng.uniform(lower, upper, size=count)
    magnitudes = (magnitude_strata + magnitude_jitter) / count
    controls = -1.0 + 2.0 * (control_strata + control_jitter) / count
    return tuple(
        InterventionCandidate(
            source_slot=int(pairs[index, 0]),
            target_slot=int(pairs[index, 1]),
            magnitude=float(magnitudes[index]),
            control=float(controls[index]),
        )
        for index in range(count)
    )


def support_pool_diagnostics(
    candidates: tuple[InterventionCandidate, ...],
) -> SupportPoolDiagnostics:
    """Measure every preregistered support invariant from concrete candidates."""

    if not candidates:
        raise ValueError("H017 diagnostics require candidates")
    legal = sum(
        item.source_slot != item.target_slot
        and 0.0 < item.magnitude < 1.0
        and -1.0 < item.control < 1.0
        for item in candidates
    )
    boundary = sum(
        item.magnitude in {0.0, 1.0} or item.control in {-1.0, 1.0}
        for item in candidates
    )
    unique = len(
        {
            (item.source_slot, item.target_slot, item.magnitude, item.control)
            for item in candidates
        }
    )
    pair_counts = Counter(
        (item.source_slot, item.target_slot) for item in candidates
    ).values()
    magnitudes = np.asarray([item.magnitude for item in candidates])
    controls = np.asarray([item.control for item in candidates])
    return SupportPoolDiagnostics(
        candidates=len(candidates),
        legal_action_rate=legal / len(candidates),
        exact_continuous_boundary_rate=boundary / len(candidates),
        unique_vector_rate=unique / len(candidates),
        pair_count_discrepancy=max(pair_counts) - min(pair_counts),
        magnitude_mean=float(magnitudes.mean()),
        absolute_control_mean=float(np.abs(controls).mean()),
    )
