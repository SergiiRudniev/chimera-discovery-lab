"""Small deterministic MAP-Elites archive for diverse candidate retention."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float32]


@dataclass(frozen=True)
class ArchiveEntry:
    candidate_id: str
    descriptors: tuple[float, ...]
    quality: float
    latent: FloatArray


class MapElitesArchive:
    """Keep the best candidate in each behavior-descriptor cell."""

    def __init__(
        self,
        bins: tuple[int, ...],
        bounds: tuple[tuple[float, float], ...],
    ) -> None:
        if not bins or len(bins) != len(bounds):
            raise ValueError("bins and bounds must be non-empty and have equal dimensionality")
        if any(value <= 0 for value in bins):
            raise ValueError("every archive dimension needs at least one bin")
        if any(low >= high for low, high in bounds):
            raise ValueError("archive bounds must be strictly increasing")
        self.bins = bins
        self.bounds = bounds
        self._cells: dict[tuple[int, ...], ArchiveEntry] = {}

    def cell_for(self, descriptors: Iterable[float]) -> tuple[int, ...]:
        values = tuple(float(value) for value in descriptors)
        if len(values) != len(self.bins):
            raise ValueError("descriptor dimensionality does not match archive")
        cell: list[int] = []
        for value, count, (low, high) in zip(values, self.bins, self.bounds, strict=True):
            normalized = (min(max(value, low), high) - low) / (high - low)
            cell.append(min(int(normalized * count), count - 1))
        return tuple(cell)

    def add(self, entry: ArchiveEntry) -> bool:
        cell = self.cell_for(entry.descriptors)
        incumbent = self._cells.get(cell)
        if incumbent is not None and incumbent.quality >= entry.quality:
            return False
        self._cells[cell] = entry
        return True

    def novelty(self, latent: FloatArray, *, neighbors: int = 5) -> float:
        if neighbors <= 0:
            raise ValueError("neighbors must be positive")
        if not self._cells:
            return 1.0
        vector = np.asarray(latent, dtype=np.float32).reshape(-1)
        norm = float(np.linalg.norm(vector))
        if norm == 0:
            return 0.0
        distances: list[float] = []
        for entry in self._cells.values():
            other = entry.latent.reshape(-1)
            denominator = norm * float(np.linalg.norm(other))
            similarity = 0.0 if denominator == 0 else float(np.dot(vector, other) / denominator)
            distances.append(1.0 - max(-1.0, min(1.0, similarity)))
        distances.sort()
        return float(np.mean(distances[: min(neighbors, len(distances))]))

    @property
    def coverage(self) -> float:
        total = int(np.prod(self.bins))
        return len(self._cells) / total

    def __len__(self) -> int:
        return len(self._cells)
