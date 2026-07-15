from __future__ import annotations

import numpy as np
import pytest
import torch

from chimera.config import ModelConfig
from chimera.data.synthetic import make_synthetic_batch
from chimera.generation.archive import ArchiveEntry, MapElitesArchive
from chimera.generation.mutate import apply_edit_program, validate_edit_program
from chimera.generation.sampler import _sample_masked, sample_edit_program
from chimera.models.venture import ChimeraVenture


def test_edit_program_changes_graph(small_model_config: ModelConfig) -> None:
    batch = make_synthetic_batch(small_model_config, batch_size=2, seed=23)
    changed = apply_edit_program(batch.graph, batch.edits)
    assert not torch.equal(changed.edge_types, batch.graph.edge_types) or not torch.equal(
        changed.node_types, batch.graph.node_types
    )


def test_sampler_returns_bounded_program(small_model_config: ModelConfig) -> None:
    batch = make_synthetic_batch(small_model_config, batch_size=2, seed=29)
    model = ChimeraVenture(small_model_config)
    program = sample_edit_program(model, batch.graph, temperature=0.0)
    program.validate(batch_size=2, max_nodes=8)
    assert program.steps == 3
    assert validate_edit_program(batch.graph, program) == ((), ())


def test_full_exploration_ignores_model_probabilities() -> None:
    logits = torch.full((4,), torch.nan)
    mask = torch.tensor([False, True, True, False])
    generator = torch.Generator().manual_seed(31)
    samples = {
        _sample_masked(logits, mask, 0.75, 1.0, generator)
        for _ in range(64)
    }
    assert samples == {1, 2}


def test_sampler_rejects_invalid_exploration_rate(small_model_config: ModelConfig) -> None:
    batch = make_synthetic_batch(small_model_config, batch_size=1, seed=37)
    model = ChimeraVenture(small_model_config)
    with pytest.raises(ValueError, match="exploration_rate"):
        sample_edit_program(model, batch.graph, exploration_rate=1.01)


def test_archive_replaces_only_with_higher_quality() -> None:
    archive = MapElitesArchive(bins=(2, 2), bounds=((0.0, 1.0), (0.0, 1.0)))
    first = ArchiveEntry("a", (0.2, 0.2), 0.5, np.array([1.0, 0.0], dtype=np.float32))
    worse = ArchiveEntry("b", (0.2, 0.2), 0.4, np.array([0.0, 1.0], dtype=np.float32))
    better = ArchiveEntry("c", (0.2, 0.2), 0.8, np.array([0.5, 0.5], dtype=np.float32))
    assert archive.add(first)
    assert not archive.add(worse)
    assert archive.add(better)
    assert len(archive) == 1
    assert archive.coverage == 0.25
    assert archive.novelty(np.array([1.0, 0.0], dtype=np.float32)) >= 0.0
