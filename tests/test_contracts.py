from __future__ import annotations

import pytest
import torch

from chimera.config import ModelConfig
from chimera.data.contracts import GraphBatch
from chimera.data.synthetic import make_synthetic_batch


def test_synthetic_batch_satisfies_contract(small_model_config: ModelConfig) -> None:
    batch = make_synthetic_batch(small_model_config, batch_size=3, seed=13)
    assert batch.graph.node_types.shape == (3, 8)
    assert batch.edits.operations.shape == (3, 3)
    assert batch.scores.shape == (3, 3)
    assert torch.all((batch.scores >= 0) & (batch.scores <= 1))


def test_padded_node_type_must_be_zero() -> None:
    graph = GraphBatch(
        node_types=torch.tensor([[1, 2]]),
        node_features=torch.zeros(1, 2, 3),
        edge_types=torch.zeros(1, 2, 2, dtype=torch.long),
        node_mask=torch.tensor([[True, False]]),
    )
    with pytest.raises(ValueError, match="padded"):
        graph.validate(feature_dim=3)
