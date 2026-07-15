from __future__ import annotations

import torch

from chimera.config import ModelConfig
from chimera.data.synthetic import make_synthetic_batch
from chimera.models.venture import ChimeraVenture


def test_model_output_shapes(small_model_config: ModelConfig) -> None:
    batch = make_synthetic_batch(small_model_config, batch_size=2, seed=11)
    model = ChimeraVenture(small_model_config).eval()
    with torch.no_grad():
        output = model(batch.graph, batch.edits)
    assert output.operation_logits.shape == (2, 3, 9)
    assert output.source_logits.shape == (2, 3, 8)
    assert output.target_logits.shape == (2, 3, 8)
    assert output.node_type_logits.shape == (2, 3, 12)
    assert output.edge_type_logits.shape == (2, 3, 16)
    assert output.score_logits.shape == (2, 3)
    assert output.predicted_next_state.shape == (2, 32)
    assert torch.isfinite(output.operation_logits).all()


def test_model_is_permutation_equivariant_in_eval(small_model_config: ModelConfig) -> None:
    batch = make_synthetic_batch(small_model_config, batch_size=1, seed=19)
    model = ChimeraVenture(small_model_config).eval()
    permutation = torch.tensor([1, 0, 2, 3, 4, 5, 6, 7])
    graph = batch.graph
    permuted = type(graph)(
        node_types=graph.node_types[:, permutation],
        node_features=graph.node_features[:, permutation],
        edge_types=graph.edge_types[:, permutation][:, :, permutation],
        node_mask=graph.node_mask[:, permutation],
    )
    with torch.no_grad():
        original = model.encoder(graph).graph_state
        changed = model.encoder(permuted).graph_state
    torch.testing.assert_close(original, changed, atol=1e-5, rtol=1e-5)


def test_trainable_parameter_count_is_positive(small_model_config: ModelConfig) -> None:
    model = ChimeraVenture(small_model_config)
    assert model.trainable_parameter_count() == sum(
        parameter.numel() for parameter in model.parameters()
    )
