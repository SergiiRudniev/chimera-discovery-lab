from __future__ import annotations

import torch

from chimera.data.contracts import GraphBatch
from chimera.data.semantics import compute_proxy_scores, with_value_proximity
from chimera.schema import EdgeType, NodeType


def test_value_proximity_is_derived_from_topology() -> None:
    graph = GraphBatch(
        node_types=torch.tensor(
            [[int(NodeType.ACTION), int(NodeType.VALUE), int(NodeType.PAD)]]
        ),
        node_features=torch.zeros((1, 3, 8)),
        edge_types=torch.tensor(
            [
                [
                    [0, int(EdgeType.DELIVERS), 0],
                    [0, 0, 0],
                    [0, 0, 0],
                ]
            ]
        ),
        node_mask=torch.tensor([[True, True, False]]),
    )
    derived = with_value_proximity(graph)
    torch.testing.assert_close(
        derived.node_features[0, :, 6], torch.tensor([0.75, 1.0, 0.0])
    )


def test_proxy_scores_are_bounded() -> None:
    graph = GraphBatch(
        node_types=torch.tensor(
            [[int(NodeType.ACTION), int(NodeType.VALUE), int(NodeType.OUTCOME)]]
        ),
        node_features=torch.full((1, 3, 8), 0.5),
        edge_types=torch.tensor(
            [
                [
                    [0, int(EdgeType.DELIVERS), 0],
                    [0, 0, int(EdgeType.PRODUCES)],
                    [0, 0, 0],
                ]
            ]
        ),
        node_mask=torch.ones((1, 3), dtype=torch.bool),
    )
    scores = compute_proxy_scores(graph)
    assert scores.shape == (1, 3)
    assert torch.all((scores >= 0) & (scores <= 1))
