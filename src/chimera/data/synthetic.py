"""Deterministic synthetic transitions used only for engineering smoke tests."""

from __future__ import annotations

import torch

from chimera.config import ModelConfig
from chimera.data.contracts import EditBatch, GraphBatch, TrainingBatch
from chimera.generation.mutate import apply_edit_program
from chimera.schema import EdgeType, EditOperation, NodeType


def make_synthetic_batch(
    config: ModelConfig,
    *,
    batch_size: int,
    seed: int,
    device: torch.device | str = "cpu",
) -> TrainingBatch:
    """Create a reproducible batch; it is not evidence of real-world creativity."""

    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    generator = torch.Generator(device="cpu").manual_seed(seed)
    batch, nodes, features = batch_size, config.max_nodes, config.node_numeric_features
    node_types = torch.zeros(batch, nodes, dtype=torch.long)
    node_features = torch.zeros(batch, nodes, features)
    edge_types = torch.zeros(batch, nodes, nodes, dtype=torch.long)
    node_mask = torch.zeros(batch, nodes, dtype=torch.bool)

    operations = torch.zeros(batch, config.max_edits, dtype=torch.long)
    sources = torch.zeros_like(operations)
    targets = torch.zeros_like(operations)
    edit_node_types = torch.zeros_like(operations)
    edit_edge_types = torch.zeros_like(operations)
    step_mask = torch.zeros(batch, config.max_edits, dtype=torch.bool)

    minimum_nodes = min(6, nodes)
    for index in range(batch):
        active_nodes = int(
            torch.randint(
                minimum_nodes,
                max(minimum_nodes + 1, nodes - 1),
                (1,),
                generator=generator,
            )
        )
        node_mask[index, :active_nodes] = True
        node_types[index, :active_nodes] = torch.randint(
            int(NodeType.ACTOR),
            int(NodeType.FEEDBACK) + 1,
            (active_nodes,),
            generator=generator,
        )
        node_features[index, :active_nodes] = torch.randn(
            active_nodes, features, generator=generator
        )
        for _ in range(active_nodes * 2):
            source = int(torch.randint(0, active_nodes, (1,), generator=generator))
            target = int(torch.randint(0, active_nodes, (1,), generator=generator))
            if source != target:
                edge_types[index, source, target] = int(
                    torch.randint(
                        int(EdgeType.HAS_NEED),
                        int(EdgeType.REDUCES) + 1,
                        (1,),
                        generator=generator,
                    )
                )

        edit_count = int(
            torch.randint(1, max(2, config.max_edits), (1,), generator=generator)
        )
        step_mask[index, :edit_count] = True
        for step in range(edit_count):
            operation_options = torch.tensor(
                [
                    int(EditOperation.CONNECT),
                    int(EditOperation.REWIRE),
                    int(EditOperation.TRANSFER_ROLE),
                    int(EditOperation.INVERT_RELATION),
                    int(EditOperation.SUBSTITUTE),
                ]
            )
            choice = int(torch.randint(0, len(operation_options), (1,), generator=generator))
            operations[index, step] = operation_options[choice]
            sources[index, step] = torch.randint(0, active_nodes, (1,), generator=generator)
            targets[index, step] = torch.randint(0, active_nodes, (1,), generator=generator)
            edit_node_types[index, step] = torch.randint(
                int(NodeType.ACTOR),
                int(NodeType.FEEDBACK) + 1,
                (1,),
                generator=generator,
            )
            edit_edge_types[index, step] = torch.randint(
                int(EdgeType.HAS_NEED),
                int(EdgeType.REDUCES) + 1,
                (1,),
                generator=generator,
            )

    graph = GraphBatch(node_types, node_features, edge_types, node_mask)
    edits = EditBatch(
        operations, sources, targets, edit_node_types, edit_edge_types, step_mask
    )
    next_graph = apply_edit_program(graph, edits)
    density = (edge_types > 0).float().sum(dim=(1, 2)) / node_mask.sum(dim=1).square().clamp_min(1)
    complexity = step_mask.float().mean(dim=1)
    signal = node_features.abs().mean(dim=(1, 2))
    scores = torch.stack(
        (
            torch.sigmoid(signal),
            (1.0 - 0.5 * complexity).clamp(0, 1),
            torch.exp(-torch.abs(density - 0.2)),
        ),
        dim=1,
    )[:, : config.score_dimensions]
    result = TrainingBatch(graph, edits, next_graph, scores).validate(
        feature_dim=config.node_numeric_features,
        score_dimensions=config.score_dimensions,
    )
    return result.to(device)
