"""Autoregressive sampling of structured edit programs."""

from __future__ import annotations

import torch
from torch import Tensor

from chimera.data.contracts import EditBatch, GraphBatch
from chimera.models.venture import ChimeraVenture
from chimera.schema import EditOperation


def _sample(logits: Tensor, temperature: float, generator: torch.Generator | None) -> Tensor:
    if temperature <= 0:
        return logits.argmax(dim=-1)
    probabilities = torch.softmax(logits / temperature, dim=-1)
    return torch.multinomial(probabilities, 1, generator=generator).squeeze(-1)


@torch.no_grad()
def sample_edit_program(
    model: ChimeraVenture,
    graph: GraphBatch,
    *,
    max_edits: int | None = None,
    temperature: float = 0.9,
    generator: torch.Generator | None = None,
) -> EditBatch:
    """Sample edit symbols; no textual prompt or decoder participates."""

    model.eval()
    steps = max_edits or model.config.max_edits
    if not 1 <= steps <= model.config.max_edits:
        raise ValueError("max_edits must be within the configured capacity")
    batch = graph.batch_size
    device = graph.node_types.device
    shape = (batch, steps)
    operations = torch.zeros(shape, dtype=torch.long, device=device)
    sources = torch.zeros_like(operations)
    targets = torch.zeros_like(operations)
    node_types = torch.zeros_like(operations)
    edge_types = torch.zeros_like(operations)
    step_mask = torch.zeros(shape, dtype=torch.bool, device=device)
    active = torch.ones(batch, dtype=torch.bool, device=device)

    for step in range(steps):
        step_mask[:, step] = active
        prefix = EditBatch(
            operations=operations[:, : step + 1],
            source_nodes=sources[:, : step + 1],
            target_nodes=targets[:, : step + 1],
            node_types=node_types[:, : step + 1],
            edge_types=edge_types[:, : step + 1],
            step_mask=step_mask[:, : step + 1],
        )
        output = model(graph, prefix)
        operations[:, step] = _sample(output.operation_logits[:, -1], temperature, generator)
        sources[:, step] = _sample(output.source_logits[:, -1], temperature, generator)
        targets[:, step] = _sample(output.target_logits[:, -1], temperature, generator)
        node_types[:, step] = _sample(output.node_type_logits[:, -1], temperature, generator)
        edge_types[:, step] = _sample(output.edge_type_logits[:, -1], temperature, generator)
        active = active & (operations[:, step] != int(EditOperation.STOP))
        if not bool(active.any()):
            break

    return EditBatch(
        operations=operations,
        source_nodes=sources,
        target_nodes=targets,
        node_types=node_types,
        edge_types=edge_types,
        step_mask=step_mask,
    )
