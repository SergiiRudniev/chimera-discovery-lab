"""Validity-constrained autoregressive sampling of structured edit programs."""

from __future__ import annotations

import torch
from torch import Tensor

from chimera.data.contracts import EditBatch, GraphBatch
from chimera.generation.mutate import apply_edit_program
from chimera.models.venture import ChimeraVenture
from chimera.schema import EditOperation, NodeType


def _sample_masked(
    logits: Tensor,
    mask: Tensor,
    temperature: float,
    exploration_rate: float,
    generator: torch.Generator | None,
) -> int:
    if not bool(mask.any()):
        raise ValueError("cannot sample from an empty categorical mask")
    masked = logits.masked_fill(~mask, torch.finfo(logits.dtype).min)
    legal_uniform = mask.to(dtype=logits.dtype) / mask.sum()
    if exploration_rate == 1.0:
        return int(torch.multinomial(legal_uniform, 1, generator=generator))
    if temperature <= 0 and exploration_rate == 0:
        return int(masked.argmax())
    if temperature <= 0:
        probabilities = torch.zeros_like(masked)
        probabilities[masked.argmax()] = 1.0
    else:
        probabilities = torch.softmax(masked / temperature, dim=-1)
    if exploration_rate:
        probabilities = (
            (1.0 - exploration_rate) * probabilities + exploration_rate * legal_uniform
        )
    return int(torch.multinomial(probabilities, 1, generator=generator))


def _sample_pair(
    left_logits: Tensor,
    right_logits: Tensor,
    pair_mask: Tensor,
    temperature: float,
    exploration_rate: float,
    generator: torch.Generator | None,
) -> tuple[int, int]:
    scores = left_logits[:, None] + right_logits[None, :]
    flat_index = _sample_masked(
        scores.flatten(),
        pair_mask.flatten(),
        temperature,
        exploration_rate,
        generator,
    )
    return divmod(flat_index, scores.shape[1])


def _allowed_operations(graph: GraphBatch, batch_index: int, *, allow_stop: bool) -> Tensor:
    active = graph.node_mask[batch_index]
    active_count = int(active.sum())
    existing = graph.edge_types[batch_index] > 0
    pair = active[:, None] & active[None, :]
    pair.fill_diagonal_(False)
    allowed = torch.zeros(9, dtype=torch.bool, device=active.device)
    allowed[int(EditOperation.STOP)] = allow_stop
    allowed[int(EditOperation.ADD_NODE)] = bool(active.any() and (~active).any())
    allowed[int(EditOperation.CONNECT)] = bool((pair & ~existing).any())
    allowed[int(EditOperation.REWIRE)] = active_count >= 2
    allowed[int(EditOperation.TRANSFER_ROLE)] = active_count >= 1
    allowed[int(EditOperation.REMOVE_CONSTRAINT)] = bool(
        ((graph.node_types[batch_index] == int(NodeType.CONSTRAINT)) & active).any()
    )
    allowed[int(EditOperation.INVERT_RELATION)] = bool((pair & existing).any())
    allowed[int(EditOperation.SUBSTITUTE)] = active_count >= 1
    allowed[int(EditOperation.MERGE)] = active_count >= 2
    if not bool(allowed.any()):
        allowed[int(EditOperation.STOP)] = True
    return allowed


@torch.no_grad()
def sample_edit_program(
    model: ChimeraVenture,
    graph: GraphBatch,
    *,
    max_edits: int | None = None,
    min_edits: int = 0,
    temperature: float = 0.9,
    exploration_rate: float = 0.0,
    generator: torch.Generator | None = None,
) -> EditBatch:
    """Sample a program while masking operations and arguments that violate graph rules."""

    model.eval()
    steps = max_edits or model.config.max_edits
    if not 1 <= steps <= model.config.max_edits:
        raise ValueError("max_edits must be within the configured capacity")
    if not 0 <= min_edits <= steps:
        raise ValueError("min_edits must be between zero and max_edits")
    if temperature < 0:
        raise ValueError("temperature must be non-negative")
    if not 0.0 <= exploration_rate <= 1.0:
        raise ValueError("exploration_rate must be in [0, 1]")
    graph.validate(feature_dim=model.config.node_numeric_features)
    batch = graph.batch_size
    device = graph.node_types.device
    shape = (batch, steps)
    operations = torch.zeros(shape, dtype=torch.long, device=device)
    sources = torch.zeros_like(operations)
    targets = torch.zeros_like(operations)
    node_types = torch.zeros_like(operations)
    edge_types = torch.zeros_like(operations)
    step_mask = torch.zeros(shape, dtype=torch.bool, device=device)
    active_program = torch.ones(batch, dtype=torch.bool, device=device)
    working_graph = graph.clone()

    for step in range(steps):
        step_mask[:, step] = active_program
        prefix = EditBatch(
            operations=operations[:, : step + 1],
            source_nodes=sources[:, : step + 1],
            target_nodes=targets[:, : step + 1],
            node_types=node_types[:, : step + 1],
            edge_types=edge_types[:, : step + 1],
            step_mask=step_mask[:, : step + 1],
        )
        output = model(graph, prefix)
        for batch_index in torch.where(active_program)[0].tolist():
            allowed = _allowed_operations(working_graph, batch_index, allow_stop=step >= min_edits)
            operation = EditOperation(
                _sample_masked(
                    output.operation_logits[batch_index, -1],
                    allowed,
                    temperature,
                    exploration_rate,
                    generator,
                )
            )
            operations[batch_index, step] = int(operation)
            if operation is EditOperation.STOP:
                continue

            active_nodes = working_graph.node_mask[batch_index]
            non_pad_types = torch.ones(model.config.node_types, dtype=torch.bool, device=device)
            non_pad_types[0] = False
            relation_types = torch.ones(model.config.edge_types, dtype=torch.bool, device=device)
            relation_types[0] = False

            if operation is EditOperation.ADD_NODE:
                sources[batch_index, step] = _sample_masked(
                    output.source_logits[batch_index, -1],
                    active_nodes,
                    temperature,
                    exploration_rate,
                    generator,
                )
                node_types[batch_index, step] = _sample_masked(
                    output.node_type_logits[batch_index, -1],
                    non_pad_types,
                    temperature,
                    exploration_rate,
                    generator,
                )
                edge_types[batch_index, step] = _sample_masked(
                    output.edge_type_logits[batch_index, -1],
                    relation_types,
                    temperature,
                    exploration_rate,
                    generator,
                )
            elif operation in {EditOperation.CONNECT, EditOperation.REWIRE}:
                pair_mask = active_nodes[:, None] & active_nodes[None, :]
                pair_mask.fill_diagonal_(False)
                if operation is EditOperation.CONNECT:
                    pair_mask &= working_graph.edge_types[batch_index] == 0
                source, target = _sample_pair(
                    output.source_logits[batch_index, -1],
                    output.target_logits[batch_index, -1],
                    pair_mask,
                    temperature,
                    exploration_rate,
                    generator,
                )
                sources[batch_index, step] = source
                targets[batch_index, step] = target
                edge_types[batch_index, step] = _sample_masked(
                    output.edge_type_logits[batch_index, -1],
                    relation_types,
                    temperature,
                    exploration_rate,
                    generator,
                )
            elif operation is EditOperation.TRANSFER_ROLE:
                current_types = working_graph.node_types[batch_index]
                pair_mask = active_nodes[:, None] & non_pad_types[None, :]
                pair_mask &= (
                    current_types[:, None]
                    != torch.arange(model.config.node_types, device=device)[None, :]
                )
                target, node_type = _sample_pair(
                    output.target_logits[batch_index, -1],
                    output.node_type_logits[batch_index, -1],
                    pair_mask,
                    temperature,
                    exploration_rate,
                    generator,
                )
                targets[batch_index, step] = target
                node_types[batch_index, step] = node_type
            elif operation is EditOperation.REMOVE_CONSTRAINT:
                constraint_nodes = active_nodes & (
                    working_graph.node_types[batch_index] == int(NodeType.CONSTRAINT)
                )
                sources[batch_index, step] = _sample_masked(
                    output.source_logits[batch_index, -1],
                    constraint_nodes,
                    temperature,
                    exploration_rate,
                    generator,
                )
            elif operation is EditOperation.INVERT_RELATION:
                pair_mask = (
                    active_nodes[:, None]
                    & active_nodes[None, :]
                    & (working_graph.edge_types[batch_index] > 0)
                )
                source, target = _sample_pair(
                    output.source_logits[batch_index, -1],
                    output.target_logits[batch_index, -1],
                    pair_mask,
                    temperature,
                    exploration_rate,
                    generator,
                )
                sources[batch_index, step] = source
                targets[batch_index, step] = target
            elif operation is EditOperation.SUBSTITUTE:
                current_types = working_graph.node_types[batch_index]
                pair_mask = active_nodes[:, None] & non_pad_types[None, :]
                pair_mask &= (
                    current_types[:, None]
                    != torch.arange(model.config.node_types, device=device)[None, :]
                )
                source, node_type = _sample_pair(
                    output.source_logits[batch_index, -1],
                    output.node_type_logits[batch_index, -1],
                    pair_mask,
                    temperature,
                    exploration_rate,
                    generator,
                )
                sources[batch_index, step] = source
                node_types[batch_index, step] = node_type
            elif operation is EditOperation.MERGE:
                pair_mask = active_nodes[:, None] & active_nodes[None, :]
                pair_mask.fill_diagonal_(False)
                source, target = _sample_pair(
                    output.source_logits[batch_index, -1],
                    output.target_logits[batch_index, -1],
                    pair_mask,
                    temperature,
                    exploration_rate,
                    generator,
                )
                sources[batch_index, step] = source
                targets[batch_index, step] = target

        current_step = EditBatch(
            operations=operations[:, step : step + 1],
            source_nodes=sources[:, step : step + 1],
            target_nodes=targets[:, step : step + 1],
            node_types=node_types[:, step : step + 1],
            edge_types=edge_types[:, step : step + 1],
            step_mask=step_mask[:, step : step + 1],
        )
        working_graph = apply_edit_program(working_graph, current_step)
        active_program &= operations[:, step] != int(EditOperation.STOP)
        if not bool(active_program.any()):
            break

    return EditBatch(
        operations=operations,
        source_nodes=sources,
        target_nodes=targets,
        node_types=node_types,
        edge_types=edge_types,
        step_mask=step_mask,
    )
