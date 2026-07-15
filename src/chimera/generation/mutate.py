"""Deterministic execution of discrete edit programs on padded graphs."""

from __future__ import annotations

import torch

from chimera.data.contracts import EditBatch, GraphBatch
from chimera.data.semantics import with_value_proximity
from chimera.schema import EditOperation, NodeType


def _remove_node(graph: GraphBatch, batch_index: int, node_index: int) -> None:
    graph.node_mask[batch_index, node_index] = False
    graph.node_types[batch_index, node_index] = int(NodeType.PAD)
    graph.node_features[batch_index, node_index].zero_()
    graph.edge_types[batch_index, node_index, :].zero_()
    graph.edge_types[batch_index, :, node_index].zero_()


@torch.no_grad()
def apply_edit_program(graph: GraphBatch, edits: EditBatch) -> GraphBatch:
    """Apply bounded edits without introducing language or external state."""

    graph.validate()
    edits.validate(batch_size=graph.batch_size, max_nodes=graph.max_nodes)
    result = graph.clone()
    for batch_index in range(graph.batch_size):
        for step in range(edits.steps):
            if not bool(edits.step_mask[batch_index, step]):
                continue
            operation = EditOperation(int(edits.operations[batch_index, step]))
            if operation is EditOperation.STOP:
                break
            source = int(edits.source_nodes[batch_index, step])
            target = int(edits.target_nodes[batch_index, step])
            node_type = int(edits.node_types[batch_index, step])
            edge_type = int(edits.edge_types[batch_index, step])

            if operation is EditOperation.ADD_NODE:
                free = torch.where(~result.node_mask[batch_index])[0]
                if free.numel():
                    new_node = int(free[0])
                    result.node_mask[batch_index, new_node] = True
                    result.node_types[batch_index, new_node] = max(node_type, 1)
                    if result.node_mask[batch_index, source] and edge_type > 0:
                        result.edge_types[batch_index, source, new_node] = edge_type
            elif operation in {EditOperation.CONNECT, EditOperation.REWIRE}:
                if result.node_mask[batch_index, source] and result.node_mask[batch_index, target]:
                    if operation is EditOperation.REWIRE:
                        result.edge_types[batch_index, source].zero_()
                    result.edge_types[batch_index, source, target] = max(edge_type, 1)
            elif operation is EditOperation.TRANSFER_ROLE:
                if result.node_mask[batch_index, target]:
                    result.node_types[batch_index, target] = max(node_type, 1)
            elif operation is EditOperation.REMOVE_CONSTRAINT:
                if result.node_types[batch_index, source] == int(NodeType.CONSTRAINT):
                    _remove_node(result, batch_index, source)
            elif operation is EditOperation.INVERT_RELATION:
                relation = result.edge_types[batch_index, source, target].clone()
                result.edge_types[batch_index, source, target] = 0
                result.edge_types[batch_index, target, source] = torch.where(
                    relation > 0,
                    relation,
                    torch.as_tensor(max(edge_type, 1), device=relation.device),
                )
            elif operation is EditOperation.SUBSTITUTE:
                if result.node_mask[batch_index, source]:
                    result.node_types[batch_index, source] = max(node_type, 1)
            elif operation is EditOperation.MERGE:
                if source != target and result.node_mask[batch_index, target]:
                    outgoing = result.edge_types[batch_index, target]
                    incoming = result.edge_types[batch_index, :, target]
                    result.edge_types[batch_index, source] = torch.maximum(
                        result.edge_types[batch_index, source], outgoing
                    )
                    result.edge_types[batch_index, :, source] = torch.maximum(
                        result.edge_types[batch_index, :, source], incoming
                    )
                    _remove_node(result, batch_index, target)
    return with_value_proximity(result)


@torch.no_grad()
def validate_edit_program(graph: GraphBatch, edits: EditBatch) -> tuple[tuple[str, ...], ...]:
    """Return semantic precondition failures while simulating each program."""

    graph.validate()
    edits.validate(batch_size=graph.batch_size, max_nodes=graph.max_nodes)
    failures: list[tuple[str, ...]] = []
    for batch_index in range(graph.batch_size):
        current = GraphBatch(
            node_types=graph.node_types[batch_index : batch_index + 1].clone(),
            node_features=graph.node_features[batch_index : batch_index + 1].clone(),
            edge_types=graph.edge_types[batch_index : batch_index + 1].clone(),
            node_mask=graph.node_mask[batch_index : batch_index + 1].clone(),
        )
        errors: list[str] = []
        for step in range(edits.steps):
            if not bool(edits.step_mask[batch_index, step]):
                continue
            operation = EditOperation(int(edits.operations[batch_index, step]))
            if operation is EditOperation.STOP:
                break
            source = int(edits.source_nodes[batch_index, step])
            target = int(edits.target_nodes[batch_index, step])
            node_type = int(edits.node_types[batch_index, step])
            edge_type = int(edits.edge_types[batch_index, step])
            source_active = bool(current.node_mask[0, source])
            target_active = bool(current.node_mask[0, target])
            reason: str | None = None
            if operation is EditOperation.ADD_NODE:
                if not source_active or node_type == 0 or edge_type == 0:
                    reason = "ADD_NODE requires active source and non-zero types"
                elif not bool((~current.node_mask[0]).any()):
                    reason = "ADD_NODE exceeds graph capacity"
            elif operation in {EditOperation.CONNECT, EditOperation.REWIRE}:
                if not source_active or not target_active or source == target or edge_type == 0:
                    reason = f"{operation.name} requires distinct active nodes and relation"
            elif operation is EditOperation.TRANSFER_ROLE:
                if not target_active or node_type == 0:
                    reason = "TRANSFER_ROLE requires active target and node type"
            elif operation is EditOperation.REMOVE_CONSTRAINT:
                if not source_active or int(current.node_types[0, source]) != int(
                    NodeType.CONSTRAINT
                ):
                    reason = "REMOVE_CONSTRAINT source is not an active constraint"
            elif operation is EditOperation.INVERT_RELATION:
                if (
                    not source_active
                    or not target_active
                    or int(current.edge_types[0, source, target]) == 0
                ):
                    reason = "INVERT_RELATION requires an existing directed relation"
            elif operation is EditOperation.SUBSTITUTE:
                if not source_active or node_type == 0:
                    reason = "SUBSTITUTE requires active source and node type"
            elif operation is EditOperation.MERGE and (
                not source_active or not target_active or source == target
            ):
                reason = "MERGE requires distinct active nodes"
            if reason is not None:
                errors.append(f"step {step}: {reason}")
                continue
            one_step = EditBatch(
                operations=edits.operations[batch_index : batch_index + 1, step : step + 1],
                source_nodes=edits.source_nodes[batch_index : batch_index + 1, step : step + 1],
                target_nodes=edits.target_nodes[batch_index : batch_index + 1, step : step + 1],
                node_types=edits.node_types[batch_index : batch_index + 1, step : step + 1],
                edge_types=edits.edge_types[batch_index : batch_index + 1, step : step + 1],
                step_mask=edits.step_mask[batch_index : batch_index + 1, step : step + 1],
            )
            current = apply_edit_program(current, one_step)
        failures.append(tuple(errors))
    return tuple(failures)
