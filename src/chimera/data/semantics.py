"""Shared numeric semantics for Chimera Venture business graphs."""

from __future__ import annotations

from collections.abc import Iterable

import torch
from torch import Tensor

from chimera.data.contracts import GraphBatch
from chimera.schema import NodeType

FEATURE_NAMES = (
    "salience",
    "evidence",
    "control",
    "immediacy",
    "recurrence",
    "scalability",
    "value_proximity",
    "risk",
)
ANNOTATED_FEATURE_NAMES = FEATURE_NAMES[:6] + FEATURE_NAMES[7:]
FEATURE_LEVELS = frozenset({0.0, 0.25, 0.5, 0.75, 1.0})
VALUE_PROXIMITY_INDEX = 6
RISK_INDEX = 7

_TARGET_TYPES = (int(NodeType.VALUE), int(NodeType.REVENUE), int(NodeType.OUTCOME))
_ENABLER_TYPES = (int(NodeType.RESOURCE), int(NodeType.ACTION), int(NodeType.CHANNEL))


def validate_annotated_features(values: Iterable[float]) -> tuple[float, ...]:
    """Validate the seven human-assigned axes used by Corpus C0."""

    normalized = tuple(float(value) for value in values)
    if len(normalized) != len(ANNOTATED_FEATURE_NAMES):
        raise ValueError(
            f"annotated features must contain {len(ANNOTATED_FEATURE_NAMES)} values"
        )
    invalid = [value for value in normalized if value not in FEATURE_LEVELS]
    if invalid:
        raise ValueError("annotated features must use the registered five-level scale")
    return normalized


def with_value_proximity(graph: GraphBatch) -> GraphBatch:
    """Return a graph with feature 6 derived from directed distance to value targets."""

    graph.validate(feature_dim=len(FEATURE_NAMES))
    features = graph.node_features.clone()
    active = graph.node_mask
    adjacency = (graph.edge_types > 0) & active[:, :, None] & active[:, None, :]
    targets = torch.zeros_like(active)
    for node_type in _TARGET_TYPES:
        targets |= graph.node_types == node_type
    targets &= active

    proximity = torch.zeros_like(features[:, :, VALUE_PROXIMITY_INDEX])
    reached = targets.clone()
    proximity = torch.where(targets, torch.ones_like(proximity), proximity)
    for value in (0.75, 0.5, 0.25):
        predecessors = (adjacency & reached[:, None, :]).any(dim=2) & active
        newly_reached = predecessors & ~reached
        proximity = torch.where(
            newly_reached,
            torch.full_like(proximity, value),
            proximity,
        )
        reached |= newly_reached
    features[:, :, VALUE_PROXIMITY_INDEX] = proximity * active
    return GraphBatch(
        node_types=graph.node_types,
        node_features=features,
        edge_types=graph.edge_types,
        node_mask=graph.node_mask,
    )


def compute_proxy_scores(graph: GraphBatch) -> Tensor:
    """Compute transparent structural targets for the uncalibrated score heads."""

    graph = with_value_proximity(graph)
    features = graph.node_features
    active = graph.node_mask

    target_mask = torch.zeros_like(active)
    for node_type in _TARGET_TYPES:
        target_mask |= graph.node_types == node_type
    target_mask &= active
    utility_values = (
        0.30 * features[:, :, 0]
        + 0.20 * features[:, :, 1]
        + 0.15 * features[:, :, 3]
        + 0.15 * features[:, :, 4]
        + 0.10 * features[:, :, 5]
        + 0.10 * features[:, :, 6]
    )
    utility = _masked_mean(utility_values, target_mask)

    enabler_mask = torch.zeros_like(active)
    for node_type in _ENABLER_TYPES:
        enabler_mask |= graph.node_types == node_type
    enabler_mask &= active
    feasibility_values = (
        0.30 * features[:, :, 2]
        + 0.25 * features[:, :, 1]
        + 0.15 * features[:, :, 3]
        + 0.15 * features[:, :, 5]
        + 0.15 * (1.0 - features[:, :, RISK_INDEX])
    )
    feasibility = _masked_mean(feasibility_values, enabler_mask)
    constraint_mask = (graph.node_types == int(NodeType.CONSTRAINT)) & active
    constraint_pressure = _masked_mean(
        features[:, :, 0] * features[:, :, RISK_INDEX], constraint_mask
    )
    has_constraint = constraint_mask.any(dim=1)
    feasibility = torch.clamp(
        feasibility - 0.20 * constraint_pressure * has_constraint,
        min=0.0,
        max=1.0,
    )

    coverage = _required_type_coverage(graph)
    reachable = _masked_mean((features[:, :, 6] > 0).to(features.dtype), active)
    connected = _largest_weak_component_ratio(graph)
    feedback_mask = (graph.node_types == int(NodeType.FEEDBACK)) & active
    feedback_connected = (
        feedback_mask
        & (
            (graph.edge_types > 0).any(dim=1)
            | (graph.edge_types > 0).any(dim=2)
        )
    ).any(dim=1)
    coherence = torch.clamp(
        0.35 * coverage
        + 0.35 * reachable
        + 0.20 * connected
        + 0.10 * feedback_connected.to(features.dtype),
        min=0.0,
        max=1.0,
    )
    return torch.stack((utility, feasibility, coherence), dim=1)


def _masked_mean(values: Tensor, mask: Tensor) -> Tensor:
    weights = mask.to(values.dtype)
    return (values * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1)


def _required_type_coverage(graph: GraphBatch) -> Tensor:
    groups = (
        (NodeType.ACTOR,),
        (NodeType.NEED,),
        (NodeType.RESOURCE, NodeType.ACTION),
        (NodeType.VALUE,),
        (NodeType.REVENUE,),
        (NodeType.OUTCOME,),
    )
    present: list[Tensor] = []
    for group in groups:
        group_present = torch.zeros(
            graph.batch_size, dtype=torch.bool, device=graph.node_mask.device
        )
        for node_type in group:
            group_present |= ((graph.node_types == int(node_type)) & graph.node_mask).any(dim=1)
        present.append(group_present)
    return torch.stack(present, dim=1).to(graph.node_features.dtype).mean(dim=1)


def _largest_weak_component_ratio(graph: GraphBatch) -> Tensor:
    ratios: list[float] = []
    adjacency = graph.edge_types > 0
    for batch_index in range(graph.batch_size):
        active_nodes = torch.where(graph.node_mask[batch_index])[0].tolist()
        if not active_nodes:
            ratios.append(0.0)
            continue
        remaining = {int(index) for index in active_nodes}
        largest = 0
        while remaining:
            frontier = [remaining.pop()]
            size = 0
            while frontier:
                node = frontier.pop()
                size += 1
                linked = adjacency[batch_index, node] | adjacency[batch_index, :, node]
                neighbors = {int(index) for index in torch.where(linked)[0].tolist()}
                discovered = remaining & neighbors
                remaining -= discovered
                frontier.extend(discovered)
            largest = max(largest, size)
        ratios.append(largest / len(active_nodes))
    return torch.tensor(ratios, dtype=graph.node_features.dtype, device=graph.node_features.device)
