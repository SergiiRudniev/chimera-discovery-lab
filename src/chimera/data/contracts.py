"""Strict tensor contracts for structured, non-text model inputs."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor


def _require_shape(name: str, tensor: Tensor, shape: tuple[int | None, ...]) -> None:
    if tensor.ndim != len(shape):
        raise ValueError(f"{name} must have {len(shape)} dimensions, got {tensor.ndim}")
    for axis, (actual, expected) in enumerate(zip(tensor.shape, shape, strict=True)):
        if expected is not None and actual != expected:
            raise ValueError(f"{name} axis {axis} must be {expected}, got {actual}")


@dataclass(frozen=True)
class GraphBatch:
    """A batch of padded typed graphs with numeric node attributes."""

    node_types: Tensor
    node_features: Tensor
    edge_types: Tensor
    node_mask: Tensor

    def validate(self, *, feature_dim: int | None = None) -> GraphBatch:
        _require_shape("node_types", self.node_types, (None, None))
        batch, nodes = self.node_types.shape
        _require_shape("node_features", self.node_features, (batch, nodes, feature_dim))
        _require_shape("edge_types", self.edge_types, (batch, nodes, nodes))
        _require_shape("node_mask", self.node_mask, (batch, nodes))
        if self.node_types.dtype != torch.long or self.edge_types.dtype != torch.long:
            raise TypeError("node_types and edge_types must be torch.long")
        if self.node_mask.dtype != torch.bool:
            raise TypeError("node_mask must be torch.bool")
        if not self.node_features.is_floating_point():
            raise TypeError("node_features must be floating point")
        if torch.any(self.node_types[~self.node_mask] != 0):
            raise ValueError("padded nodes must use node type 0")
        if torch.any(self.edge_types < 0) or torch.any(self.node_types < 0):
            raise ValueError("categorical IDs must be non-negative")
        return self

    @property
    def batch_size(self) -> int:
        return int(self.node_types.shape[0])

    @property
    def max_nodes(self) -> int:
        return int(self.node_types.shape[1])

    def to(self, device: torch.device | str) -> GraphBatch:
        return GraphBatch(
            node_types=self.node_types.to(device),
            node_features=self.node_features.to(device),
            edge_types=self.edge_types.to(device),
            node_mask=self.node_mask.to(device),
        )

    def clone(self) -> GraphBatch:
        return GraphBatch(
            node_types=self.node_types.clone(),
            node_features=self.node_features.clone(),
            edge_types=self.edge_types.clone(),
            node_mask=self.node_mask.clone(),
        )


@dataclass(frozen=True)
class EditBatch:
    """A padded batch of discrete graph-edit programs."""

    operations: Tensor
    source_nodes: Tensor
    target_nodes: Tensor
    node_types: Tensor
    edge_types: Tensor
    step_mask: Tensor

    def validate(
        self, *, batch_size: int | None = None, max_nodes: int | None = None
    ) -> EditBatch:
        _require_shape("operations", self.operations, (batch_size, None))
        shape = tuple(self.operations.shape)
        for name, value in (
            ("source_nodes", self.source_nodes),
            ("target_nodes", self.target_nodes),
            ("node_types", self.node_types),
            ("edge_types", self.edge_types),
            ("step_mask", self.step_mask),
        ):
            _require_shape(name, value, shape)
        for name, value in (
            ("operations", self.operations),
            ("source_nodes", self.source_nodes),
            ("target_nodes", self.target_nodes),
            ("node_types", self.node_types),
            ("edge_types", self.edge_types),
        ):
            if value.dtype != torch.long:
                raise TypeError(f"{name} must be torch.long")
            if torch.any(value < 0):
                raise ValueError(f"{name} IDs must be non-negative")
        if self.step_mask.dtype != torch.bool:
            raise TypeError("step_mask must be torch.bool")
        if max_nodes is not None and (
            torch.any(self.source_nodes >= max_nodes)
            or torch.any(self.target_nodes >= max_nodes)
        ):
            raise ValueError("edit node index exceeds graph capacity")
        return self

    @property
    def steps(self) -> int:
        return int(self.operations.shape[1])

    def to(self, device: torch.device | str) -> EditBatch:
        return EditBatch(
            operations=self.operations.to(device),
            source_nodes=self.source_nodes.to(device),
            target_nodes=self.target_nodes.to(device),
            node_types=self.node_types.to(device),
            edge_types=self.edge_types.to(device),
            step_mask=self.step_mask.to(device),
        )


@dataclass(frozen=True)
class TrainingBatch:
    """One supervised world-transition batch."""

    graph: GraphBatch
    edits: EditBatch
    next_graph: GraphBatch
    scores: Tensor

    def validate(self, *, feature_dim: int, score_dimensions: int) -> TrainingBatch:
        self.graph.validate(feature_dim=feature_dim)
        self.next_graph.validate(feature_dim=feature_dim)
        if self.graph.node_types.shape != self.next_graph.node_types.shape:
            raise ValueError("graph and next_graph must share padded shape")
        self.edits.validate(batch_size=self.graph.batch_size, max_nodes=self.graph.max_nodes)
        _require_shape("scores", self.scores, (self.graph.batch_size, score_dimensions))
        if not self.scores.is_floating_point():
            raise TypeError("scores must be floating point")
        if torch.any((self.scores < 0) | (self.scores > 1)):
            raise ValueError("scores must be normalized to [0, 1]")
        return self

    def to(self, device: torch.device | str) -> TrainingBatch:
        return TrainingBatch(
            graph=self.graph.to(device),
            edits=self.edits.to(device),
            next_graph=self.next_graph.to(device),
            scores=self.scores.to(device),
        )
