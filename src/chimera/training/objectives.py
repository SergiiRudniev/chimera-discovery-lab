"""Registered multi-objective loss for structured idea transitions."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import Tensor

from chimera.data.contracts import EditBatch
from chimera.models.venture import ChimeraOutput
from chimera.schema import EditOperation


@dataclass(frozen=True)
class LossWeights:
    operation: float = 1.0
    pointer: float = 0.5
    node_type: float = 0.25
    edge_type: float = 0.25
    score: float = 0.5
    transition: float = 1.0
    entropy: float = 0.01


def _masked_cross_entropy(logits: Tensor, target: Tensor, mask: Tensor) -> Tensor:
    losses = F.cross_entropy(logits.transpose(1, 2), target, reduction="none")
    weights = mask.to(losses.dtype)
    return (losses * weights).sum() / weights.sum().clamp_min(1)


def edit_argument_masks(edits: EditBatch) -> dict[str, Tensor]:
    """Return operation-specific supervision masks for edit arguments."""

    active = edits.step_mask
    operations = edits.operations

    def operation_mask(*values: EditOperation) -> Tensor:
        selected = torch.zeros_like(active)
        for value in values:
            selected |= operations == int(value)
        return active & selected

    return {
        "source": operation_mask(
            EditOperation.ADD_NODE,
            EditOperation.CONNECT,
            EditOperation.REWIRE,
            EditOperation.REMOVE_CONSTRAINT,
            EditOperation.INVERT_RELATION,
            EditOperation.SUBSTITUTE,
            EditOperation.MERGE,
        ),
        "target": operation_mask(
            EditOperation.CONNECT,
            EditOperation.REWIRE,
            EditOperation.TRANSFER_ROLE,
            EditOperation.INVERT_RELATION,
            EditOperation.MERGE,
        ),
        "node_type": operation_mask(
            EditOperation.ADD_NODE,
            EditOperation.TRANSFER_ROLE,
            EditOperation.SUBSTITUTE,
        ),
        "edge_type": operation_mask(
            EditOperation.ADD_NODE,
            EditOperation.CONNECT,
            EditOperation.REWIRE,
        ),
    }


def chimera_loss(
    output: ChimeraOutput,
    edits: EditBatch,
    scores: Tensor,
    target_next_state: Tensor,
    weights: LossWeights | None = None,
    *,
    argument_loss_mode: str = "all_fields",
) -> dict[str, Tensor]:
    """Compute edit reconstruction, world prediction and diversity pressure."""

    weights = weights or LossWeights()
    operation = _masked_cross_entropy(output.operation_logits, edits.operations, edits.step_mask)
    if argument_loss_mode == "all_fields":
        argument_masks = dict.fromkeys(
            ("source", "target", "node_type", "edge_type"), edits.step_mask
        )
    elif argument_loss_mode == "operation_conditioned":
        argument_masks = edit_argument_masks(edits)
    else:
        raise ValueError("unknown argument_loss_mode")
    source = _masked_cross_entropy(
        output.source_logits, edits.source_nodes, argument_masks["source"]
    )
    target = _masked_cross_entropy(
        output.target_logits, edits.target_nodes, argument_masks["target"]
    )
    node_type = _masked_cross_entropy(
        output.node_type_logits, edits.node_types, argument_masks["node_type"]
    )
    edge_type = _masked_cross_entropy(
        output.edge_type_logits, edits.edge_types, argument_masks["edge_type"]
    )
    score = F.binary_cross_entropy_with_logits(output.score_logits, scores)
    transition = (
        1.0 - F.cosine_similarity(output.predicted_next_state, target_next_state.detach(), dim=-1)
    ).mean()
    operation_probabilities = torch.softmax(output.operation_logits, dim=-1)
    entropy_per_step = -(
        operation_probabilities
        * torch.log(
            operation_probabilities.clamp_min(torch.finfo(operation_probabilities.dtype).tiny)
        )
    ).sum(dim=-1)
    mask = edits.step_mask.to(entropy_per_step.dtype)
    entropy = (entropy_per_step * mask).sum() / mask.sum().clamp_min(1)

    total = (
        weights.operation * operation
        + weights.pointer * (source + target)
        + weights.node_type * node_type
        + weights.edge_type * edge_type
        + weights.score * score
        + weights.transition * transition
        - weights.entropy * entropy
    )
    return {
        "loss": total,
        "operation_loss": operation,
        "source_loss": source,
        "target_loss": target,
        "node_type_loss": node_type,
        "edge_type_loss": edge_type,
        "score_loss": score,
        "transition_loss": transition,
        "operation_entropy": entropy,
    }
