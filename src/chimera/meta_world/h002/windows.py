"""Leakage-safe adapters from generated trajectories to W0 transition windows."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from chimera.meta_world.batch import MetaWorldBatch
from chimera.meta_world.generators import (
    GeneratedWorldBatch,
    SplitName,
    WorldGenerationPipeline,
    collate_trajectories,
)


@dataclass(frozen=True)
class GeneratedSequenceSample:
    """Model tensors plus evaluator-only labels kept outside GeneratedWorldBatch."""

    batch: GeneratedWorldBatch
    mechanism_ids: Tensor
    world_family_ids: Tensor
    renderer_profile_ids: Tensor
    trajectory_indices: Tensor
    state_features: int

    def __post_init__(self) -> None:
        expected = (self.batch.batch_size,)
        for name in (
            "mechanism_ids",
            "world_family_ids",
            "renderer_profile_ids",
            "trajectory_indices",
        ):
            if tuple(getattr(self, name).shape) != expected:
                raise ValueError(f"{name} must have shape {expected}")
        if not 0 < self.state_features <= self.batch.observations.shape[-1]:
            raise ValueError("state_features must fit the observation tensor")


def materialize_sequence_sample(
    pipeline: WorldGenerationPipeline,
    split: SplitName,
    *,
    start_index: int,
    batch_size: int,
) -> GeneratedSequenceSample:
    """Materialize complete mechanism-view groups without exposing metadata to W0."""

    views = pipeline.config.views_per_mechanism
    if batch_size <= 0 or batch_size % views:
        raise ValueError("batch_size must contain complete mechanism-view groups")
    if start_index < 0 or start_index % views:
        raise ValueError("start_index must begin at a mechanism-view group boundary")
    indices = list(range(start_index, start_index + batch_size))
    trajectories = [pipeline.materialize(split, index) for index in indices]
    sequence_batch = collate_trajectories(
        trajectories,
        max_objects=pipeline.config.max_objects,
    )
    mechanism_lookup: dict[str, int] = {}
    mechanism_values: list[int] = []
    for trajectory in trajectories:
        mechanism_id = trajectory.metadata.mechanism_id
        if mechanism_id not in mechanism_lookup:
            mechanism_lookup[mechanism_id] = len(mechanism_lookup)
        mechanism_values.append(mechanism_lookup[mechanism_id])
    return GeneratedSequenceSample(
        batch=sequence_batch,
        mechanism_ids=torch.tensor(mechanism_values, dtype=torch.long),
        world_family_ids=torch.tensor(
            [trajectory.metadata.world_family_id for trajectory in trajectories],
            dtype=torch.long,
        ),
        renderer_profile_ids=torch.tensor(
            [trajectory.metadata.renderer_profile_id for trajectory in trajectories],
            dtype=torch.long,
        ),
        trajectory_indices=torch.tensor(indices, dtype=torch.long),
        state_features=pipeline.config.state_features,
    )


def make_transition_window(
    sample: GeneratedSequenceSample,
    *,
    prediction_step: int,
    context_steps: int,
) -> MetaWorldBatch:
    """Create one causal W0 window; generator metadata remains outside model inputs."""

    generated = sample.batch
    generated.validate()
    batch, time, slots, features = generated.observations.shape
    if not 0 <= prediction_step < time - 1:
        raise ValueError("prediction_step must have a following observed state")
    if context_steps <= 0:
        raise ValueError("context_steps must be positive")
    history_start = max(0, prediction_step - context_steps + 1)
    history_length = prediction_step - history_start + 1
    observations = torch.zeros(
        batch,
        context_steps,
        slots,
        features,
        dtype=generated.observations.dtype,
    )
    observation_mask = torch.zeros_like(observations, dtype=torch.bool)
    slot_mask = torch.zeros(batch, context_steps, slots, dtype=torch.bool)
    relations = torch.zeros(
        batch,
        context_steps,
        slots,
        slots,
        generated.relations.shape[-1],
        dtype=generated.relations.dtype,
    )
    time_mask = torch.zeros(batch, context_steps, dtype=torch.bool)
    action_history = torch.zeros(
        batch,
        context_steps,
        generated.actions.shape[-1] + 1,
        dtype=generated.actions.dtype,
    )
    action_target_history = torch.zeros(
        batch,
        context_steps,
        slots,
        dtype=generated.action_targets.dtype,
    )
    history = slice(history_start, prediction_step + 1)
    observations[:, :history_length] = generated.observations[:, history]
    slot_mask[:, :history_length] = generated.object_mask[:, history]
    observation_mask[:, :history_length] = generated.object_mask[
        :, history
    ].unsqueeze(-1)
    relations[:, :history_length] = generated.relations[:, history] * generated.relation_mask[
        :, history
    ].unsqueeze(-1).to(generated.relations.dtype)
    time_mask[:, :history_length] = generated.sequence_mask[:, history]
    if history_length > 1:
        prior_actions = slice(history_start, prediction_step)
        action_history[:, 1:history_length] = torch.cat(
            [
                generated.actions[:, prior_actions],
                generated.delta_time[:, prior_actions, None],
            ],
            dim=-1,
        )
        action_target_history[:, 1:history_length] = generated.action_targets[
            :, prior_actions
        ]
    action_targets = generated.action_targets[:, prediction_step]
    source_slots = action_targets.argmin(dim=1)
    target_slots = action_targets.argmax(dim=1)
    intervention_parameters = torch.cat(
        [
            generated.actions[:, prediction_step],
            generated.delta_time[:, prediction_step, None],
        ],
        dim=1,
    )
    next_observations = generated.observations[:, prediction_step + 1]
    next_slot_mask = generated.object_mask[:, prediction_step + 1]
    next_observation_mask = torch.zeros_like(next_observations, dtype=torch.bool)
    next_observation_mask[:, :, : sample.state_features] = next_slot_mask.unsqueeze(-1)
    window = MetaWorldBatch(
        observations=observations,
        observation_mask=observation_mask,
        slot_mask=slot_mask,
        relations=relations,
        time_mask=time_mask,
        domain_ids=torch.zeros(batch, dtype=torch.long),
        intervention_types=torch.zeros(batch, dtype=torch.long),
        source_slots=source_slots,
        target_slots=target_slots,
        intervention_parameters=intervention_parameters,
        next_observations=next_observations,
        next_observation_mask=next_observation_mask,
        effect_targets=generated.outcomes[:, prediction_step],
        mechanism_ids=sample.mechanism_ids,
        action_history=action_history,
        action_target_history=action_target_history,
    )
    window.validate()
    return window
