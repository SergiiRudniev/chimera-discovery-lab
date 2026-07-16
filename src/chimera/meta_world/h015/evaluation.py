"""Model scoring and exact simulator replay for H015 candidates."""

from __future__ import annotations

from dataclasses import dataclass
from typing import cast

import numpy as np
import torch
from numpy.typing import NDArray
from torch import Tensor, nn

from chimera.meta_world.batch import MetaWorldBatch
from chimera.meta_world.generators import (
    GeneratedWorldBatch,
    GeneratedWorldDatasetConfig,
    MechanismGenerator,
    ViewCoupling,
    WorldAction,
    WorldFamily,
    WorldGenerator,
    WorldTrajectory,
)
from chimera.meta_world.h002.windows import GeneratedSequenceSample
from chimera.meta_world.h014.model import ResponseConditionedEffectWorldModel
from chimera.meta_world.h015.search import InterventionCandidate
from chimera.meta_world.model import MetaWorldOutput


def _repeat_tensor(value: Tensor, count: int) -> Tensor:
    return value.repeat((count,) + (1,) * (value.ndim - 1))


def candidate_batch(
    window: MetaWorldBatch,
    candidates: tuple[InterventionCandidate, ...],
) -> MetaWorldBatch:
    """Repeat one state and replace only the legal intervention input."""

    count = len(candidates)
    if window.batch_size != 1 or count <= 0:
        raise ValueError("candidate scoring requires one state and candidates")
    repeated = MetaWorldBatch(
        observations=_repeat_tensor(window.observations, count),
        observation_mask=_repeat_tensor(window.observation_mask, count),
        slot_mask=_repeat_tensor(window.slot_mask, count),
        relations=_repeat_tensor(window.relations, count),
        time_mask=_repeat_tensor(window.time_mask, count),
        domain_ids=_repeat_tensor(window.domain_ids, count),
        intervention_types=_repeat_tensor(window.intervention_types, count),
        source_slots=torch.tensor(
            [item.source_slot for item in candidates], dtype=torch.long
        ),
        target_slots=torch.tensor(
            [item.target_slot for item in candidates], dtype=torch.long
        ),
        intervention_parameters=torch.tensor(
            [
                [item.magnitude, item.control, float(window.intervention_parameters[0, 2])]
                for item in candidates
            ],
            dtype=window.intervention_parameters.dtype,
        ),
        next_observations=_repeat_tensor(window.next_observations, count),
        next_observation_mask=_repeat_tensor(window.next_observation_mask, count),
        effect_targets=_repeat_tensor(window.effect_targets, count),
        mechanism_ids=_repeat_tensor(window.mechanism_ids, count),
        action_history=(
            None
            if window.action_history is None
            else _repeat_tensor(window.action_history, count)
        ),
        action_target_history=(
            None
            if window.action_target_history is None
            else _repeat_tensor(window.action_target_history, count)
        ),
        counterfactual_no_op_observations=(
            None
            if window.counterfactual_no_op_observations is None
            else _repeat_tensor(window.counterfactual_no_op_observations, count)
        ),
    )
    repeated.validate()
    return repeated


def slice_sequence_sample(
    sample: GeneratedSequenceSample,
    index: int,
) -> GeneratedSequenceSample:
    """Select one evaluator trajectory without exposing its metadata forward."""

    if not 0 <= index < sample.batch.batch_size:
        raise IndexError("sequence sample index is out of range")
    batch = sample.batch
    sliced = GeneratedWorldBatch(
        observations=batch.observations[index : index + 1],
        object_mask=batch.object_mask[index : index + 1],
        relations=batch.relations[index : index + 1],
        relation_mask=batch.relation_mask[index : index + 1],
        actions=batch.actions[index : index + 1],
        action_targets=batch.action_targets[index : index + 1],
        delta_time=batch.delta_time[index : index + 1],
        outcomes=batch.outcomes[index : index + 1],
        sequence_mask=batch.sequence_mask[index : index + 1],
        counterfactual_no_op_observations=(
            None
            if batch.counterfactual_no_op_observations is None
            else batch.counterfactual_no_op_observations[index : index + 1]
        ),
    )
    return GeneratedSequenceSample(
        batch=sliced,
        mechanism_ids=sample.mechanism_ids[index : index + 1],
        mechanism_keys=sample.mechanism_keys[index : index + 1],
        world_instance_keys=sample.world_instance_keys[index : index + 1],
        world_family_ids=sample.world_family_ids[index : index + 1],
        renderer_profile_ids=sample.renderer_profile_ids[index : index + 1],
        trajectory_indices=sample.trajectory_indices[index : index + 1],
        state_features=sample.state_features,
    )


@dataclass
class CheckpointCandidatePredictor:
    """Frozen CUDA/CPU model adapter returning effect mean and standard deviation."""

    model: nn.Module
    device: torch.device
    use_autocast: bool

    @torch.no_grad()
    def predict(
        self,
        window: MetaWorldBatch,
        candidates: tuple[InterventionCandidate, ...],
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        self.model.eval()
        batch = candidate_batch(window, candidates).to(self.device)
        with torch.autocast(
            device_type=self.device.type,
            dtype=torch.bfloat16,
            enabled=self.use_autocast,
        ):
            output = cast(MetaWorldOutput, self.model(batch))
        mean = output.effect_mean[:, 3].float()
        standard_deviation = torch.exp(
            0.5 * output.effect_log_variance[:, 3].float()
        )
        return (
            mean.detach().cpu().numpy().astype(np.float64),
            standard_deviation.detach().cpu().numpy().astype(np.float64),
        )


def load_candidate_predictor(
    model: ResponseConditionedEffectWorldModel,
    checkpoint_path: str,
    *,
    device: torch.device,
    use_autocast: bool,
) -> CheckpointCandidatePredictor:
    """Load the validation-selected EMA checkpoint with a strict key audit."""

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    model.load_state_dict(checkpoint["model"], strict=True)
    return CheckpointCandidatePredictor(
        model=model.to(device),
        device=device,
        use_autocast=use_autocast,
    )


def realized_candidate_effect(
    config: GeneratedWorldDatasetConfig,
    trajectory: WorldTrajectory,
    *,
    prediction_step: int,
    candidate: InterventionCandidate,
) -> float:
    """Reset, replay history and execute a candidate under the same event stream."""

    metadata = trajectory.metadata
    mechanism = MechanismGenerator().generate(
        metadata.mechanism_template_id,
        metadata.mechanism_seed,
    )
    world = WorldGenerator(
        min_objects=config.min_objects,
        max_objects=config.max_objects,
        observation_features=config.observation_features,
        relation_features=config.relation_features,
    ).generate(
        mechanism,
        WorldFamily(metadata.world_family_id),
        world_seed=metadata.world_seed,
        renderer_seed=metadata.renderer_seed,
        renderer_profile=metadata.renderer_profile_id,
        independent_renderer_rng=(
            config.view_coupling is ViewCoupling.PAIRED_WORLD_RENDERERS
        ),
    )
    world.reset(metadata.generation_seed)
    for transition in trajectory.transitions[:prediction_step]:
        world.step(transition.action)
    result = world.step(
        WorldAction(
            source=candidate.source_slot,
            target=candidate.target_slot,
            magnitude=candidate.magnitude,
            control=candidate.control,
        )
    )
    return float(result.outcome[3])


def uniform_legal_pool(
    *,
    objects: int,
    count: int,
    seed: int,
) -> tuple[InterventionCandidate, ...]:
    """Create the evaluator-only fixed legal candidate pool."""

    if objects <= 1 or count <= 0 or seed < 0:
        raise ValueError("invalid fixed-pool request")
    rng = np.random.default_rng(seed)
    sources = rng.integers(0, objects, size=count)
    offsets = rng.integers(1, objects, size=count)
    targets = (sources + offsets) % objects
    magnitudes = rng.uniform(0.0, 1.0, size=count)
    controls = rng.uniform(-1.0, 1.0, size=count)
    return tuple(
        InterventionCandidate(
            source_slot=int(sources[index]),
            target_slot=int(targets[index]),
            magnitude=float(magnitudes[index]),
            control=float(controls[index]),
        )
        for index in range(count)
    )
