"""Immutable run configuration for CHM-W-H003."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass
from enum import Enum
from pathlib import Path
from typing import Any, cast

import yaml

from chimera.meta_world.config import MetaWorldModelConfig, MetaWorldTrainingConfig
from chimera.meta_world.generators import GeneratedWorldDatasetConfig
from chimera.meta_world.h002.config import H002EvaluationConfig


class H003Arm(str, Enum):
    """Trainable H003 comparison arms."""

    CLOSED_LOOP_ALIGNED = "closed_loop_with_cross_batch_mechanism_discrimination"
    CLOSED_LOOP_NO_ALIGNMENT = "closed_loop_without_mechanism_discrimination"
    H002_ONE_STEP = "h002_one_step_relational_without_alignment"
    TEMPORAL = "temporal_predictor_without_relational_state"
    RANDOM = "legal_random_intervention"


def _mapping(value: object, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{name} must be a mapping")
    return cast(Mapping[str, Any], value)


@dataclass(frozen=True)
class H003ClosedLoopConfig:
    """Autoregressive horizon and detached mechanism-memory bounds."""

    rollout_horizon: int
    queue_minimum_entries: int
    queue_maximum_entries: int

    def __post_init__(self) -> None:
        if self.rollout_horizon <= 1:
            raise ValueError("H003 rollout_horizon must exceed one")
        if self.queue_minimum_entries <= 0:
            raise ValueError("queue_minimum_entries must be positive")
        if self.queue_maximum_entries < self.queue_minimum_entries:
            raise ValueError("queue maximum must be at least its minimum")

    @classmethod
    def from_mapping(cls, values: Mapping[str, Any]) -> H003ClosedLoopConfig:
        allowed = {
            "rollout_horizon",
            "queue_minimum_entries",
            "queue_maximum_entries",
        }
        unknown = set(values) - allowed
        if unknown:
            raise ValueError(f"unknown H003 closed-loop fields: {sorted(unknown)}")
        return cls(
            rollout_horizon=int(values["rollout_horizon"]),
            queue_minimum_entries=int(values["queue_minimum_entries"]),
            queue_maximum_entries=int(values["queue_maximum_entries"]),
        )


@dataclass(frozen=True)
class H003RunConfig:
    """Complete H003 model, optimizer, generator and validation contract."""

    run_id: str
    mode: str
    arm: H003Arm
    generator_config: Path
    model: MetaWorldModelConfig
    training: MetaWorldTrainingConfig
    closed_loop: H003ClosedLoopConfig
    evaluation: H002EvaluationConfig

    def __post_init__(self) -> None:
        if not self.run_id or self.mode not in {"preflight", "trial"}:
            raise ValueError("run_id and mode must be registered")
        generator = GeneratedWorldDatasetConfig.from_yaml(self.generator_config)
        if self.model.observation_features != generator.observation_features:
            raise ValueError("model observation features differ from generator")
        if self.model.relation_features != generator.relation_features:
            raise ValueError("model relation features differ from generator")
        if self.model.max_slots != generator.max_objects:
            raise ValueError("model max_slots differs from generator max_objects")
        if self.model.intervention_types != 1 or self.model.intervention_parameters != 3:
            raise ValueError("H003 action adapter requires one type and three parameters")
        if self.model.effect_dimensions != generator.outcome_features:
            raise ValueError("model effect head differs from generated outcomes")
        if self.model.domain_count != 1:
            raise ValueError("H003 forbids service-domain IDs in model inputs")
        if self.training.batch_size % generator.views_per_mechanism:
            raise ValueError("training batch must contain complete mechanism views")
        if self.evaluation.validation_trajectories % generator.views_per_mechanism:
            raise ValueError("validation set must contain complete mechanism views")
        if self.closed_loop.rollout_horizon != self.evaluation.rollout_horizon:
            raise ValueError("training and evaluation rollout horizons must match")
        available_predictions = generator.trajectory_steps - self.model.context_steps + 1
        if self.closed_loop.rollout_horizon > available_predictions:
            raise ValueError("generated trajectories are too short for H003 closed-loop training")
        if self.arm is H003Arm.CLOSED_LOOP_ALIGNED:
            if self.training.alignment_weight <= 0.0:
                raise ValueError("aligned H003 arm requires mechanism discrimination")
        elif self.training.alignment_weight != 0.0:
            raise ValueError("this H003 arm must disable mechanism discrimination")
        if self.arm is H003Arm.RANDOM:
            raise ValueError("legal random intervention has no trainable run config")

    @classmethod
    def from_mapping(cls, values: Mapping[str, Any]) -> H003RunConfig:
        allowed = {
            "run_id",
            "mode",
            "arm",
            "generator_config",
            "model",
            "training",
            "closed_loop",
            "evaluation",
        }
        unknown = set(values) - allowed
        if unknown:
            raise ValueError(f"unknown H003 run fields: {sorted(unknown)}")
        return cls(
            run_id=str(values["run_id"]),
            mode=str(values["mode"]),
            arm=H003Arm(str(values["arm"])),
            generator_config=Path(str(values["generator_config"])),
            model=MetaWorldModelConfig.from_mapping(_mapping(values["model"], "model")),
            training=MetaWorldTrainingConfig.from_mapping(
                _mapping(values["training"], "training")
            ),
            closed_loop=H003ClosedLoopConfig.from_mapping(
                _mapping(values["closed_loop"], "closed_loop")
            ),
            evaluation=H002EvaluationConfig.from_mapping(
                _mapping(values["evaluation"], "evaluation")
            ),
        )

    @classmethod
    def from_yaml(cls, path: str | Path) -> H003RunConfig:
        with Path(path).open("r", encoding="utf-8") as handle:
            values = yaml.safe_load(handle)
        return cls.from_mapping(_mapping(values, "H003 run config"))

    def to_dict(self) -> dict[str, object]:
        return {
            "run_id": self.run_id,
            "mode": self.mode,
            "arm": self.arm.value,
            "generator_config": self.generator_config.as_posix(),
            "model": asdict(self.model),
            "training": asdict(self.training),
            "closed_loop": asdict(self.closed_loop),
            "evaluation": asdict(self.evaluation),
        }
