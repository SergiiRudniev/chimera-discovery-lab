"""Immutable run configuration for CHM-W-H005."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass
from enum import Enum
from pathlib import Path
from typing import Any, cast

import yaml

from chimera.meta_world.config import MetaWorldModelConfig, MetaWorldTrainingConfig
from chimera.meta_world.h002.config import H002EvaluationConfig
from chimera.meta_world.h003.config import H003ClosedLoopConfig
from chimera.meta_world.h004.dataset import H004DatasetConfig


class H005Arm(str, Enum):
    """Trainable mixed-curriculum comparison arms."""

    MIXED = "mixed_probe_random_closed_loop_without_discrimination"
    RANDOM_ONLY = "random_only_closed_loop_without_discrimination"
    PROBE_ONLY = "probe_only_closed_loop_without_discrimination"
    ONE_STEP = "one_step_relational_without_discrimination"
    TEMPORAL = "temporal_predictor_without_relational_state"
    RANDOM_INTERVENTION = "legal_random_intervention"


def _mapping(value: object, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{name} must be a mapping")
    return cast(Mapping[str, Any], value)


@dataclass(frozen=True)
class H005CurriculumConfig:
    """Frozen policy mixture for one batch."""

    probe_fraction: float

    def __post_init__(self) -> None:
        if self.probe_fraction != 0.5:
            raise ValueError("H005 preregisters an exact 50:50 policy mixture")

    @classmethod
    def from_mapping(cls, values: Mapping[str, Any]) -> H005CurriculumConfig:
        if set(values) != {"probe_fraction"}:
            raise ValueError("H005 curriculum requires only probe_fraction")
        return cls(probe_fraction=float(values["probe_fraction"]))


@dataclass(frozen=True)
class H005RunConfig:
    """Model, policy mixture and validation-only execution contract."""

    run_id: str
    mode: str
    arm: H005Arm
    dataset_config_path: Path
    dataset: H004DatasetConfig
    model: MetaWorldModelConfig
    training: MetaWorldTrainingConfig
    curriculum: H005CurriculumConfig
    closed_loop: H003ClosedLoopConfig
    evaluation: H002EvaluationConfig
    frozen_checkpoint_step: int | None = None

    def __post_init__(self) -> None:
        if not self.run_id or self.mode not in {
            "preflight",
            "frozen_validation",
            "trial",
        }:
            raise ValueError("run_id and mode must be registered")
        if self.mode == "preflight" and self.frozen_checkpoint_step is not None:
            raise ValueError("development preflight cannot declare a frozen checkpoint")
        if self.mode == "frozen_validation":
            if self.training.seed not in {260911, 260912, 260913}:
                raise ValueError("H005 frozen validation seed is not preregistered")
            if self.frozen_checkpoint_step != self.training.steps:
                raise ValueError("H005 frozen validation must evaluate its final step")
        worlds = self.dataset.worlds
        if self.model.observation_features != worlds.observation_features:
            raise ValueError("model observation features differ from WG1")
        if self.model.relation_features != worlds.relation_features:
            raise ValueError("model relation features differ from WG1")
        if self.model.max_slots != worlds.max_objects:
            raise ValueError("model max_slots differs from WG1")
        if self.model.intervention_types != 1 or self.model.intervention_parameters != 3:
            raise ValueError("H005 action adapter requires one type and three parameters")
        if self.model.effect_dimensions != worlds.outcome_features:
            raise ValueError("model effect head differs from WG1")
        if self.model.domain_count != 1:
            raise ValueError("H005 forbids service-domain IDs in model inputs")
        views = worlds.views_per_mechanism
        if self.training.batch_size % views:
            raise ValueError("training batch must contain complete mechanism views")
        if self.arm in {H005Arm.MIXED, H005Arm.RANDOM_ONLY} and (
            self.training.batch_size % (2 * views)
        ):
            raise ValueError("paired batch halves must contain complete view groups")
        if self.evaluation.validation_trajectories % views:
            raise ValueError("validation must contain complete mechanism views")
        if self.closed_loop.rollout_horizon != self.evaluation.rollout_horizon:
            raise ValueError("training and evaluation horizons must match")
        first_rollout_step = self.model.context_steps - 1
        if first_rollout_step + self.closed_loop.rollout_horizon >= worlds.trajectory_steps:
            raise ValueError("WG1 is too short for the registered H005 rollout")
        if self.training.alignment_weight != 0.0:
            raise ValueError("H005 disables instance-discrimination loss in every arm")
        if self.arm is H005Arm.RANDOM_INTERVENTION:
            raise ValueError("legal random intervention has no trainable run config")

    @classmethod
    def from_mapping(cls, values: Mapping[str, Any]) -> H005RunConfig:
        allowed = {
            "run_id",
            "mode",
            "arm",
            "dataset_config",
            "model",
            "training",
            "curriculum",
            "closed_loop",
            "evaluation",
            "frozen_checkpoint_step",
        }
        unknown = set(values) - allowed
        if unknown:
            raise ValueError(f"unknown H005 run fields: {sorted(unknown)}")
        dataset_path = Path(str(values["dataset_config"]))
        return cls(
            run_id=str(values["run_id"]),
            mode=str(values["mode"]),
            arm=H005Arm(str(values["arm"])),
            dataset_config_path=dataset_path,
            dataset=H004DatasetConfig.from_yaml(dataset_path),
            model=MetaWorldModelConfig.from_mapping(_mapping(values["model"], "model")),
            training=MetaWorldTrainingConfig.from_mapping(
                _mapping(values["training"], "training")
            ),
            curriculum=H005CurriculumConfig.from_mapping(
                _mapping(values["curriculum"], "curriculum")
            ),
            closed_loop=H003ClosedLoopConfig.from_mapping(
                _mapping(values["closed_loop"], "closed_loop")
            ),
            evaluation=H002EvaluationConfig.from_mapping(
                _mapping(values["evaluation"], "evaluation")
            ),
            frozen_checkpoint_step=(
                int(values["frozen_checkpoint_step"])
                if values.get("frozen_checkpoint_step") is not None
                else None
            ),
        )

    @classmethod
    def from_yaml(cls, path: str | Path) -> H005RunConfig:
        with Path(path).open("r", encoding="utf-8") as handle:
            values = yaml.safe_load(handle)
        return cls.from_mapping(_mapping(values, "H005 run config"))

    def to_dict(self) -> dict[str, object]:
        return {
            "run_id": self.run_id,
            "mode": self.mode,
            "arm": self.arm.value,
            "dataset_config": self.dataset_config_path.as_posix(),
            "model": asdict(self.model),
            "training": asdict(self.training),
            "curriculum": asdict(self.curriculum),
            "closed_loop": asdict(self.closed_loop),
            "evaluation": asdict(self.evaluation),
            "frozen_checkpoint_step": self.frozen_checkpoint_step,
        }
