"""Validated configuration for H002 development and evidence runs."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass
from enum import Enum
from pathlib import Path
from typing import Any, cast

import yaml

from chimera.meta_world.config import MetaWorldModelConfig, MetaWorldTrainingConfig
from chimera.meta_world.generators import GeneratedWorldDatasetConfig


class H002Arm(str, Enum):
    ALIGNED = "cross_world_pretraining_with_mechanism_alignment"
    NO_ALIGNMENT = "cross_world_pretraining_without_mechanism_alignment"
    TARGET_FAMILY_ONLY = "target_family_only_training"
    TEMPORAL = "temporal_predictor_without_relational_state"
    RANDOM = "legal_random_intervention"


def _mapping(value: object, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{name} must be a mapping")
    return cast(Mapping[str, Any], value)


@dataclass(frozen=True)
class H002EvaluationConfig:
    evaluation_interval: int
    validation_trajectories: int
    rollout_horizon: int

    def __post_init__(self) -> None:
        if self.evaluation_interval <= 0 or self.validation_trajectories <= 0:
            raise ValueError("evaluation interval and trajectory count must be positive")
        if self.rollout_horizon <= 0:
            raise ValueError("rollout_horizon must be positive")

    @classmethod
    def from_mapping(cls, values: Mapping[str, Any]) -> H002EvaluationConfig:
        allowed = {"evaluation_interval", "validation_trajectories", "rollout_horizon"}
        unknown = set(values) - allowed
        if unknown:
            raise ValueError(f"unknown H002 evaluation fields: {sorted(unknown)}")
        return cls(
            evaluation_interval=int(values["evaluation_interval"]),
            validation_trajectories=int(values["validation_trajectories"]),
            rollout_horizon=int(values["rollout_horizon"]),
        )


@dataclass(frozen=True)
class H002RunConfig:
    """Complete model, optimizer, generator and validation-only run contract."""

    run_id: str
    mode: str
    arm: H002Arm
    generator_config: Path
    model: MetaWorldModelConfig
    training: MetaWorldTrainingConfig
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
            raise ValueError("H002 action adapter requires one type and three parameters")
        if self.model.effect_dimensions != generator.outcome_features:
            raise ValueError("model effect head differs from generated outcomes")
        if self.model.domain_count != 1:
            raise ValueError("H002 forbids service-domain IDs in model inputs")
        if self.training.batch_size % generator.views_per_mechanism:
            raise ValueError("training batch must contain complete mechanism views")
        if self.evaluation.validation_trajectories % generator.views_per_mechanism:
            raise ValueError("validation set must contain complete mechanism views")
        if self.arm in {H002Arm.NO_ALIGNMENT, H002Arm.TEMPORAL}:
            if self.training.alignment_weight != 0.0:
                raise ValueError("this H002 arm must disable mechanism alignment")
        elif (
            self.arm in {H002Arm.ALIGNED, H002Arm.TARGET_FAMILY_ONLY}
            and self.training.alignment_weight <= 0.0
        ):
            raise ValueError("this H002 arm requires mechanism alignment")
        if self.arm is H002Arm.RANDOM:
            raise ValueError("legal random intervention has no trainable run config")

    @classmethod
    def from_mapping(cls, values: Mapping[str, Any]) -> H002RunConfig:
        allowed = {
            "run_id",
            "mode",
            "arm",
            "generator_config",
            "model",
            "training",
            "evaluation",
        }
        unknown = set(values) - allowed
        if unknown:
            raise ValueError(f"unknown H002 run fields: {sorted(unknown)}")
        return cls(
            run_id=str(values["run_id"]),
            mode=str(values["mode"]),
            arm=H002Arm(str(values["arm"])),
            generator_config=Path(str(values["generator_config"])),
            model=MetaWorldModelConfig.from_mapping(_mapping(values["model"], "model")),
            training=MetaWorldTrainingConfig.from_mapping(
                _mapping(values["training"], "training")
            ),
            evaluation=H002EvaluationConfig.from_mapping(
                _mapping(values["evaluation"], "evaluation")
            ),
        )

    @classmethod
    def from_yaml(cls, path: str | Path) -> H002RunConfig:
        with Path(path).open("r", encoding="utf-8") as handle:
            values = yaml.safe_load(handle)
        return cls.from_mapping(_mapping(values, "H002 run config"))

    def to_dict(self) -> dict[str, object]:
        return {
            "run_id": self.run_id,
            "mode": self.mode,
            "arm": self.arm.value,
            "generator_config": self.generator_config.as_posix(),
            "model": asdict(self.model),
            "training": asdict(self.training),
            "evaluation": asdict(self.evaluation),
        }
