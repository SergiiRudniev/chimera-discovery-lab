"""Immutable model-run configuration for CHM-W-H004."""

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


class H004Arm(str, Enum):
    """Trainable active-identification comparison arms."""

    PROBE_ALIGNED = "probe_curriculum_closed_loop_with_mechanism_discrimination"
    RANDOM_ALIGNED = "random_curriculum_closed_loop_with_mechanism_discrimination"
    PROBE_NO_ALIGNMENT = "probe_curriculum_closed_loop_without_mechanism_discrimination"
    ONE_STEP = "h003_one_step_relational_without_alignment"
    TEMPORAL = "temporal_predictor_without_relational_state"
    RANDOM_INTERVENTION = "legal_random_intervention"


def _mapping(value: object, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{name} must be a mapping")
    return cast(Mapping[str, Any], value)


@dataclass(frozen=True)
class H004RunConfig:
    """Model, optimizer, action-policy and validation-only run contract."""

    run_id: str
    mode: str
    arm: H004Arm
    dataset_config_path: Path
    dataset: H004DatasetConfig
    model: MetaWorldModelConfig
    training: MetaWorldTrainingConfig
    closed_loop: H003ClosedLoopConfig
    evaluation: H002EvaluationConfig

    def __post_init__(self) -> None:
        if not self.run_id or self.mode not in {"preflight", "trial"}:
            raise ValueError("run_id and mode must be registered")
        worlds = self.dataset.worlds
        if self.model.observation_features != worlds.observation_features:
            raise ValueError("model observation features differ from WG1")
        if self.model.relation_features != worlds.relation_features:
            raise ValueError("model relation features differ from WG1")
        if self.model.max_slots != worlds.max_objects:
            raise ValueError("model max_slots differs from WG1")
        if self.model.intervention_types != 1 or self.model.intervention_parameters != 3:
            raise ValueError("H004 action adapter requires one type and three parameters")
        if self.model.effect_dimensions != worlds.outcome_features:
            raise ValueError("model effect head differs from WG1 outcomes")
        if self.model.domain_count != 1:
            raise ValueError("H004 forbids service-domain IDs in model inputs")
        if self.training.batch_size % worlds.views_per_mechanism:
            raise ValueError("training batch must contain complete mechanism views")
        if self.evaluation.validation_trajectories % worlds.views_per_mechanism:
            raise ValueError("validation must contain complete mechanism views")
        if self.closed_loop.rollout_horizon != self.evaluation.rollout_horizon:
            raise ValueError("training and evaluation horizons must match")
        first_rollout_step = self.model.context_steps - 1
        if first_rollout_step + self.closed_loop.rollout_horizon >= worlds.trajectory_steps:
            raise ValueError("WG1 is too short for the registered H004 rollout")
        aligned = {H004Arm.PROBE_ALIGNED, H004Arm.RANDOM_ALIGNED}
        if self.arm in aligned:
            if self.training.alignment_weight <= 0.0:
                raise ValueError("aligned H004 arms require mechanism discrimination")
        elif self.training.alignment_weight != 0.0:
            raise ValueError("this H004 arm must disable mechanism discrimination")
        if self.arm is H004Arm.RANDOM_INTERVENTION:
            raise ValueError("legal random intervention has no trainable run config")

    @classmethod
    def from_mapping(cls, values: Mapping[str, Any]) -> H004RunConfig:
        allowed = {
            "run_id",
            "mode",
            "arm",
            "dataset_config",
            "model",
            "training",
            "closed_loop",
            "evaluation",
        }
        unknown = set(values) - allowed
        if unknown:
            raise ValueError(f"unknown H004 run fields: {sorted(unknown)}")
        dataset_path = Path(str(values["dataset_config"]))
        return cls(
            run_id=str(values["run_id"]),
            mode=str(values["mode"]),
            arm=H004Arm(str(values["arm"])),
            dataset_config_path=dataset_path,
            dataset=H004DatasetConfig.from_yaml(dataset_path),
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
    def from_yaml(cls, path: str | Path) -> H004RunConfig:
        with Path(path).open("r", encoding="utf-8") as handle:
            values = yaml.safe_load(handle)
        return cls.from_mapping(_mapping(values, "H004 run config"))

    def to_dict(self) -> dict[str, object]:
        return {
            "run_id": self.run_id,
            "mode": self.mode,
            "arm": self.arm.value,
            "dataset_config": self.dataset_config_path.as_posix(),
            "model": asdict(self.model),
            "training": asdict(self.training),
            "closed_loop": asdict(self.closed_loop),
            "evaluation": asdict(self.evaluation),
        }
