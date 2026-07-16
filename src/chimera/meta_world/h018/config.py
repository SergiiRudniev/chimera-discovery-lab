"""Strict configuration for the preregistered H018 five-arm protocol."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import yaml

from chimera.meta_world.generators import GeneratedWorldDatasetConfig, SplitName
from chimera.meta_world.h002.config import H002Arm, H002RunConfig

ALIGNED = "compositional_cross_world_pretraining_with_mechanism_alignment"
NO_ALIGNMENT = "compositional_cross_world_pretraining_without_mechanism_alignment"
TARGET_FAMILY = "compositional_target_family_only_training"
TEMPORAL = "temporal_predictor_without_relational_world_state"
RANDOM = "legal_random_intervention_baseline"

_EXPECTED_ARMS = {ALIGNED, NO_ALIGNMENT, TARGET_FAMILY, TEMPORAL, RANDOM}
_EXPECTED_RUN_ARMS = {
    ALIGNED: H002Arm.ALIGNED,
    NO_ALIGNMENT: H002Arm.NO_ALIGNMENT,
    TARGET_FAMILY: H002Arm.TARGET_FAMILY_ONLY,
    TEMPORAL: H002Arm.TEMPORAL,
}
_PRIMARY_BASELINES = {NO_ALIGNMENT, TARGET_FAMILY, TEMPORAL}


def _mapping(value: object, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{name} must be a mapping")
    return cast(Mapping[str, Any], value)


@dataclass(frozen=True)
class H018ArmSpec:
    name: str
    config: Path | None
    primary_predictive_baseline: bool
    metric_scope: str | None


@dataclass(frozen=True)
class H018SuiteConfig:
    hypothesis_id: str
    trial_id: str
    generator_config: Path
    development_seed: int
    frozen_validation_seeds: tuple[int, ...]
    test_access: str
    arms: tuple[H018ArmSpec, ...]
    primary_split: SplitName
    primary_metrics: tuple[str, ...]

    def __post_init__(self) -> None:
        if self.hypothesis_id != "CHM-W-H018" or self.trial_id != "CHM-W-T018":
            raise ValueError("unexpected H018 suite identity")
        if self.development_seed != 260962:
            raise ValueError("unexpected H018 development seed")
        if self.development_seed in self.frozen_validation_seeds:
            raise ValueError("development and validation seeds must be disjoint")
        if self.frozen_validation_seeds != (260963, 260964, 260965):
            raise ValueError("H018 validation seeds differ from preregistration")
        if self.test_access != "sealed_until_all_validation_decisions_are_frozen":
            raise ValueError("H018 test access must remain sealed")
        if {arm.name for arm in self.arms} != _EXPECTED_ARMS:
            raise ValueError("H018 comparison arms do not match preregistration")
        if self.primary_split is not SplitName.TEST_WORLD_TRANSFER:
            raise ValueError("H018 primary split must be test_world_transfer")
        if self.primary_metrics != (
            "intervention_effect_nrmse",
            "four_step_rollout_nrmse",
        ):
            raise ValueError("H018 primary metrics do not match preregistration")
        generator = GeneratedWorldDatasetConfig.from_yaml(self.generator_config)
        if (
            generator.hypothesis_id != self.hypothesis_id
            or generator.dataset_id != "CHM-W-WG5"
        ):
            raise ValueError("H018 suite requires the registered WG5 generator")
        for arm in self.arms:
            if arm.primary_predictive_baseline != (arm.name in _PRIMARY_BASELINES):
                raise ValueError(
                    f"predictive-baseline eligibility is wrong for {arm.name}"
                )
            expected = _EXPECTED_RUN_ARMS.get(arm.name)
            if expected is None:
                if arm.config is not None or arm.primary_predictive_baseline:
                    raise ValueError("random baseline cannot have a model config")
                if arm.metric_scope != "intervention_regret_only":
                    raise ValueError(
                        "random baseline is restricted to intervention regret"
                    )
                continue
            if arm.config is None:
                raise ValueError(f"trainable arm {arm.name} requires a config")
            run = H002RunConfig.from_yaml(arm.config)
            if run.arm is not expected or run.generator_config != self.generator_config:
                raise ValueError(f"run config does not match arm {arm.name}")
            if run.training.seed != self.development_seed:
                raise ValueError(f"run seed does not match H018 for {arm.name}")

    @classmethod
    def from_yaml(cls, path: str | Path) -> H018SuiteConfig:
        with Path(path).open("r", encoding="utf-8") as handle:
            raw = yaml.safe_load(handle)
        values = _mapping(raw, "H018 suite")
        allowed = {
            "hypothesis_id",
            "trial_id",
            "generator_config",
            "development_seed",
            "frozen_validation_seeds",
            "test_access",
            "arms",
            "primary_split",
            "primary_metrics",
        }
        unknown = set(values) - allowed
        if unknown:
            raise ValueError(f"unknown H018 suite fields: {sorted(unknown)}")
        raw_arms = _mapping(values["arms"], "H018 arms")
        arms: list[H018ArmSpec] = []
        for name, value in raw_arms.items():
            arm = _mapping(value, f"H018 arm {name}")
            unknown_arm = set(arm) - {
                "config",
                "primary_predictive_baseline",
                "metric_scope",
            }
            if unknown_arm:
                raise ValueError(
                    f"unknown fields for H018 arm {name}: {sorted(unknown_arm)}"
                )
            config_value = arm.get("config")
            arms.append(
                H018ArmSpec(
                    name=str(name),
                    config=None if config_value is None else Path(str(config_value)),
                    primary_predictive_baseline=bool(
                        arm["primary_predictive_baseline"]
                    ),
                    metric_scope=(
                        None
                        if arm.get("metric_scope") is None
                        else str(arm["metric_scope"])
                    ),
                )
            )
        return cls(
            hypothesis_id=str(values["hypothesis_id"]),
            trial_id=str(values["trial_id"]),
            generator_config=Path(str(values["generator_config"])),
            development_seed=int(values["development_seed"]),
            frozen_validation_seeds=tuple(
                int(seed) for seed in values["frozen_validation_seeds"]
            ),
            test_access=str(values["test_access"]),
            arms=tuple(arms),
            primary_split=SplitName(str(values["primary_split"])),
            primary_metrics=tuple(str(metric) for metric in values["primary_metrics"]),
        )

