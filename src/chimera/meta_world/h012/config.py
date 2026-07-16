"""Strict suite configuration for the preregistered H012 comparison."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import yaml

from chimera.meta_world.generators import GeneratedWorldDatasetConfig, SplitName
from chimera.meta_world.h002.config import H002Arm, H002RunConfig

ALIGNED = "cross_world_pretraining_with_mechanism_alignment"
NO_ALIGNMENT = "cross_world_pretraining_without_mechanism_alignment"
TARGET_FAMILY = "target_family_only_training"
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
class H012ArmSpec:
    """One immutable comparison arm and its metric eligibility."""

    name: str
    config: Path | None
    primary_predictive_baseline: bool
    metric_scope: str | None


@dataclass(frozen=True)
class H012SuiteConfig:
    """Complete five-arm protocol with explicit sealed-test semantics."""

    hypothesis_id: str
    trial_id: str
    generator_config: Path
    development_seed: int
    frozen_validation_seeds: tuple[int, ...]
    test_access: str
    arms: tuple[H012ArmSpec, ...]
    primary_split: SplitName
    primary_metrics: tuple[str, ...]

    def __post_init__(self) -> None:
        if self.hypothesis_id != "CHM-W-H012" or self.trial_id != "CHM-W-T012":
            raise ValueError("unexpected H012 suite identity")
        if self.development_seed in self.frozen_validation_seeds:
            raise ValueError("development and validation seeds must be disjoint")
        if len(self.frozen_validation_seeds) != 3 or len(
            set(self.frozen_validation_seeds)
        ) != 3:
            raise ValueError("H012 requires three distinct validation seeds")
        if self.test_access != "sealed_until_all_validation_decisions_are_frozen":
            raise ValueError("H012 test access must remain sealed")
        if {arm.name for arm in self.arms} != _EXPECTED_ARMS:
            raise ValueError("H012 comparison arms do not match preregistration")
        if self.primary_split is not SplitName.TEST_WORLD_TRANSFER:
            raise ValueError("H012 primary split must be test_world_transfer")
        if self.primary_metrics != (
            "intervention_effect_nrmse",
            "four_step_rollout_nrmse",
        ):
            raise ValueError("H012 primary metrics do not match preregistration")
        generator = GeneratedWorldDatasetConfig.from_yaml(self.generator_config)
        if generator.hypothesis_id != self.hypothesis_id or generator.dataset_id != "CHM-W-WG3":
            raise ValueError("H012 suite requires the registered WG3 generator")
        for arm in self.arms:
            if arm.primary_predictive_baseline != (arm.name in _PRIMARY_BASELINES):
                raise ValueError(f"predictive-baseline eligibility is wrong for {arm.name}")
            expected = _EXPECTED_RUN_ARMS.get(arm.name)
            if expected is None:
                if arm.config is not None or arm.primary_predictive_baseline:
                    raise ValueError("random baseline cannot have a model config")
                if arm.metric_scope != "intervention_regret_only":
                    raise ValueError("random baseline is restricted to intervention regret")
                continue
            if arm.config is None:
                raise ValueError(f"trainable arm {arm.name} requires a config")
            run = H002RunConfig.from_yaml(arm.config)
            if run.arm is not expected or run.generator_config != self.generator_config:
                raise ValueError(f"run config does not match arm {arm.name}")

    @classmethod
    def from_yaml(cls, path: str | Path) -> H012SuiteConfig:
        with Path(path).open("r", encoding="utf-8") as handle:
            raw = yaml.safe_load(handle)
        values = _mapping(raw, "H012 suite")
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
            raise ValueError(f"unknown H012 suite fields: {sorted(unknown)}")
        raw_arms = _mapping(values["arms"], "H012 arms")
        arms: list[H012ArmSpec] = []
        for name, value in raw_arms.items():
            arm = _mapping(value, f"H012 arm {name}")
            unknown_arm = set(arm) - {
                "config",
                "primary_predictive_baseline",
                "metric_scope",
            }
            if unknown_arm:
                raise ValueError(f"unknown fields for H012 arm {name}: {sorted(unknown_arm)}")
            config_value = arm.get("config")
            arms.append(
                H012ArmSpec(
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
