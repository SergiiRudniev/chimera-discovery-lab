"""Immutable run configuration for CHM-W-H008."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Literal, cast

import yaml

from chimera.meta_world.h005 import H005Arm, H005RunConfig


class H008Arm(str, Enum):
    """Preregistered outcome-head comparison arms."""

    COUNTERFACTUAL_MIXED = "counterfactual_head_mixed_closed_loop"
    DIRECT_MIXED = "direct_head_mixed_closed_loop"
    COUNTERFACTUAL_RANDOM = "counterfactual_head_random_closed_loop"
    DIRECT_RANDOM = "direct_head_random_closed_loop"
    ONE_STEP = "one_step_relational_without_discrimination"
    TEMPORAL = "temporal_predictor_without_relational_state"


def _mapping(value: object, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{name} must be a mapping")
    return cast(Mapping[str, Any], value)


@dataclass(frozen=True)
class H008RunConfig:
    """Research outcome-head semantics plus a matched training runtime."""

    arm: H008Arm
    outcome_head: Literal["counterfactual_difference_v1", "direct_effect"]
    runtime: H005RunConfig

    @property
    def is_counterfactual(self) -> bool:
        return self.outcome_head == "counterfactual_difference_v1"

    def __post_init__(self) -> None:
        if self.runtime.mode != "preflight":
            raise ValueError("H008 v1 runner is development-preflight only")
        sampler_by_arm = {
            H008Arm.COUNTERFACTUAL_MIXED: H005Arm.MIXED,
            H008Arm.DIRECT_MIXED: H005Arm.MIXED,
            H008Arm.COUNTERFACTUAL_RANDOM: H005Arm.RANDOM_ONLY,
            H008Arm.DIRECT_RANDOM: H005Arm.RANDOM_ONLY,
            H008Arm.ONE_STEP: H005Arm.ONE_STEP,
            H008Arm.TEMPORAL: H005Arm.TEMPORAL,
        }
        expected_sampler = sampler_by_arm[self.arm]
        if self.runtime.arm is not expected_sampler:
            raise ValueError("H008 arm and sampler disagree")
        counterfactual = self.arm in {
            H008Arm.COUNTERFACTUAL_MIXED,
            H008Arm.COUNTERFACTUAL_RANDOM,
        }
        if self.is_counterfactual != counterfactual:
            raise ValueError("H008 arm and outcome-head semantics disagree")

    @classmethod
    def from_mapping(cls, values: Mapping[str, Any]) -> H008RunConfig:
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
            "outcome_head",
        }
        if set(values) != allowed:
            raise ValueError("H008 run fields must exactly match the frozen schema")
        arm = H008Arm(str(values["arm"]))
        head = str(values["outcome_head"])
        if head not in {"counterfactual_difference_v1", "direct_effect"}:
            raise ValueError("unknown H008 outcome head")
        runtime_values = dict(values)
        runtime_values.pop("outcome_head")
        runtime_values["arm"] = {
            H008Arm.COUNTERFACTUAL_MIXED: H005Arm.MIXED.value,
            H008Arm.DIRECT_MIXED: H005Arm.MIXED.value,
            H008Arm.COUNTERFACTUAL_RANDOM: H005Arm.RANDOM_ONLY.value,
            H008Arm.DIRECT_RANDOM: H005Arm.RANDOM_ONLY.value,
            H008Arm.ONE_STEP: H005Arm.ONE_STEP.value,
            H008Arm.TEMPORAL: H005Arm.TEMPORAL.value,
        }[arm]
        return cls(
            arm=arm,
            outcome_head=cast(
                Literal["counterfactual_difference_v1", "direct_effect"],
                head,
            ),
            runtime=H005RunConfig.from_mapping(runtime_values),
        )

    @classmethod
    def from_yaml(cls, path: str | Path) -> H008RunConfig:
        with Path(path).open("r", encoding="utf-8") as handle:
            values = yaml.safe_load(handle)
        return cls.from_mapping(_mapping(values, "H008 run config"))
