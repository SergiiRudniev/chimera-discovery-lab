"""Immutable run configuration for CHM-W-H007."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Literal, cast

import yaml

from chimera.meta_world.h005 import H005Arm, H005RunConfig


class H007Arm(str, Enum):
    """Preregistered gradient-stability comparison arms."""

    PCGRAD_MIXED = "pcgrad_mixed_closed_loop_without_discrimination"
    STANDARD_MIXED = "standard_mixed_closed_loop_without_discrimination"
    RANDOM_ONLY = "matched_random_only_closed_loop_without_discrimination"


def _mapping(value: object, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{name} must be a mapping")
    return cast(Mapping[str, Any], value)


@dataclass(frozen=True)
class H007RunConfig:
    """Research arm plus an H005-compatible matched runtime."""

    arm: H007Arm
    gradient_intervention: Literal["symmetric_global_pcgrad_v1", "none"]
    runtime: H005RunConfig

    def __post_init__(self) -> None:
        if self.runtime.mode != "preflight":
            raise ValueError("H007 v1 runner is development-preflight only")
        if self.runtime.training.alignment_weight != 0.0:
            raise ValueError("H007 keeps mechanism discrimination disabled")
        expected_sampler = (
            H005Arm.RANDOM_ONLY
            if self.arm is H007Arm.RANDOM_ONLY
            else H005Arm.MIXED
        )
        if self.runtime.arm is not expected_sampler:
            raise ValueError("H007 research arm and sampler disagree")
        expected_intervention = (
            "symmetric_global_pcgrad_v1"
            if self.arm is H007Arm.PCGRAD_MIXED
            else "none"
        )
        if self.gradient_intervention != expected_intervention:
            raise ValueError("H007 arm and gradient intervention disagree")

    @classmethod
    def from_mapping(cls, values: Mapping[str, Any]) -> H007RunConfig:
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
            "gradient_intervention",
        }
        if set(values) != allowed:
            raise ValueError("H007 run fields must exactly match the frozen schema")
        arm = H007Arm(str(values["arm"]))
        intervention = str(values["gradient_intervention"])
        if intervention not in {"symmetric_global_pcgrad_v1", "none"}:
            raise ValueError("unknown H007 gradient intervention")
        runtime_values = dict(values)
        runtime_values.pop("gradient_intervention")
        runtime_values["arm"] = (
            H005Arm.RANDOM_ONLY.value
            if arm is H007Arm.RANDOM_ONLY
            else H005Arm.MIXED.value
        )
        return cls(
            arm=arm,
            gradient_intervention=cast(
                Literal["symmetric_global_pcgrad_v1", "none"],
                intervention,
            ),
            runtime=H005RunConfig.from_mapping(runtime_values),
        )

    @classmethod
    def from_yaml(cls, path: str | Path) -> H007RunConfig:
        with Path(path).open("r", encoding="utf-8") as handle:
            values = yaml.safe_load(handle)
        return cls.from_mapping(_mapping(values, "H007 run config"))
