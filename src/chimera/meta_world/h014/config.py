"""Immutable run configuration for CHM-W-H014."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, cast

import yaml

from chimera.meta_world.h013.config import H013Arm, H013RunConfig


class H014Arm(str, Enum):
    """Parameter-matched H014 effect-head conditions."""

    RESPONSE = "no_op_subtracted_response_conditioned_effect"
    CONTROL = "factual_residual_conditioned_effect_control"


def _mapping(value: object, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{name} must be a mapping")
    return cast(Mapping[str, Any], value)


@dataclass(frozen=True)
class H014RunConfig:
    """Response source plus the shared direct dual-transition runtime."""

    arm: H014Arm
    response_source: str
    paired_runtime: H013RunConfig

    def __post_init__(self) -> None:
        expected_source = {
            H014Arm.RESPONSE: "predicted_factual_minus_predicted_no_op",
            H014Arm.CONTROL: "predicted_factual_minus_final_observation",
        }
        if self.response_source != expected_source[self.arm]:
            raise ValueError("H014 arm and response source disagree")
        if self.paired_runtime.arm is not H013Arm.DIRECT:
            raise ValueError("H014 must retain direct dual-transition dynamics")
        if self.paired_runtime.runtime.training.seed != 260946:
            raise ValueError("H014 development seed is immutable")

    @classmethod
    def from_mapping(cls, values: Mapping[str, Any]) -> H014RunConfig:
        expected = {
            "run_id",
            "mode",
            "arm",
            "generator_config",
            "model",
            "training",
            "evaluation",
            "transition_semantics",
            "no_op_state_weight",
            "intervention_delta_weight",
            "response_source",
        }
        if set(values) != expected:
            raise ValueError("H014 run fields must exactly match the frozen schema")
        arm = H014Arm(str(values["arm"]))
        paired_values = dict(values)
        paired_values.pop("response_source")
        paired_values["arm"] = H013Arm.DIRECT.value
        return cls(
            arm=arm,
            response_source=str(values["response_source"]),
            paired_runtime=H013RunConfig.from_mapping(paired_values),
        )

    @classmethod
    def from_yaml(cls, path: str | Path) -> H014RunConfig:
        with Path(path).open("r", encoding="utf-8") as handle:
            values = yaml.safe_load(handle)
        return cls.from_mapping(_mapping(values, "H014 run config"))
