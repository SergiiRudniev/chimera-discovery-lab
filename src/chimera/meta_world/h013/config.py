"""Immutable run configuration for CHM-W-H013."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, cast

import yaml

from chimera.meta_world.h002.config import H002Arm, H002RunConfig


class H013Arm(str, Enum):
    """Preregistered H013 trainable comparison arms."""

    FACTORIZED = "factorized_counterfactual_transition"
    DIRECT = "matched_direct_factual_no_op_transition"
    FACTUAL_ONLY = "factual_only_relational_reference"


def _mapping(value: object, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{name} must be a mapping")
    return cast(Mapping[str, Any], value)


@dataclass(frozen=True)
class H013RunConfig:
    """Transition semantics plus the matched H002 training runtime."""

    arm: H013Arm
    transition_semantics: str
    no_op_state_weight: float
    intervention_delta_weight: float
    runtime: H002RunConfig

    def __post_init__(self) -> None:
        expected_semantics = {
            H013Arm.FACTORIZED: "no_op_plus_intervention_delta_v1",
            H013Arm.DIRECT: "direct_factual_plus_auxiliary_no_op_v1",
            H013Arm.FACTUAL_ONLY: "direct_factual_only_v1",
        }
        if self.runtime.mode != "preflight":
            raise ValueError("H013 runner is development-preflight only")
        if self.runtime.arm is not H002Arm.NO_ALIGNMENT:
            raise ValueError("H013 trainable arms use cross-world no-alignment data")
        if self.transition_semantics != expected_semantics[self.arm]:
            raise ValueError("H013 arm and transition semantics disagree")
        paired = self.arm in {H013Arm.FACTORIZED, H013Arm.DIRECT}
        expected_weight = 1.0 if paired else 0.0
        if (
            self.no_op_state_weight != expected_weight
            or self.intervention_delta_weight != expected_weight
        ):
            raise ValueError("H013 auxiliary loss weights differ from registration")
        generator = self.runtime.generator_config
        if generator.as_posix() != "configs/meta_world/world_generators_h013.yaml":
            raise ValueError("H013 requires the registered WG4 generator")

    @classmethod
    def from_mapping(cls, values: Mapping[str, Any]) -> H013RunConfig:
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
        }
        if set(values) != expected:
            raise ValueError("H013 run fields must exactly match the frozen schema")
        arm = H013Arm(str(values["arm"]))
        runtime_values = dict(values)
        for name in (
            "transition_semantics",
            "no_op_state_weight",
            "intervention_delta_weight",
        ):
            runtime_values.pop(name)
        runtime_values["arm"] = H002Arm.NO_ALIGNMENT.value
        return cls(
            arm=arm,
            transition_semantics=str(values["transition_semantics"]),
            no_op_state_weight=float(values["no_op_state_weight"]),
            intervention_delta_weight=float(values["intervention_delta_weight"]),
            runtime=H002RunConfig.from_mapping(runtime_values),
        )

    @classmethod
    def from_yaml(cls, path: str | Path) -> H013RunConfig:
        with Path(path).open("r", encoding="utf-8") as handle:
            values = yaml.safe_load(handle)
        return cls.from_mapping(_mapping(values, "H013 run config"))
