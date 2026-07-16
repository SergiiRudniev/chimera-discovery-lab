"""Immutable run configuration for CHM-W-H006."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Literal, cast

import yaml

from chimera.meta_world.h005.config import H005Arm, H005RunConfig


class H006Arm(str, Enum):
    """Preregistered policy-routing comparison arms."""

    ROUTED_MIXED = "routed_mixed_closed_loop_without_discrimination"
    SHARED_MIXED = "shared_loss_mixed_closed_loop_without_discrimination"
    RANDOM_ONLY = "matched_random_only_closed_loop_without_discrimination"


def _mapping(value: object, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{name} must be a mapping")
    return cast(Mapping[str, Any], value)


@dataclass(frozen=True)
class H006ObjectiveRouting:
    """Trainer-only loss route; never part of the numerical model batch."""

    state_supervision: Literal["all"]
    effect_supervision: Literal["all", "random_half"]
    route_passed_to_model: bool

    def __post_init__(self) -> None:
        if self.state_supervision != "all":
            raise ValueError("H006 routes state supervision to every trajectory")
        if self.route_passed_to_model:
            raise ValueError("H006 forbids objective routes in model inputs")

    @classmethod
    def from_mapping(cls, values: Mapping[str, Any]) -> H006ObjectiveRouting:
        required = {
            "state_supervision",
            "effect_supervision",
            "route_passed_to_model",
        }
        if set(values) != required:
            raise ValueError("H006 objective routing fields must match preregistration")
        state = str(values["state_supervision"])
        effect = str(values["effect_supervision"])
        if state != "all" or effect not in {"all", "random_half"}:
            raise ValueError("invalid H006 objective route")
        return cls(
            state_supervision=cast(Literal["all"], state),
            effect_supervision=cast(Literal["all", "random_half"], effect),
            route_passed_to_model=bool(values["route_passed_to_model"]),
        )


@dataclass(frozen=True)
class H006RunConfig:
    """H006 research arm plus its matched H005-compatible runtime config."""

    arm: H006Arm
    routing: H006ObjectiveRouting
    runtime: H005RunConfig

    def __post_init__(self) -> None:
        if self.runtime.mode != "preflight":
            raise ValueError("H006 v1 runner is development-preflight only")
        if self.arm is H006Arm.RANDOM_ONLY:
            if self.runtime.arm is not H005Arm.RANDOM_ONLY:
                raise ValueError("H006 random control requires the paired random sampler")
            if self.routing.effect_supervision != "all":
                raise ValueError("H006 random control supervises every random example")
        else:
            if self.runtime.arm is not H005Arm.MIXED:
                raise ValueError("H006 mixed arms require the paired mixed sampler")
            expected = "random_half" if self.arm is H006Arm.ROUTED_MIXED else "all"
            if self.routing.effect_supervision != expected:
                raise ValueError("H006 arm and objective route disagree")

    @classmethod
    def from_mapping(cls, values: Mapping[str, Any]) -> H006RunConfig:
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
            "objective_routing",
        }
        if set(values) != allowed:
            unknown = set(values) - allowed
            missing = allowed - set(values)
            raise ValueError(
                f"H006 run fields differ: unknown={sorted(unknown)}, missing={sorted(missing)}"
            )
        arm = H006Arm(str(values["arm"]))
        runtime_values = dict(values)
        runtime_values.pop("objective_routing")
        runtime_values["arm"] = (
            H005Arm.RANDOM_ONLY.value
            if arm is H006Arm.RANDOM_ONLY
            else H005Arm.MIXED.value
        )
        return cls(
            arm=arm,
            routing=H006ObjectiveRouting.from_mapping(
                _mapping(values["objective_routing"], "objective_routing")
            ),
            runtime=H005RunConfig.from_mapping(runtime_values),
        )

    @classmethod
    def from_yaml(cls, path: str | Path) -> H006RunConfig:
        with Path(path).open("r", encoding="utf-8") as handle:
            values = yaml.safe_load(handle)
        return cls.from_mapping(_mapping(values, "H006 run config"))
