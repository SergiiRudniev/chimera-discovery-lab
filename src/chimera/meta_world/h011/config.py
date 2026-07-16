"""Validated compact configuration for H011 response consistency."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, cast

import yaml

from chimera.meta_world.h002.config import H002Arm, H002RunConfig
from chimera.meta_world.h010.config import H010ModelVariant, H010RunConfig


@dataclass(frozen=True)
class H011RunConfig:
    run_id: str
    base_config: Path
    training_seed: int
    response_consistency_weight: float
    uncertainty_consistency_fraction: float
    common: H002RunConfig

    def __post_init__(self) -> None:
        if not self.run_id or self.training_seed < 0:
            raise ValueError("H011 run ID and seed must be valid")
        if self.response_consistency_weight < 0.0:
            raise ValueError("response consistency weight must be non-negative")
        if not 0.0 <= self.uncertainty_consistency_fraction <= 1.0:
            raise ValueError("uncertainty consistency fraction must be in [0, 1]")
        if self.common.arm is not H002Arm.NO_ALIGNMENT:
            raise ValueError("H011 disables global mechanism alignment")

    @classmethod
    def from_mapping(cls, values: Mapping[str, Any]) -> H011RunConfig:
        allowed = {
            "run_id",
            "base_config",
            "training_seed",
            "response_consistency_weight",
            "uncertainty_consistency_fraction",
        }
        unknown = set(values) - allowed
        if unknown:
            raise ValueError(f"unknown H011 fields: {sorted(unknown)}")
        base_path = Path(str(values["base_config"]))
        base = H010RunConfig.from_yaml(base_path)
        if base.model_variant is not H010ModelVariant.SEPARATE:
            raise ValueError("H011 requires the separate-projection control model")
        run_id = str(values["run_id"])
        seed = int(values["training_seed"])
        common = replace(
            base.common,
            run_id=run_id,
            training=replace(base.common.training, seed=seed),
        )
        return cls(
            run_id=run_id,
            base_config=base_path,
            training_seed=seed,
            response_consistency_weight=float(values["response_consistency_weight"]),
            uncertainty_consistency_fraction=float(
                values["uncertainty_consistency_fraction"]
            ),
            common=common,
        )

    @classmethod
    def from_yaml(cls, path: str | Path) -> H011RunConfig:
        with Path(path).open("r", encoding="utf-8") as handle:
            values = yaml.safe_load(handle)
        if not isinstance(values, Mapping):
            raise TypeError("H011 config must be a mapping")
        return cls.from_mapping(cast(Mapping[str, Any], values))
