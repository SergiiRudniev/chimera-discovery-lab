"""Validated run configuration for H010 mechanism-path controls."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, cast

import yaml

from chimera.meta_world.h002.config import H002Arm, H002RunConfig


class H010ModelVariant(str, Enum):
    SHARED = "shared_aligned_bottleneck"
    SEPARATE = "separate_alignment_projection"


@dataclass(frozen=True)
class H010RunConfig:
    """H010 model-path selection plus the unchanged generated-world run contract."""

    model_variant: H010ModelVariant
    common: H002RunConfig

    def __post_init__(self) -> None:
        if self.common.arm not in {H002Arm.ALIGNED, H002Arm.NO_ALIGNMENT}:
            raise ValueError("H010 only compares relational mechanism-path arms")

    @classmethod
    def from_mapping(cls, values: Mapping[str, Any]) -> H010RunConfig:
        mutable = dict(values)
        if "model_variant" not in mutable:
            raise ValueError("model_variant is required for H010")
        variant = H010ModelVariant(str(mutable.pop("model_variant")))
        return cls(
            model_variant=variant,
            common=H002RunConfig.from_mapping(cast(Mapping[str, Any], mutable)),
        )

    @classmethod
    def from_yaml(cls, path: str | Path) -> H010RunConfig:
        with Path(path).open("r", encoding="utf-8") as handle:
            values = yaml.safe_load(handle)
        if not isinstance(values, Mapping):
            raise TypeError("H010 run config must be a mapping")
        return cls.from_mapping(cast(Mapping[str, Any], values))

    def to_dict(self) -> dict[str, object]:
        return {
            **self.common.to_dict(),
            "model_variant": self.model_variant.value,
        }
