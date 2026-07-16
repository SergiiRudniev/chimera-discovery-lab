"""Validation-only H009 runner; every test split remains sealed."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from chimera.meta_world.generators import (
    GeneratedWorldDatasetConfig,
    ViewCoupling,
)
from chimera.meta_world.h002.config import H002RunConfig
from chimera.meta_world.h002.preflight import run_generated_world_preflight


def run_h009_preflight(
    config_path: str | Path,
    output_dir: str | Path,
) -> dict[str, Any]:
    """Run a train/validation H009 preflight after verifying paired-view data."""

    run_config = H002RunConfig.from_yaml(config_path)
    generator = GeneratedWorldDatasetConfig.from_yaml(run_config.generator_config)
    if generator.hypothesis_id != "CHM-W-H009":
        raise ValueError("H009 preflight requires the registered H009 generator")
    if generator.view_coupling is not ViewCoupling.PAIRED_WORLD_RENDERERS:
        raise ValueError("H009 preflight requires paired renderer views")
    return run_generated_world_preflight(
        config_path,
        output_dir,
        hypothesis_id="CHM-W-H009",
    )
