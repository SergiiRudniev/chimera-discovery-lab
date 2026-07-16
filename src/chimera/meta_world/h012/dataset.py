"""Fixed deterministic engineering dataset for H012."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from chimera.meta_world.generators import (
    GeneratedWorldDatasetConfig,
    build_generated_world_dataset,
    validate_generated_world_dataset,
)


def build_h012_smoke_dataset(
    output_dir: str | Path,
    config_path: str | Path,
    *,
    trajectories_per_split: int = 16,
) -> dict[str, Any]:
    """Build and validate WG3 without creating a scientific result."""

    output = Path(output_dir)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("H012 smoke output directory must be empty")
    config = GeneratedWorldDatasetConfig.from_yaml(config_path)
    if config.hypothesis_id != "CHM-W-H012" or config.dataset_id != "CHM-W-WG3":
        raise ValueError("H012 smoke requires the registered WG3 generator")
    build_generated_world_dataset(
        output,
        config_path,
        trajectories_per_split=trajectories_per_split,
        claim_boundary=(
            "H012 generated-world engineering smoke only; no transfer, causal, "
            "business-utility, language-independence or production claim."
        ),
    )
    report = validate_generated_world_dataset(output / "manifest.json")
    if report["status"] != "passed":
        raise RuntimeError("H012 smoke dataset failed integrity validation")
    return report
