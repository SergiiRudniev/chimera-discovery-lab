"""Fixed WG4 smoke dataset and integrity evidence for H013."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from chimera.meta_world.generators import (
    GeneratedWorldDatasetConfig,
    build_generated_world_dataset,
    validate_generated_world_dataset,
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build_h013_smoke_dataset(
    config_path: str | Path,
    output_dir: str | Path,
    report_path: str | Path,
    *,
    trajectories_per_split: int = 16,
) -> dict[str, Any]:
    """Build and validate WG4 once without opening any model test metric."""

    config_file = Path(config_path)
    config = GeneratedWorldDatasetConfig.from_yaml(config_file)
    if (
        config.hypothesis_id != "CHM-W-H013"
        or config.dataset_id != "CHM-W-WG4"
        or config.schema_version != 3
        or not config.paired_counterfactual_transitions
    ):
        raise ValueError("H013 smoke requires the registered WG4 schema")
    output = Path(output_dir)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("H013 smoke output directory must be empty")
    build_generated_world_dataset(
        output,
        config_file,
        trajectories_per_split=trajectories_per_split,
        claim_boundary=(
            "H013 WG4 engineering integrity evidence only; no model-quality, "
            "transfer, causal, business-utility or production claim."
        ),
    )
    validation = validate_generated_world_dataset(output / "manifest.json")
    validation.update(
        {
            "schema_version": 1,
            "preflight_id": "CHM-W-H013-WG4-INTEGRITY-001",
            "dataset_config": config_file.as_posix(),
            "dataset_config_sha256": _sha256(config_file),
            "test_metrics_opened": False,
            "human_or_llm_judging": False,
        }
    )
    if validation["status"] != "passed":
        raise RuntimeError("WG4 integrity validation failed")
    report = Path(report_path)
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_bytes(
        (json.dumps(validation, indent=2, sort_keys=True) + "\n").encode("utf-8")
    )
    return validation
