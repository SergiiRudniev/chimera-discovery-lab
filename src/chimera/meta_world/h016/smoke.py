"""Postcommit GPU engineering smoke for the complete H016 training path."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from chimera.meta_world.h016.config import H016BackboneConfig, H016SuiteConfig
from chimera.meta_world.h016.preflight import run_h016_backbone_preflight
from chimera.meta_world.h016.run import run_h016_ranking_training
from chimera.meta_world.trainer import resolve_device


def run_h016_engineering_smoke(
    backbone_config_path: str | Path,
    suite_config_path: str | Path,
    output_dir: str | Path,
) -> dict[str, Any]:
    """Run two backbone and two rank-head steps without opening new data splits."""

    output = Path(output_dir)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("H016 smoke output must be empty")
    output.mkdir(parents=True, exist_ok=True)
    backbone_result = run_h016_backbone_preflight(
        backbone_config_path,
        output / "backbone",
    )
    backbone_config = H016BackboneConfig.from_yaml(backbone_config_path)
    suite = H016SuiteConfig.from_yaml(suite_config_path)
    runtime = backbone_config.paired_runtime.runtime
    device = resolve_device(runtime.training.device)
    _, ranking_result = run_h016_ranking_training(
        suite,
        backbone_checkpoint=output / "backbone" / "checkpoint.pt",
        output_dir=output / "ranking",
        device=device,
        use_autocast=(
            runtime.training.precision == "bfloat16" and device.type == "cuda"
        ),
        steps=2,
    )
    result = {
        "status": "completed_engineering_smoke",
        "hypothesis_id": "CHM-W-H016",
        "backbone_steps": backbone_result["best_step"],
        "backbone_parameters": backbone_result["parameters"],
        "ranking_steps": ranking_result["steps"],
        "ranking_trainable_parameters": ranking_result["trainable_parameters"],
        "ranking_total_parameters": ranking_result["total_parameters"],
        "ranking_checkpoint_sha256": ranking_result["checkpoint"]["sha256"],
        "deterministic_training_candidate_replay_rate": ranking_result[
            "deterministic_training_candidate_replay_rate"
        ],
        "backbone_unchanged": ranking_result["backbone_unchanged"],
        "final_training": ranking_result["final_training"],
        "peak_memory_bytes": max(
            int(backbone_result["peak_memory_bytes"]),
            int(ranking_result["peak_memory_bytes"]),
        ),
        "opened_splits": ["train", "validation"],
        "frozen_validation_seeds_opened": False,
        "test_metrics_opened": False,
        "checkpoint_promoted": False,
    }
    (output / "smoke_result.json").write_bytes(
        (json.dumps(result, indent=2, sort_keys=True) + "\n").encode("utf-8")
    )
    return result
