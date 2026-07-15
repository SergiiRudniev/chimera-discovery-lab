"""Development runner for CHM-W-H007 gradient stability."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any

from torch import nn

from chimera.meta_world.h002 import H002Trainer
from chimera.meta_world.h005 import H005RunConfig
from chimera.meta_world.h005.preflight import execute_policy_curriculum_run
from chimera.meta_world.h007.config import H007Arm, H007RunConfig
from chimera.meta_world.h007.trainer import H007Trainer


def _git_commit() -> str | None:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        check=False,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip() or None if result.returncode == 0 else None


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _pcgrad_trainer(model: nn.Module, config: H005RunConfig) -> H002Trainer:
    return H007Trainer(
        model,
        config.training,
        rollout_horizon=config.closed_loop.rollout_horizon,
        state_features=config.dataset.worlds.state_features,
        queue_minimum_entries=config.closed_loop.queue_minimum_entries,
        queue_maximum_entries=config.closed_loop.queue_maximum_entries,
    )


def run_h007_preflight(
    config_path: str | Path,
    output_dir: str | Path,
) -> dict[str, Any]:
    """Run one H007 development arm while keeping frozen data sealed."""

    try:
        config_file = Path(config_path)
        config = H007RunConfig.from_yaml(config_file)
        return execute_policy_curriculum_run(
            config_file,
            config.runtime,
            output_dir,
            expected_mode="preflight",
            hypothesis_id="CHM-W-H007",
            reported_arm=config.arm.value,
            effect_supervision="all",
            result_metadata={
                "gradient_intervention": config.gradient_intervention,
                "gradient_task_id_passed_to_model": False,
            },
            trainer_factory=(
                _pcgrad_trainer if config.arm is H007Arm.PCGRAD_MIXED else None
            ),
        )
    except Exception as error:
        output = Path(output_dir)
        output.mkdir(parents=True, exist_ok=True)
        result_path = output / "result.json"
        if not result_path.exists():
            _write_json(
                result_path,
                {
                    "hypothesis_id": "CHM-W-H007",
                    "status": "execution_failed",
                    "decision": "engineering_failure",
                    "frozen_validation_seeds_opened": False,
                    "test_metrics_opened": False,
                    "exception": {
                        "type": type(error).__name__,
                        "message": str(error),
                    },
                    "environment": {"git_commit": _git_commit()},
                    "claim_boundary": (
                        "Execution failure only; no model-quality or transfer evidence."
                    ),
                },
            )
        raise
