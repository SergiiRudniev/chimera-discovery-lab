"""Development preflight for policy-selective H006 objective routing."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any

from chimera.meta_world.h005.preflight import execute_policy_curriculum_run
from chimera.meta_world.h006.config import H006RunConfig


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


def run_h006_preflight(
    config_path: str | Path,
    output_dir: str | Path,
) -> dict[str, Any]:
    """Run H006 development only and persist any execution failure."""

    try:
        config_file = Path(config_path)
        config = H006RunConfig.from_yaml(config_file)
        return execute_policy_curriculum_run(
            config_file,
            config.runtime,
            output_dir,
            expected_mode="preflight",
            hypothesis_id="CHM-W-H006",
            reported_arm=config.arm.value,
            effect_supervision=config.routing.effect_supervision,
            result_metadata={
                "objective_routing": {
                    "state_supervision": config.routing.state_supervision,
                    "effect_supervision": config.routing.effect_supervision,
                    "route_passed_to_model": config.routing.route_passed_to_model,
                }
            },
        )
    except Exception as error:
        output = Path(output_dir)
        output.mkdir(parents=True, exist_ok=True)
        result_path = output / "result.json"
        if not result_path.exists():
            _write_json(
                result_path,
                {
                    "hypothesis_id": "CHM-W-H006",
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
