"""Validation-only runner for the matched H014 effect heads."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any

from chimera.meta_world.h013.config import H013RunConfig
from chimera.meta_world.h013.preflight import execute_paired_transition_preflight
from chimera.meta_world.h014.config import H014RunConfig
from chimera.meta_world.h014.model import (
    ResponseConditionedEffectWorldModel,
    ResponseSource,
)


def _git_commit() -> str | None:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        check=False,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip() or None if result.returncode == 0 else None


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_bytes(
        (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode("utf-8")
    )


def run_h014_preflight(
    config_path: str | Path,
    output_dir: str | Path,
) -> dict[str, Any]:
    """Run one H014 arm while keeping frozen validation and test sealed."""

    try:
        config = H014RunConfig.from_yaml(config_path)

        def model_factory(runtime: H013RunConfig) -> ResponseConditionedEffectWorldModel:
            return ResponseConditionedEffectWorldModel(
                runtime.runtime.model,
                response_source=ResponseSource(config.response_source),
            )

        return execute_paired_transition_preflight(
            config_path,
            output_dir,
            run_config=config.paired_runtime,
            hypothesis_id="CHM-W-H014",
            reported_arm=config.arm.value,
            model_factory=model_factory,
            selection_metrics=("intervention_effect_nrmse",),
            result_metadata={"response_source": config.response_source},
        )
    except Exception as error:
        output = Path(output_dir)
        output.mkdir(parents=True, exist_ok=True)
        result_path = output / "result.json"
        if not result_path.exists():
            _write_json(
                result_path,
                {
                    "hypothesis_id": "CHM-W-H014",
                    "status": "execution_failed",
                    "decision": "engineering_failure",
                    "frozen_validation_seeds_opened": False,
                    "test_metrics_opened": False,
                    "exception": {
                        "type": type(error).__name__,
                        "message": str(error),
                    },
                    "environment": {"git_commit": _git_commit()},
                },
            )
        raise
