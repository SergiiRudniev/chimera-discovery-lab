"""Matched development preflight for CHM-W-H008 outcome heads."""

from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import torch
from torch import nn

from chimera.meta_world.batch import MetaWorldBatch
from chimera.meta_world.generators import SplitName, WorldGenerationPipeline
from chimera.meta_world.h002 import (
    TemporalWorldBaseline,
    materialize_sequence_sample,
)
from chimera.meta_world.h005 import H005Arm, H005RunConfig
from chimera.meta_world.h005.preflight import execute_policy_curriculum_run
from chimera.meta_world.h008.config import H008RunConfig
from chimera.meta_world.h008.evaluation import evaluate_counterfactual_structure
from chimera.meta_world.h008.model import (
    CounterfactualRelationalWorldModel,
    DirectOutcomeRelationalWorldModel,
)
from chimera.meta_world.model import MetaWorldOutput


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


def _counterfactual_model(config: H005RunConfig) -> nn.Module:
    return CounterfactualRelationalWorldModel(config.model)


def _direct_relational_model(config: H005RunConfig) -> nn.Module:
    return DirectOutcomeRelationalWorldModel(config.model)


@dataclass
class _CheckpointRuntime:
    model: nn.Module
    device: torch.device
    use_autocast: bool

    @torch.no_grad()
    def predict(self, batch: MetaWorldBatch) -> MetaWorldOutput:
        self.model.eval()
        with torch.autocast(
            device_type=self.device.type,
            dtype=torch.bfloat16,
            enabled=self.use_autocast,
        ):
            return cast(MetaWorldOutput, self.model(batch.to(self.device)))


def _audit_selected_checkpoint(
    config: H008RunConfig,
    output_dir: Path,
) -> dict[str, float | None]:
    runtime = config.runtime
    device = torch.device("cuda" if runtime.training.device == "cuda" else "cpu")
    if config.is_counterfactual:
        model: nn.Module = CounterfactualRelationalWorldModel(runtime.model)
    elif runtime.arm is H005Arm.TEMPORAL:
        model = TemporalWorldBaseline(runtime.model)
    else:
        model = DirectOutcomeRelationalWorldModel(runtime.model)
    checkpoint = torch.load(
        output_dir / "checkpoint.pt",
        map_location="cpu",
        weights_only=True,
    )
    model.load_state_dict(checkpoint["model"], strict=True)
    model.to(device)
    evaluation_pipeline = WorldGenerationPipeline(
        runtime.dataset.worlds,
        runtime.dataset.policies()[SplitName.VALIDATION],
    )
    sample = materialize_sequence_sample(
        evaluation_pipeline,
        SplitName.VALIDATION,
        start_index=0,
        batch_size=runtime.evaluation.validation_trajectories,
    )
    return evaluate_counterfactual_structure(
        _CheckpointRuntime(
            model=model,
            device=device,
            use_autocast=(
                runtime.training.precision == "bfloat16" and device.type == "cuda"
            ),
        ),
        sample,
        context_steps=runtime.model.context_steps,
    )


def run_h008_preflight(
    config_path: str | Path,
    output_dir: str | Path,
) -> dict[str, Any]:
    """Run one H008 arm and audit its frozen head identity."""

    try:
        config_file = Path(config_path)
        config = H008RunConfig.from_yaml(config_file)
        output = Path(output_dir)
        result = execute_policy_curriculum_run(
            config_file,
            config.runtime,
            output,
            expected_mode="preflight",
            hypothesis_id="CHM-W-H008",
            reported_arm=config.arm.value,
            effect_supervision="all",
            result_metadata={
                "outcome_head": config.outcome_head,
                "derived_no_op_target_passed_to_model": False,
            },
            model_factory=(
                _counterfactual_model
                if config.is_counterfactual
                else (
                    _direct_relational_model
                    if config.runtime.arm is not H005Arm.TEMPORAL
                    else None
                )
            ),
        )
        result["counterfactual_audit"] = _audit_selected_checkpoint(config, output)
        _write_json(output / "result.json", result)
        return result
    except Exception as error:
        output = Path(output_dir)
        output.mkdir(parents=True, exist_ok=True)
        result_path = output / "result.json"
        failure: dict[str, Any] = {}
        if result_path.exists():
            failure = json.loads(result_path.read_text(encoding="utf-8"))
        failure.update(
            {
                "hypothesis_id": "CHM-W-H008",
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
            }
        )
        _write_json(result_path, failure)
        raise
