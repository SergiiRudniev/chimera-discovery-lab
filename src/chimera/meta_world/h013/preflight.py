"""Validation-only H013 training runner with sealed test partitions."""

from __future__ import annotations

import hashlib
import json
import platform
import subprocess
import time
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

import torch
from torch import nn

from chimera.meta_world.generators import (
    GeneratedWorldDatasetConfig,
    SplitName,
    WorldGenerationPipeline,
)
from chimera.meta_world.h002.windows import (
    make_transition_window,
    materialize_sequence_sample,
)
from chimera.meta_world.h008.model import CounterfactualRelationalWorldModel
from chimera.meta_world.h013.config import H013Arm, H013RunConfig
from chimera.meta_world.h013.evaluation import evaluate_h013_model
from chimera.meta_world.h013.model import (
    DirectDualTransitionWorldModel,
    FactorizedCounterfactualTransitionWorldModel,
)
from chimera.meta_world.h013.trainer import H013Trainer


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


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


def _make_model(config: H013RunConfig) -> nn.Module:
    model_config = config.runtime.model
    if config.arm is H013Arm.FACTORIZED:
        return FactorizedCounterfactualTransitionWorldModel(model_config)
    if config.arm is H013Arm.DIRECT:
        return DirectDualTransitionWorldModel(model_config)
    return CounterfactualRelationalWorldModel(model_config)


def _selection_score(
    metrics: dict[str, Any],
    names: tuple[str, ...],
) -> float:
    values = [metrics[name] for name in names]
    if not values or any(value is None for value in values):
        raise RuntimeError("paired-transition selection metric is unavailable")
    return sum(float(value) for value in values) / len(values)


def execute_paired_transition_preflight(
    config_path: str | Path,
    output_dir: str | Path,
    *,
    run_config: H013RunConfig | None = None,
    hypothesis_id: str = "CHM-W-H013",
    reported_arm: str | None = None,
    model_factory: Callable[[H013RunConfig], nn.Module] | None = None,
    selection_metrics: tuple[str, ...] = (
        "intervention_state_delta_nrmse",
        "four_step_rollout_nrmse",
    ),
    result_metadata: Mapping[str, object] | None = None,
) -> dict[str, Any]:
    config_file = Path(config_path)
    config = run_config or H013RunConfig.from_yaml(config_file)
    runtime = config.runtime
    arm = reported_arm or config.arm.value
    metadata = dict(result_metadata or {})
    generator = GeneratedWorldDatasetConfig.from_yaml(runtime.generator_config)
    if (
        generator.hypothesis_id != "CHM-W-H013"
        or generator.dataset_id != "CHM-W-WG4"
        or not generator.paired_counterfactual_transitions
    ):
        raise ValueError("paired-transition preflight requires WG4")
    output = Path(output_dir)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("paired-transition preflight output must be empty")
    output.mkdir(parents=True, exist_ok=True)
    validation_pipeline = WorldGenerationPipeline(generator)
    training_pipeline = WorldGenerationPipeline(generator)
    validation_sample = materialize_sequence_sample(
        validation_pipeline,
        SplitName.VALIDATION,
        start_index=0,
        batch_size=runtime.evaluation.validation_trajectories,
    )
    torch.manual_seed(runtime.training.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(runtime.training.seed)
    model = _make_model(config) if model_factory is None else model_factory(config)
    model_class = f"{type(model).__module__}.{type(model).__qualname__}"
    trainer = H013Trainer(
        model,
        runtime.training,
        no_op_state_weight=config.no_op_state_weight,
        intervention_delta_weight=config.intervention_delta_weight,
    )
    started = time.perf_counter()
    initial = evaluate_h013_model(
        trainer,
        validation_sample,
        context_steps=runtime.model.context_steps,
        rollout_horizon=runtime.evaluation.rollout_horizon,
    )
    best_metrics = initial
    best_score = _selection_score(initial, selection_metrics)
    best_step = 0
    checkpoint_path = output / "checkpoint.pt"

    def save_checkpoint(step: int) -> None:
        torch.save(
            {
                "run_id": runtime.run_id,
                "arm": arm,
                "transition_semantics": config.transition_semantics,
                "model_class": model_class,
                "step": step,
                "model": trainer.evaluation_state_dict(),
                "weights_kind": trainer.evaluation_weights_kind,
                "model_config": runtime.to_dict()["model"],
                **metadata,
            },
            checkpoint_path,
        )

    save_checkpoint(0)
    metric_rows: list[dict[str, Any]] = [
        {"phase": "validation", "step": 0, "selection_score": best_score, **initial}
    ]
    first_training: dict[str, float] | None = None
    final_training: dict[str, float] | None = None
    prediction_count = generator.trajectory_steps - 1
    for step in range(1, runtime.training.steps + 1):
        train_sample = materialize_sequence_sample(
            training_pipeline,
            SplitName.TRAIN,
            start_index=(step - 1) * runtime.training.batch_size,
            batch_size=runtime.training.batch_size,
        )
        window = make_transition_window(
            train_sample,
            prediction_step=(step - 1) % prediction_count,
            context_steps=runtime.model.context_steps,
        )
        training_metrics = trainer.train_step(window)
        if first_training is None:
            first_training = training_metrics
        final_training = training_metrics
        metric_rows.append({"phase": "train", "step": step, **training_metrics})
        if (
            step % runtime.evaluation.evaluation_interval == 0
            or step == runtime.training.steps
        ):
            evaluation = evaluate_h013_model(
                trainer,
                validation_sample,
                context_steps=runtime.model.context_steps,
                rollout_horizon=runtime.evaluation.rollout_horizon,
            )
            score = _selection_score(evaluation, selection_metrics)
            metric_rows.append(
                {
                    "phase": "validation",
                    "step": step,
                    "selection_score": score,
                    **evaluation,
                }
            )
            if score < best_score:
                best_score = score
                best_step = step
                best_metrics = evaluation
                save_checkpoint(step)
    metrics_path = output / "metrics.jsonl"
    metrics_path.write_bytes(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in metric_rows).encode(
            "utf-8"
        )
    )
    manifest = {
        "run_id": runtime.run_id,
        "arm": arm,
        "transition_semantics": config.transition_semantics,
        "weights_kind": trainer.evaluation_weights_kind,
        "checkpoint": checkpoint_path.name,
        "checkpoint_sha256": _sha256(checkpoint_path),
        "selected_step": best_step,
        "selection_metric": "mean(" + ", ".join(
            f"validation {name}" for name in selection_metrics
        ) + ")",
        "selection_score": best_score,
        "promoted": False,
        "scope": "development-only validation preflight",
        "opened_splits": ["train", "validation"],
        **metadata,
    }
    _write_json(output / "checkpoint_manifest.json", manifest)
    result: dict[str, Any] = {
        "run_id": runtime.run_id,
        "hypothesis_id": hypothesis_id,
        "status": "completed_preflight",
        "decision": "engineering_only",
        "arm": arm,
        "transition_semantics": config.transition_semantics,
        "model_class": model_class,
        "weights_kind": trainer.evaluation_weights_kind,
        "parameters": sum(
            parameter.numel() for parameter in model.parameters() if parameter.requires_grad
        ),
        "opened_splits": ["train", "validation"],
        "frozen_validation_seeds_opened": False,
        "test_metrics_opened": False,
        "initial_validation": initial,
        "best_validation": best_metrics,
        "best_step": best_step,
        "best_selection_score": best_score,
        "first_training": first_training,
        "final_training": final_training,
        "runtime_seconds": time.perf_counter() - started,
        "peak_memory_bytes": trainer.peak_memory_bytes(),
        "environment": {
            "git_commit": _git_commit(),
            "config_sha256": _sha256(config_file),
            "generator_config_sha256": _sha256(runtime.generator_config),
            "python": platform.python_version(),
            "torch": torch.__version__,
            "device": str(trainer.device),
            "device_name": (
                torch.cuda.get_device_name(trainer.device)
                if trainer.device.type == "cuda"
                else None
            ),
            "precision": runtime.training.precision,
        },
        "artifacts": {
            "metrics": metrics_path.name,
            "checkpoint_manifest": "checkpoint_manifest.json",
        },
        "claim_boundary": (
            "Development-only simulator evidence; frozen validation and test "
            "remain sealed and no checkpoint is promoted."
        ),
        **metadata,
    }
    _write_json(output / "result.json", result)
    return result


def run_h013_preflight(
    config_path: str | Path,
    output_dir: str | Path,
) -> dict[str, Any]:
    """Persist an auditable failure record if H013 execution stops."""

    try:
        return execute_paired_transition_preflight(config_path, output_dir)
    except Exception as error:
        output = Path(output_dir)
        output.mkdir(parents=True, exist_ok=True)
        result_path = output / "result.json"
        if not result_path.exists():
            _write_json(
                result_path,
                {
                    "hypothesis_id": "CHM-W-H013",
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
