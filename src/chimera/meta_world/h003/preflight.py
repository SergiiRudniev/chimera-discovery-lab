"""Validation-only H003 preflight; registered test splits are never materialized."""

from __future__ import annotations

import hashlib
import json
import platform
import subprocess
import time
from pathlib import Path
from typing import Any

import torch
from torch import nn

from chimera.meta_world.generators import (
    GeneratedWorldDatasetConfig,
    SplitName,
    WorldGenerationPipeline,
)
from chimera.meta_world.h002.evaluation import evaluate_h002_model
from chimera.meta_world.h002.model import (
    RelationalSequenceWorldModel,
    TemporalWorldBaseline,
)
from chimera.meta_world.h002.trainer import H002Trainer
from chimera.meta_world.h002.windows import (
    make_transition_window,
    materialize_sequence_sample,
)
from chimera.meta_world.h003.config import H003Arm, H003RunConfig
from chimera.meta_world.h003.trainer import H003Trainer


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
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _model(config: H003RunConfig) -> nn.Module:
    if config.arm is H003Arm.TEMPORAL:
        return TemporalWorldBaseline(config.model)
    if config.arm in {
        H003Arm.CLOSED_LOOP_ALIGNED,
        H003Arm.CLOSED_LOOP_NO_ALIGNMENT,
        H003Arm.H002_ONE_STEP,
    }:
        return RelationalSequenceWorldModel(config.model)
    raise ValueError("legal random intervention has no trainable model")


def _trainer(
    model: nn.Module,
    config: H003RunConfig,
    generator: GeneratedWorldDatasetConfig,
) -> H002Trainer:
    if config.arm in {
        H003Arm.CLOSED_LOOP_ALIGNED,
        H003Arm.CLOSED_LOOP_NO_ALIGNMENT,
    }:
        return H003Trainer(
            model,
            config.training,
            rollout_horizon=config.closed_loop.rollout_horizon,
            state_features=generator.state_features,
            queue_minimum_entries=config.closed_loop.queue_minimum_entries,
            queue_maximum_entries=config.closed_loop.queue_maximum_entries,
        )
    return H002Trainer(model, config.training)


def _parameter_count(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)


def _selection_score(metrics: dict[str, float]) -> float:
    return 0.5 * (
        metrics["intervention_effect_nrmse"]
        + metrics["four_step_rollout_nrmse"]
    )


def _execute_h003_preflight(
    config_path: str | Path,
    output_dir: str | Path,
) -> dict[str, Any]:
    config_file = Path(config_path)
    config = H003RunConfig.from_yaml(config_file)
    if config.mode != "preflight":
        raise ValueError("run_h003_preflight only accepts mode=preflight")
    output = Path(output_dir)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("preflight output directory must be empty")
    output.mkdir(parents=True, exist_ok=True)
    generator_config = GeneratedWorldDatasetConfig.from_yaml(config.generator_config)
    pipeline = WorldGenerationPipeline(generator_config)
    validation_sample = materialize_sequence_sample(
        pipeline,
        SplitName.VALIDATION,
        start_index=0,
        batch_size=config.evaluation.validation_trajectories,
    )
    model = _model(config)
    model_class = f"{type(model).__module__}.{type(model).__qualname__}"
    trainer = _trainer(model, config, generator_config)
    started = time.perf_counter()
    initial_evaluation = evaluate_h002_model(
        trainer,
        validation_sample,
        context_steps=config.model.context_steps,
        rollout_horizon=config.evaluation.rollout_horizon,
    ).to_dict()
    best_metrics = initial_evaluation
    best_score = _selection_score(best_metrics)
    best_step = 0
    checkpoint_path = output / "checkpoint.pt"

    def save_checkpoint(step: int) -> None:
        torch.save(
            {
                "run_id": config.run_id,
                "hypothesis_id": "CHM-W-H003",
                "arm": config.arm.value,
                "model_class": model_class,
                "step": step,
                "model": trainer.evaluation_state_dict(),
                "weights_kind": trainer.evaluation_weights_kind,
                "model_config": config.to_dict()["model"],
            },
            checkpoint_path,
        )

    save_checkpoint(best_step)
    metric_rows: list[dict[str, Any]] = [
        {
            "phase": "validation",
            "step": 0,
            "selection_score": best_score,
            **initial_evaluation,
        }
    ]
    first_training: dict[str, float] | None = None
    final_training: dict[str, float] | None = None
    prediction_count = generator_config.trajectory_steps - 1
    rollout_start = config.model.context_steps - 1
    for step in range(1, config.training.steps + 1):
        train_sample = materialize_sequence_sample(
            pipeline,
            SplitName.TRAIN,
            start_index=(step - 1) * config.training.batch_size,
            batch_size=config.training.batch_size,
        )
        if isinstance(trainer, H003Trainer):
            training_metrics = trainer.train_sequence_step(
                train_sample,
                prediction_step=rollout_start,
                context_steps=config.model.context_steps,
            )
        else:
            window = make_transition_window(
                train_sample,
                prediction_step=(step - 1) % prediction_count,
                context_steps=config.model.context_steps,
            )
            training_metrics = trainer.train_step(window)
        if first_training is None:
            first_training = training_metrics
        final_training = training_metrics
        metric_rows.append({"phase": "train", "step": step, **training_metrics})
        if (
            step % config.evaluation.evaluation_interval == 0
            or step == config.training.steps
        ):
            evaluation = evaluate_h002_model(
                trainer,
                validation_sample,
                context_steps=config.model.context_steps,
                rollout_horizon=config.evaluation.rollout_horizon,
            ).to_dict()
            score = _selection_score(evaluation)
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
                save_checkpoint(best_step)
    metrics_path = output / "metrics.jsonl"
    metrics_path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in metric_rows),
        encoding="utf-8",
    )
    queue_entries = trainer.queue.size if isinstance(trainer, H003Trainer) else 0
    checkpoint_manifest = {
        "run_id": config.run_id,
        "hypothesis_id": "CHM-W-H003",
        "arm": config.arm.value,
        "model_class": model_class,
        "weights_kind": trainer.evaluation_weights_kind,
        "checkpoint": checkpoint_path.name,
        "checkpoint_sha256": _sha256(checkpoint_path),
        "selected_step": best_step,
        "selection_metric": (
            "mean(validation intervention_effect_nrmse, "
            "validation four_step_rollout_nrmse)"
        ),
        "selection_score": best_score,
        "promoted": False,
        "scope": "validation-only H003 engineering preflight",
        "opened_splits": [SplitName.TRAIN.value, SplitName.VALIDATION.value],
    }
    _write_json(output / "checkpoint_manifest.json", checkpoint_manifest)
    result: dict[str, Any] = {
        "run_id": config.run_id,
        "hypothesis_id": "CHM-W-H003",
        "status": "completed_validation_preflight",
        "decision": "engineering_only",
        "arm": config.arm.value,
        "model_class": model_class,
        "weights_kind": trainer.evaluation_weights_kind,
        "parameters": _parameter_count(model),
        "opened_splits": [SplitName.TRAIN.value, SplitName.VALIDATION.value],
        "test_metrics_opened": False,
        "initial_validation": initial_evaluation,
        "best_validation": best_metrics,
        "best_step": best_step,
        "best_selection_score": best_score,
        "first_training": first_training,
        "final_training": final_training,
        "mechanism_queue_entries": queue_entries,
        "runtime_seconds": time.perf_counter() - started,
        "peak_memory_bytes": trainer.peak_memory_bytes(),
        "environment": {
            "git_commit": _git_commit(),
            "config_sha256": _sha256(config_file),
            "generator_config_sha256": _sha256(config.generator_config),
            "python": platform.python_version(),
            "torch": torch.__version__,
            "device": str(trainer.device),
            "device_name": (
                torch.cuda.get_device_name(trainer.device)
                if trainer.device.type == "cuda"
                else None
            ),
            "precision": config.training.precision,
        },
        "artifacts": {
            "metrics": metrics_path.name,
            "checkpoint_manifest": "checkpoint_manifest.json",
        },
        "claim_boundary": (
            "Validation-only simulator engineering evidence; no test transfer, causal "
            "discovery, business utility or production checkpoint claim."
        ),
    }
    _write_json(output / "result.json", result)
    return result


def run_h003_preflight(
    config_path: str | Path,
    output_dir: str | Path,
) -> dict[str, Any]:
    """Run H003 preflight and persist unexpected failures before re-raising."""

    try:
        return _execute_h003_preflight(config_path, output_dir)
    except Exception as error:
        output = Path(output_dir)
        output.mkdir(parents=True, exist_ok=True)
        result_path = output / "result.json"
        if not result_path.exists():
            run_id: str | None = None
            try:
                run_id = H003RunConfig.from_yaml(config_path).run_id
            except Exception:
                run_id = None
            _write_json(
                result_path,
                {
                    "run_id": run_id,
                    "hypothesis_id": "CHM-W-H003",
                    "status": "execution_failed",
                    "decision": "engineering_failure",
                    "test_metrics_opened": False,
                    "exception": {
                        "type": type(error).__name__,
                        "message": str(error),
                    },
                    "environment": {"git_commit": _git_commit()},
                    "claim_boundary": (
                        "Execution failure only; no transfer or model-quality evidence."
                    ),
                },
            )
        raise
