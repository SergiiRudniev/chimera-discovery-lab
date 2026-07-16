"""Validation-only H002 training preflight with no test-split access."""

from __future__ import annotations

import hashlib
import json
import platform
import subprocess
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import torch
from torch import nn

from chimera.meta_world.batch import MetaWorldBatch
from chimera.meta_world.config import MetaWorldTrainingConfig
from chimera.meta_world.generators import (
    GeneratedWorldDatasetConfig,
    SplitName,
    WorldGenerationPipeline,
)
from chimera.meta_world.h002.config import H002Arm, H002RunConfig
from chimera.meta_world.h002.evaluation import evaluate_h002_model
from chimera.meta_world.h002.model import (
    RelationalSequenceWorldModel,
    TemporalWorldBaseline,
)
from chimera.meta_world.h002.trainer import H002Trainer
from chimera.meta_world.h002.windows import (
    GeneratedSequenceSample,
    make_transition_window,
    materialize_sequence_sample,
)


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


def _model(config: H002RunConfig) -> nn.Module:
    if config.arm is H002Arm.TEMPORAL:
        return TemporalWorldBaseline(config.model)
    if config.arm in {
        H002Arm.ALIGNED,
        H002Arm.NO_ALIGNMENT,
        H002Arm.TARGET_FAMILY_ONLY,
    }:
        return RelationalSequenceWorldModel(config.model)
    raise ValueError("legal random intervention has no trainable model")


def _parameter_count(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)


def _selection_score(metrics: dict[str, float]) -> float:
    return 0.5 * (
        metrics["intervention_effect_nrmse"]
        + metrics["four_step_rollout_nrmse"]
    )


def _execute_h002_preflight(
    config_path: str | Path,
    output_dir: str | Path,
    *,
    hypothesis_id: str = "CHM-W-H002",
    run_config: H002RunConfig | None = None,
    model_factory: Callable[[H002RunConfig], nn.Module] | None = None,
    trainer_factory: Callable[
        [nn.Module, MetaWorldTrainingConfig], H002Trainer
    ]
    | None = None,
    window_factory: Callable[
        [GeneratedSequenceSample, int, int], MetaWorldBatch
    ]
    | None = None,
    training_pipeline_factory: Callable[
        [GeneratedWorldDatasetConfig], WorldGenerationPipeline
    ]
    | None = None,
    validation_pipeline_factory: Callable[
        [GeneratedWorldDatasetConfig], WorldGenerationPipeline
    ]
    | None = None,
    allow_target_family_only: bool = False,
) -> dict[str, Any]:
    """Train on online train worlds and select only against frozen validation worlds."""

    config_file = Path(config_path)
    config = run_config or H002RunConfig.from_yaml(config_file)
    if config.mode != "preflight":
        raise ValueError("run_h002_preflight only accepts mode=preflight")
    if config.arm is H002Arm.TARGET_FAMILY_ONLY and not allow_target_family_only:
        raise ValueError("target-family sampling is reserved for the frozen trial runner")
    output = Path(output_dir)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("preflight output directory must be empty")
    output.mkdir(parents=True, exist_ok=True)
    generator_config = GeneratedWorldDatasetConfig.from_yaml(config.generator_config)
    validation_pipeline = (
        WorldGenerationPipeline(generator_config)
        if validation_pipeline_factory is None
        else validation_pipeline_factory(generator_config)
    )
    training_pipeline = (
        WorldGenerationPipeline(generator_config)
        if training_pipeline_factory is None
        else training_pipeline_factory(generator_config)
    )
    validation_sample = materialize_sequence_sample(
        validation_pipeline,
        SplitName.VALIDATION,
        start_index=0,
        batch_size=config.evaluation.validation_trajectories,
    )
    model = _model(config) if model_factory is None else model_factory(config)
    model_class = f"{type(model).__module__}.{type(model).__qualname__}"
    trainer = (
        H002Trainer(model, config.training)
        if trainer_factory is None
        else trainer_factory(model, config.training)
    )
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
    torch.save(
        {
            "run_id": config.run_id,
            "arm": config.arm.value,
            "model_class": model_class,
            "step": best_step,
            "model": trainer.evaluation_state_dict(),
            "weights_kind": trainer.evaluation_weights_kind,
            "model_config": config.to_dict()["model"],
        },
        checkpoint_path,
    )
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
    for step in range(1, config.training.steps + 1):
        train_sample = materialize_sequence_sample(
            training_pipeline,
            SplitName.TRAIN,
            start_index=(step - 1) * config.training.batch_size,
            batch_size=config.training.batch_size,
        )
        prediction_step = (step - 1) % prediction_count
        window = (
            make_transition_window(
                train_sample,
                prediction_step=prediction_step,
                context_steps=config.model.context_steps,
            )
            if window_factory is None
            else window_factory(
                train_sample,
                prediction_step,
                config.model.context_steps,
            )
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
                torch.save(
                    {
                        "run_id": config.run_id,
                        "arm": config.arm.value,
                        "model_class": model_class,
                        "step": best_step,
                        "model": trainer.evaluation_state_dict(),
                        "weights_kind": trainer.evaluation_weights_kind,
                        "model_config": config.to_dict()["model"],
                    },
                    checkpoint_path,
                )
    metrics_path = output / "metrics.jsonl"
    metrics_path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in metric_rows),
        encoding="utf-8",
    )
    checkpoint_manifest = {
        "run_id": config.run_id,
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
        "scope": "validation-only engineering preflight",
        "opened_splits": [SplitName.TRAIN.value, SplitName.VALIDATION.value],
    }
    _write_json(output / "checkpoint_manifest.json", checkpoint_manifest)
    runtime_seconds = time.perf_counter() - started
    result: dict[str, Any] = {
        "run_id": config.run_id,
        "hypothesis_id": hypothesis_id,
        "status": "completed_preflight",
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
        "runtime_seconds": runtime_seconds,
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
            "Validation-only engineering evidence; no test transfer, causal discovery, "
            "business utility or production checkpoint claim."
        ),
    }
    _write_json(output / "result.json", result)
    return result


def run_h002_preflight(
    config_path: str | Path,
    output_dir: str | Path,
) -> dict[str, Any]:
    """Run the preflight and persist unexpected execution failures before re-raising."""

    return run_generated_world_preflight(
        config_path,
        output_dir,
        hypothesis_id="CHM-W-H002",
    )


def run_generated_world_preflight(
    config_path: str | Path,
    output_dir: str | Path,
    *,
    hypothesis_id: str,
    run_config: H002RunConfig | None = None,
    model_factory: Callable[[H002RunConfig], nn.Module] | None = None,
    trainer_factory: Callable[
        [nn.Module, MetaWorldTrainingConfig], H002Trainer
    ]
    | None = None,
    window_factory: Callable[
        [GeneratedSequenceSample, int, int], MetaWorldBatch
    ]
    | None = None,
    training_pipeline_factory: Callable[
        [GeneratedWorldDatasetConfig], WorldGenerationPipeline
    ]
    | None = None,
    validation_pipeline_factory: Callable[
        [GeneratedWorldDatasetConfig], WorldGenerationPipeline
    ]
    | None = None,
    allow_target_family_only: bool = False,
) -> dict[str, Any]:
    """Run the shared generated-world preflight under an explicit hypothesis ID."""

    try:
        return _execute_h002_preflight(
            config_path,
            output_dir,
            hypothesis_id=hypothesis_id,
            run_config=run_config,
            model_factory=model_factory,
            trainer_factory=trainer_factory,
            window_factory=window_factory,
            training_pipeline_factory=training_pipeline_factory,
            validation_pipeline_factory=validation_pipeline_factory,
            allow_target_family_only=allow_target_family_only,
        )
    except Exception as error:
        output = Path(output_dir)
        output.mkdir(parents=True, exist_ok=True)
        result_path = output / "result.json"
        if not result_path.exists():
            run_id: str | None = None
            try:
                run_id = (
                    run_config.run_id
                    if run_config is not None
                    else H002RunConfig.from_yaml(config_path).run_id
                )
            except Exception:
                run_id = None
            _write_json(
                result_path,
                {
                    "run_id": run_id,
                    "hypothesis_id": hypothesis_id,
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
