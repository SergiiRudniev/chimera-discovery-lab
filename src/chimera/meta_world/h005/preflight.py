"""Matched development preflight for the H005 mixed policy curriculum."""

from __future__ import annotations

import hashlib
import json
import platform
import subprocess
import time
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any, Literal

import torch
from torch import nn

from chimera.meta_world.generators import SplitName, WorldGenerationPipeline
from chimera.meta_world.h002 import (
    H002Trainer,
    RelationalSequenceWorldModel,
    TemporalWorldBaseline,
    concatenate_sequence_samples,
    evaluate_h002_model,
    make_transition_window,
    materialize_sequence_sample,
)
from chimera.meta_world.h002.windows import GeneratedSequenceSample
from chimera.meta_world.h003.trainer import H003Trainer
from chimera.meta_world.h004 import SeededRandomPolicy
from chimera.meta_world.h005.config import H005Arm, H005RunConfig


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


def _model(config: H005RunConfig) -> nn.Module:
    if config.arm is H005Arm.TEMPORAL:
        return TemporalWorldBaseline(config.model)
    if config.arm in {
        H005Arm.MIXED,
        H005Arm.RANDOM_ONLY,
        H005Arm.PROBE_ONLY,
        H005Arm.ONE_STEP,
    }:
        return RelationalSequenceWorldModel(config.model)
    raise ValueError("legal random intervention has no trainable H005 model")


def _trainer(model: nn.Module, config: H005RunConfig) -> H002Trainer:
    if config.arm in {H005Arm.MIXED, H005Arm.RANDOM_ONLY, H005Arm.PROBE_ONLY}:
        return H003Trainer(
            model,
            config.training,
            rollout_horizon=config.closed_loop.rollout_horizon,
            state_features=config.dataset.worlds.state_features,
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


def _training_sample(
    config: H005RunConfig,
    probe_pipeline: WorldGenerationPipeline,
    random_pipeline: WorldGenerationPipeline,
    random_pipeline_b: WorldGenerationPipeline,
    step: int,
) -> GeneratedSequenceSample:
    batch_size = config.training.batch_size
    if config.arm is H005Arm.MIXED:
        per_policy = batch_size // 2
        start_index = (step - 1) * per_policy
        probe = materialize_sequence_sample(
            probe_pipeline,
            SplitName.TRAIN,
            start_index=start_index,
            batch_size=per_policy,
        )
        random = materialize_sequence_sample(
            random_pipeline,
            SplitName.TRAIN,
            start_index=start_index,
            batch_size=per_policy,
        )
        return concatenate_sequence_samples(probe, random)
    if config.arm is H005Arm.RANDOM_ONLY:
        per_policy = batch_size // 2
        start_index = (step - 1) * per_policy
        random_a = materialize_sequence_sample(
            random_pipeline,
            SplitName.TRAIN,
            start_index=start_index,
            batch_size=per_policy,
        )
        random_b = materialize_sequence_sample(
            random_pipeline_b,
            SplitName.TRAIN,
            start_index=start_index,
            batch_size=per_policy,
        )
        return concatenate_sequence_samples(random_a, random_b)
    pipeline = probe_pipeline if config.arm is H005Arm.PROBE_ONLY else random_pipeline
    return materialize_sequence_sample(
        pipeline,
        SplitName.TRAIN,
        start_index=(step - 1) * batch_size,
        batch_size=batch_size,
    )


def _policy_label(arm: H005Arm) -> str:
    if arm is H005Arm.MIXED:
        return "mixed_probe_0.5_seeded_random_0.5"
    if arm is H005Arm.PROBE_ONLY:
        return "deterministic_system_identification_probe_v1"
    if arm is H005Arm.RANDOM_ONLY:
        return "paired_seeded_random_views"
    return "seeded_random"


def execute_policy_curriculum_run(
    config_file: Path,
    config: H005RunConfig,
    output_dir: str | Path,
    *,
    expected_mode: str,
    hypothesis_id: str,
    reported_arm: str,
    effect_supervision: Literal["all", "random_half"],
    result_metadata: Mapping[str, object] | None = None,
    trainer_factory: Callable[[nn.Module, H005RunConfig], H002Trainer] | None = None,
    model_factory: Callable[[H005RunConfig], nn.Module] | None = None,
) -> dict[str, Any]:
    if config.mode != expected_mode:
        raise ValueError(f"policy curriculum runner expected mode={expected_mode}")
    if effect_supervision == "random_half" and config.arm is not H005Arm.MIXED:
        raise ValueError("random-half effect routing requires the mixed sampler")
    frozen_validation = config.mode == "frozen_validation"
    output = Path(output_dir)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("preflight output directory must be empty")
    output.mkdir(parents=True, exist_ok=True)
    worlds = config.dataset.worlds
    probe_policy = config.dataset.policies()[SplitName.TRAIN]
    random_policy = config.dataset.random_training_policies()[SplitName.TRAIN]
    evaluation_policy = config.dataset.policies()[SplitName.VALIDATION]
    probe_pipeline = WorldGenerationPipeline(worlds, probe_policy)
    random_pipeline = WorldGenerationPipeline(worlds, random_policy)
    random_pipeline_b = WorldGenerationPipeline(
        worlds,
        SeededRandomPolicy(draw_offset=1),
    )
    evaluation_pipeline = WorldGenerationPipeline(worlds, evaluation_policy)
    validation_sample = materialize_sequence_sample(
        evaluation_pipeline,
        SplitName.VALIDATION,
        start_index=0,
        batch_size=config.evaluation.validation_trajectories,
    )
    torch.manual_seed(config.training.seed)
    if config.training.device == "cuda":
        torch.cuda.manual_seed_all(config.training.seed)
    model = model_factory(config) if model_factory is not None else _model(config)
    model_class = f"{type(model).__module__}.{type(model).__qualname__}"
    trainer = (
        trainer_factory(model, config)
        if trainer_factory is not None
        else _trainer(model, config)
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

    def save_checkpoint(step: int) -> None:
        torch.save(
            {
                "run_id": config.run_id,
                "hypothesis_id": hypothesis_id,
                "arm": reported_arm,
                "model_class": model_class,
                "step": step,
                "model": trainer.evaluation_state_dict(),
                "weights_kind": trainer.evaluation_weights_kind,
                "model_config": config.to_dict()["model"],
            },
            checkpoint_path,
        )

    if not frozen_validation:
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
    gradient_conflicts: list[float] = []
    gradient_cosines: list[float] = []
    one_step_prediction_count = worlds.trajectory_steps - 1
    first_rollout_step = config.model.context_steps - 1
    rollout_start_count = (
        worlds.trajectory_steps
        - config.closed_loop.rollout_horizon
        - first_rollout_step
    )
    for step in range(1, config.training.steps + 1):
        train_sample = _training_sample(
            config,
            probe_pipeline,
            random_pipeline,
            random_pipeline_b,
            step,
        )
        if isinstance(trainer, H003Trainer):
            effect_supervision_mask: torch.Tensor | None = None
            if effect_supervision == "random_half":
                per_policy = train_sample.batch.batch_size // 2
                effect_supervision_mask = torch.cat(
                    [
                        torch.zeros(per_policy, dtype=torch.bool),
                        torch.ones(per_policy, dtype=torch.bool),
                    ]
                )
            training_metrics = trainer.train_sequence_step(
                train_sample,
                prediction_step=(
                    first_rollout_step + (step - 1) % rollout_start_count
                ),
                context_steps=config.model.context_steps,
                effect_supervision_mask=effect_supervision_mask,
            )
        else:
            window = make_transition_window(
                train_sample,
                prediction_step=(step - 1) % one_step_prediction_count,
                context_steps=config.model.context_steps,
            )
            training_metrics = trainer.train_step(window)
        if first_training is None:
            first_training = training_metrics
        final_training = training_metrics
        if "gradient_conflict_applied" in training_metrics:
            gradient_conflicts.append(training_metrics["gradient_conflict_applied"])
        if "gradient_cosine" in training_metrics:
            gradient_cosines.append(training_metrics["gradient_cosine"])
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
            select_checkpoint = (
                frozen_validation and step == config.frozen_checkpoint_step
            ) or (not frozen_validation and score < best_score)
            if select_checkpoint:
                best_score = score
                best_step = step
                best_metrics = evaluation
                save_checkpoint(best_step)
    if frozen_validation and best_step != config.frozen_checkpoint_step:
        raise RuntimeError("frozen policy-curriculum checkpoint was not evaluated")
    metrics_path = output / "metrics.jsonl"
    metrics_path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in metric_rows),
        encoding="utf-8",
    )
    checkpoint_manifest = {
        "run_id": config.run_id,
        "hypothesis_id": hypothesis_id,
        "arm": reported_arm,
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
        "scope": (
            f"frozen-validation-only {hypothesis_id} run"
            if frozen_validation
            else f"development-only {hypothesis_id} preflight"
        ),
        "opened_splits": [SplitName.TRAIN.value, SplitName.VALIDATION.value],
    }
    _write_json(output / "checkpoint_manifest.json", checkpoint_manifest)
    result: dict[str, Any] = {
        "run_id": config.run_id,
        "hypothesis_id": hypothesis_id,
        "status": (
            "completed_frozen_validation"
            if frozen_validation
            else "completed_development_preflight"
        ),
        "decision": "validation_only" if frozen_validation else "engineering_only",
        "arm": reported_arm,
        "seed": config.training.seed,
        "train_action_policy": _policy_label(config.arm),
        "evaluation_action_policy": evaluation_policy.policy_id,
        "model_class": model_class,
        "weights_kind": trainer.evaluation_weights_kind,
        "parameters": _parameter_count(model),
        "opened_splits": [SplitName.TRAIN.value, SplitName.VALIDATION.value],
        "frozen_validation_seeds_opened": frozen_validation,
        "test_metrics_opened": False,
        "initial_validation": initial_evaluation,
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
            "dataset_config_sha256": _sha256(config.dataset_config_path),
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
            "Frozen validation only; no test transfer, causal discovery, real-world "
            "probe safety, business utility or production claim."
            if frozen_validation
            else "Development validation only; no frozen validation, test transfer, "
            "causal discovery, real-world probe safety, business utility or "
            "production claim."
        ),
    }
    if gradient_conflicts and gradient_cosines:
        result["training_diagnostics"] = {
            "gradient_conflict_fraction": sum(gradient_conflicts)
            / len(gradient_conflicts),
            "gradient_cosine_mean": sum(gradient_cosines) / len(gradient_cosines),
            "gradient_cosine_minimum": min(gradient_cosines),
            "gradient_cosine_maximum": max(gradient_cosines),
        }
    if result_metadata is not None:
        result.update(result_metadata)
    _write_json(output / "result.json", result)
    return result


def _execute_h005_run(
    config_path: str | Path,
    output_dir: str | Path,
    *,
    expected_mode: str,
) -> dict[str, Any]:
    config_file = Path(config_path)
    config = H005RunConfig.from_yaml(config_file)
    return execute_policy_curriculum_run(
        config_file,
        config,
        output_dir,
        expected_mode=expected_mode,
        hypothesis_id="CHM-W-H005",
        reported_arm=config.arm.value,
        effect_supervision="all",
    )


def _run_h005(
    config_path: str | Path,
    output_dir: str | Path,
    *,
    expected_mode: str,
) -> dict[str, Any]:
    """Run one H005 stage and persist failures before re-raising."""

    try:
        return _execute_h005_run(
            config_path,
            output_dir,
            expected_mode=expected_mode,
        )
    except Exception as error:
        output = Path(output_dir)
        output.mkdir(parents=True, exist_ok=True)
        result_path = output / "result.json"
        if not result_path.exists():
            run_id: str | None = None
            try:
                run_id = H005RunConfig.from_yaml(config_path).run_id
            except Exception:
                run_id = None
            _write_json(
                result_path,
                {
                    "run_id": run_id,
                    "hypothesis_id": "CHM-W-H005",
                    "status": "execution_failed",
                    "decision": "engineering_failure",
                    "frozen_validation_seeds_opened": (
                        expected_mode == "frozen_validation"
                    ),
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


def run_h005_preflight(
    config_path: str | Path,
    output_dir: str | Path,
) -> dict[str, Any]:
    """Run an H005 development preflight."""

    return _run_h005(config_path, output_dir, expected_mode="preflight")


def run_h005_validation(
    config_path: str | Path,
    output_dir: str | Path,
) -> dict[str, Any]:
    """Run an H005 frozen-validation seed without opening test splits."""

    return _run_h005(config_path, output_dir, expected_mode="frozen_validation")
