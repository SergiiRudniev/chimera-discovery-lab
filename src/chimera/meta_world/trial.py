"""Reproducible engineering qualification for Chimera Meta-World W0."""

from __future__ import annotations

import hashlib
import json
import math
import platform
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import torch

from chimera.meta_world.config import MetaWorldExperimentConfig
from chimera.meta_world.model import ChimeraMetaWorld, MetaWorldOutput
from chimera.meta_world.synthetic import make_mechanistic_batch
from chimera.meta_world.trainer import MetaWorldTrainer


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _git_commit() -> str | None:
    completed = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        check=False,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip() if completed.returncode == 0 else None


def _replay_delta(first: MetaWorldOutput, second: MetaWorldOutput) -> float:
    pairs = (
        (first.next_state_mean, second.next_state_mean),
        (first.next_state_log_variance, second.next_state_log_variance),
        (first.effect_mean, second.effect_mean),
        (first.effect_log_variance, second.effect_log_variance),
        (first.proposal_embedding, second.proposal_embedding),
    )
    return max(float((left.float() - right.float()).abs().max().cpu()) for left, right in pairs)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def run_meta_world_trial(
    config_path: str | Path,
    output_dir: str | Path,
    result_path: str | Path,
) -> dict[str, Any]:
    """Execute the preregistered fixed-batch W0 engineering trial."""

    config_file = Path(config_path)
    output = Path(output_dir)
    public_result = Path(result_path)
    output.mkdir(parents=True, exist_ok=True)
    public_result.parent.mkdir(parents=True, exist_ok=True)
    config = MetaWorldExperimentConfig.from_yaml(config_file)
    model = ChimeraMetaWorld(config.model)
    parameters = model.trainable_parameter_count()
    trainer = MetaWorldTrainer(model, config.training)
    if trainer.device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(trainer.device)
    batch = make_mechanistic_batch(
        config.model,
        batch_size=config.training.batch_size,
        active_slots=config.training.active_slots,
        seed=config.training.seed,
    )
    replay_before = _replay_delta(trainer.predict(batch), trainer.predict(batch))
    started = time.perf_counter()
    records: list[dict[str, float | int]] = []
    for step in range(1, config.training.steps + 1):
        metrics = trainer.train_step(batch)
        records.append({"step": step, **metrics})
    runtime_seconds = time.perf_counter() - started
    replay_after = _replay_delta(trainer.predict(batch), trainer.predict(batch))
    initial_loss = float(records[0]["loss"])
    final_loss = float(records[-1]["loss"])
    loss_reduction_fraction = (initial_loss - final_loss) / max(abs(initial_loss), 1e-12)
    all_finite = all(
        math.isfinite(float(value))
        for record in records
        for key, value in record.items()
        if key != "step"
    )
    qualification = config.qualification
    checks = {
        "parameter_range": (
            qualification.minimum_parameters
            <= parameters
            <= qualification.maximum_parameters
        ),
        "loss_reduction": (
            loss_reduction_fraction >= qualification.minimum_loss_reduction_fraction
        ),
        "deterministic_replay": (
            max(replay_before, replay_after) <= qualification.maximum_replay_delta
        ),
        "all_metrics_finite": (all_finite if qualification.require_all_finite else True),
        "required_device": (
            trainer.device.type == "cuda" if qualification.require_cuda else True
        ),
    }
    decision = "accepted" if all(checks.values()) else "rejected"
    environment = {
        "platform": platform.platform(),
        "python": sys.version.split()[0],
        "torch": torch.__version__,
        "device": str(trainer.device),
        "device_name": (
            torch.cuda.get_device_name(trainer.device) if trainer.device.type == "cuda" else None
        ),
        "device_memory_bytes": (
            torch.cuda.get_device_properties(trainer.device).total_memory
            if trainer.device.type == "cuda"
            else None
        ),
        "precision": config.training.precision,
        "git_commit": _git_commit(),
        "config_sha256": _sha256(config_file),
    }
    result: dict[str, Any] = {
        "id": config.experiment_id,
        "trial_id": config.trial_id,
        "status": "completed",
        "decision": decision,
        "architecture": "Chimera Meta-World W0",
        "parameters": parameters,
        "metrics": {
            "steps": config.training.steps,
            "batch_size": config.training.batch_size,
            "initial_loss": initial_loss,
            "final_loss": final_loss,
            "loss_reduction_fraction": loss_reduction_fraction,
            "replay_delta_before": replay_before,
            "replay_delta_after": replay_after,
            "peak_memory_bytes": trainer.peak_memory_bytes(),
            "runtime_seconds": runtime_seconds,
            "all_metrics_finite": all_finite,
        },
        "checks": checks,
        "environment": environment,
        "claim_boundary": (
            "Engineering validation only; no evidence for causal discovery, "
            "cross-domain transfer, grounding quality or production ideation."
        ),
    }
    metrics_path = output / "metrics.jsonl"
    metrics_path.write_text(
        "".join(json.dumps(record, sort_keys=True) + "\n" for record in records),
        encoding="utf-8",
    )
    _write_json(output / "environment.json", environment)
    _write_json(output / "result.json", result)
    _write_json(public_result, result)
    return result
