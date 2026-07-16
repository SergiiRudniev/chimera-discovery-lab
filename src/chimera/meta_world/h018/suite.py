"""Five-arm validation-only development suite for H018."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, cast

from chimera.meta_world.generators import GeneratedWorldDatasetConfig
from chimera.meta_world.h018.baselines import evaluate_h018_random_interventions
from chimera.meta_world.h018.config import (
    ALIGNED,
    RANDOM,
    H018SuiteConfig,
)
from chimera.meta_world.h018.dataset import build_h018_smoke_dataset
from chimera.meta_world.h018.preflight import run_h018_preflight


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _all_finite(value: object) -> bool:
    if isinstance(value, bool) or value is None or isinstance(value, str):
        return True
    if isinstance(value, int):
        return True
    if isinstance(value, float):
        return math.isfinite(value)
    if isinstance(value, dict):
        return all(_all_finite(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return all(_all_finite(item) for item in value)
    return True


def run_h018_development_suite(
    config_path: str | Path,
    output_dir: str | Path,
    report_path: str | Path,
) -> dict[str, Any]:
    """Run matched train/validation arms without reading a transfer shard."""

    suite = H018SuiteConfig.from_yaml(config_path)
    output = Path(output_dir)
    report_file = Path(report_path)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("H018 suite output directory must be empty")
    if report_file.exists():
        raise FileExistsError("H018 suite report must not already exist")
    output.mkdir(parents=True, exist_ok=True)

    integrity = build_h018_smoke_dataset(
        output / "fixed_integrity",
        suite.generator_config,
        trajectories_per_split=24,
    )
    generator = GeneratedWorldDatasetConfig.from_yaml(suite.generator_config)
    arms: dict[str, dict[str, Any]] = {}
    for arm in suite.arms:
        if arm.name == RANDOM:
            arms[arm.name] = {
                "status": "completed_evaluator_baseline",
                "metric_scope": arm.metric_scope,
                "validation": evaluate_h018_random_interventions(
                    generator,
                    samples=16,
                    candidates_per_sample=16,
                ).to_dict(),
                "opened_splits": ["validation"],
                "test_metrics_opened": False,
            }
            continue
        if arm.config is None:
            raise RuntimeError(f"missing trainable config for {arm.name}")
        arms[arm.name] = run_h018_preflight(
            arm.config,
            output / arm.name,
        )

    baseline_names = [
        arm.name
        for arm in suite.arms
        if arm.primary_predictive_baseline
    ]
    aligned_validation = cast(dict[str, float], arms[ALIGNED]["best_validation"])
    baseline_validation = {
        name: cast(dict[str, float], arms[name]["best_validation"])
        for name in baseline_names
    }
    effect_best = min(
        metrics["intervention_effect_nrmse"]
        for metrics in baseline_validation.values()
    )
    rollout_best = min(
        metrics["four_step_rollout_nrmse"]
        for metrics in baseline_validation.values()
    )
    integrity_checks = cast(dict[str, bool], integrity["checks"])
    all_trainable_completed = all(
        arms[arm.name]["status"] == "completed_preflight"
        and arms[arm.name]["test_metrics_opened"] is False
        for arm in suite.arms
        if arm.name != RANDOM
    )
    gate = {
        "deterministic_replay_rate": (
            1.0 if integrity_checks["deterministic_replay"] else 0.0
        ),
        "split_leakage_findings": sum(
            not integrity_checks[key]
            for key in (
                "mechanism_id_isolation",
                "world_instance_isolation",
                "seed_isolation",
                "exact_configuration_isolation",
                "exact_train_transfer_program_overlap_zero",
            )
        ),
        "exact_train_transfer_program_overlap": integrity["programs"][
            "exact_train_transfer_overlap"
        ],
        "all_transfer_primitives_seen_in_train": integrity_checks[
            "all_transfer_primitives_seen_in_train"
        ],
        "all_metrics_finite": _all_finite(arms),
        "all_trainable_arms_completed": all_trainable_completed,
        "test_metrics_opened": False,
    }
    hard_pass = (
        gate["deterministic_replay_rate"] == 1.0
        and gate["split_leakage_findings"] == 0
        and gate["exact_train_transfer_program_overlap"] == 0
        and gate["all_transfer_primitives_seen_in_train"] is True
        and gate["all_metrics_finite"] is True
        and gate["all_trainable_arms_completed"] is True
        and gate["test_metrics_opened"] is False
    )
    report: dict[str, Any] = {
        "hypothesis_id": suite.hypothesis_id,
        "trial_id": suite.trial_id,
        "status": "completed_development_suite",
        "decision": (
            "engineering_gate_passed_test_remains_sealed"
            if hard_pass
            else "development_gate_failed"
        ),
        "development_seed": suite.development_seed,
        "frozen_validation_seeds": list(suite.frozen_validation_seeds),
        "frozen_validation_seeds_opened": False,
        "opened_splits": ["train", "validation"],
        "test_metrics_opened": False,
        "checkpoint_promoted": False,
        "scientific_result": False,
        "integrity": integrity,
        "arms": arms,
        "validation_diagnostic": {
            "aligned_intervention_effect_nrmse": aligned_validation[
                "intervention_effect_nrmse"
            ],
            "aligned_four_step_rollout_nrmse": aligned_validation[
                "four_step_rollout_nrmse"
            ],
            "best_baseline_intervention_effect_nrmse": effect_best,
            "best_baseline_four_step_rollout_nrmse": rollout_best,
            "intervention_effect_ratio_vs_best_baseline": (
                aligned_validation["intervention_effect_nrmse"] / effect_best
            ),
            "four_step_rollout_ratio_vs_best_baseline": (
                aligned_validation["four_step_rollout_nrmse"] / rollout_best
            ),
            "primary_transfer_gate_evaluated": False,
        },
        "development_gate": gate,
        "claim_boundary": (
            "Train/validation simulator diagnostics only. No compositional transfer "
            "test, causal, business-utility, language-independence or production claim."
        ),
    }
    _write_json(report_file, report)
    return report

