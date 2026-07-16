"""Deterministic multi-arm development gate for CHM-W-H008."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import yaml

from chimera.meta_world.h004 import H004DatasetConfig
from chimera.meta_world.h008.baselines import evaluate_legal_random_interventions
from chimera.meta_world.h008.config import H008Arm, H008RunConfig
from chimera.meta_world.h008.preflight import run_h008_preflight


def _mapping(value: object, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{name} must be a mapping")
    return cast(Mapping[str, Any], value)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


@dataclass(frozen=True)
class H008SuiteConfig:
    """Frozen development suite and reuse boundary for validated WG1 data."""

    hypothesis_id: str
    trial_id: str
    seed: int
    generator_config: Path
    dataset_integrity_report: Path
    arms: Mapping[H008Arm, Path]
    random_samples: int
    random_candidates: int
    effect_ratio_maximum: float
    rollout_ratio_maximum: float
    coverage_minimum: float
    identity_residual_maximum: float
    test_access: str

    def __post_init__(self) -> None:
        if self.hypothesis_id != "CHM-W-H008" or self.trial_id != "CHM-W-T008":
            raise ValueError("H008 suite IDs are immutable")
        if self.seed != 260922:
            raise ValueError("H008 development seed is immutable")
        if set(self.arms) != set(H008Arm):
            raise ValueError("H008 suite must contain every trainable comparison arm")
        if self.random_samples <= 0 or self.random_candidates < 2:
            raise ValueError("H008 random baseline shape is invalid")
        if (
            self.effect_ratio_maximum != 0.90
            or self.rollout_ratio_maximum != 1.00
            or self.coverage_minimum != 0.85
            or self.identity_residual_maximum != 0.000001
        ):
            raise ValueError("H008 development gate differs from preregistration")
        if self.test_access != "sealed_until_development_gate_passes":
            raise ValueError("H008 test-access boundary is invalid")

    @classmethod
    def from_yaml(cls, path: str | Path) -> H008SuiteConfig:
        with Path(path).open("r", encoding="utf-8") as handle:
            values = _mapping(yaml.safe_load(handle), "H008 suite")
        expected = {
            "hypothesis_id",
            "trial_id",
            "seed",
            "generator_config",
            "dataset_integrity_report",
            "arms",
            "legal_random_intervention",
            "development_gate",
            "test_access",
        }
        if set(values) != expected:
            raise ValueError("H008 suite fields must exactly match the frozen schema")
        arm_values = _mapping(values["arms"], "H008 arms")
        random = _mapping(
            values["legal_random_intervention"],
            "H008 legal random baseline",
        )
        gate = _mapping(values["development_gate"], "H008 development gate")
        if set(random) != {"samples", "candidates_per_sample"}:
            raise ValueError("H008 legal random baseline fields are invalid")
        if set(gate) != {
            "intervention_effect_nrmse_ratio_maximum",
            "four_step_rollout_nrmse_ratio_maximum",
            "intervention_effect_90_coverage_minimum",
            "counterfactual_identity_maximum_absolute_residual",
        }:
            raise ValueError("H008 development gate fields are invalid")
        return cls(
            hypothesis_id=str(values["hypothesis_id"]),
            trial_id=str(values["trial_id"]),
            seed=int(values["seed"]),
            generator_config=Path(str(values["generator_config"])),
            dataset_integrity_report=Path(str(values["dataset_integrity_report"])),
            arms={H008Arm(str(key)): Path(str(value)) for key, value in arm_values.items()},
            random_samples=int(random["samples"]),
            random_candidates=int(random["candidates_per_sample"]),
            effect_ratio_maximum=float(
                gate["intervention_effect_nrmse_ratio_maximum"]
            ),
            rollout_ratio_maximum=float(
                gate["four_step_rollout_nrmse_ratio_maximum"]
            ),
            coverage_minimum=float(
                gate["intervention_effect_90_coverage_minimum"]
            ),
            identity_residual_maximum=float(
                gate["counterfactual_identity_maximum_absolute_residual"]
            ),
            test_access=str(values["test_access"]),
        )


def _metric(result: Mapping[str, Any], name: str) -> float:
    metrics = _mapping(result["best_validation"], "H008 validation metrics")
    return float(metrics[name])


def _audit(result: Mapping[str, Any], name: str) -> float | None:
    audit = _mapping(result["counterfactual_audit"], "H008 counterfactual audit")
    value = audit[name]
    return None if value is None else float(value)


def _integrity_evidence(config: H008SuiteConfig) -> dict[str, object]:
    evidence = json.loads(config.dataset_integrity_report.read_text(encoding="utf-8"))
    expected_hash = _sha256(config.generator_config)
    if evidence.get("dataset_config_sha256") != expected_hash:
        raise ValueError("WG1 integrity evidence does not match the H008 generator config")
    gate = _mapping(evidence["development_gate"], "WG1 integrity gate")
    replay = float(gate["deterministic_dataset_replay_rate"])
    leakage = int(gate["split_leakage_findings"])
    if replay != 1.0 or leakage != 0:
        raise ValueError("validated WG1 integrity evidence is not clean")
    return {
        "source": config.dataset_integrity_report.as_posix(),
        "source_sha256": _sha256(config.dataset_integrity_report),
        "generator_config_sha256": expected_hash,
        "deterministic_replay_rate": replay,
        "split_leakage_findings": leakage,
        "revalidated": False,
    }


def _arm_summary(config_path: Path, result: Mapping[str, Any]) -> dict[str, object]:
    manifest_path = Path(str(result["artifacts"]["checkpoint_manifest"]))
    if not manifest_path.is_absolute():
        manifest_path = Path(str(result["output_dir"])) / manifest_path
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    return {
        "run_id": result["run_id"],
        "config": config_path.as_posix(),
        "config_sha256": _sha256(config_path),
        "selected_step": result["best_step"],
        "model_class": result["model_class"],
        "parameters": result["parameters"],
        "metrics": result["best_validation"],
        "counterfactual_audit": result["counterfactual_audit"],
        "checkpoint": {
            "sha256": manifest["checkpoint_sha256"],
            "weights_kind": manifest["weights_kind"],
            "promoted": False,
        },
        "runtime_seconds": result["runtime_seconds"],
        "peak_memory_bytes": result["peak_memory_bytes"],
    }


def run_h008_development_suite(
    config_path: str | Path,
    output_dir: str | Path,
    report_path: str | Path,
) -> dict[str, Any]:
    """Run every H008 development arm and apply the preregistered gate once."""

    suite = H008SuiteConfig.from_yaml(config_path)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    results: dict[H008Arm, dict[str, Any]] = {}
    for arm, arm_config_path in suite.arms.items():
        arm_config = H008RunConfig.from_yaml(arm_config_path)
        if arm_config.arm is not arm or arm_config.runtime.training.seed != suite.seed:
            raise ValueError("H008 suite arm config does not match the frozen suite")
        arm_output = output / arm.value
        result = run_h008_preflight(arm_config_path, arm_output)
        result["output_dir"] = str(arm_output)
        results[arm] = result

    counterfactual = results[H008Arm.COUNTERFACTUAL_MIXED]
    direct = results[H008Arm.DIRECT_MIXED]
    effect_ratio = _metric(counterfactual, "intervention_effect_nrmse") / _metric(
        direct,
        "intervention_effect_nrmse",
    )
    rollout_ratio = _metric(counterfactual, "four_step_rollout_nrmse") / _metric(
        direct,
        "four_step_rollout_nrmse",
    )
    random_effect_ratio = _metric(
        results[H008Arm.COUNTERFACTUAL_RANDOM],
        "intervention_effect_nrmse",
    ) / _metric(results[H008Arm.DIRECT_RANDOM], "intervention_effect_nrmse")
    identity_residual = _audit(
        counterfactual,
        "counterfactual_identity_maximum_absolute_residual",
    )
    if identity_residual is None:
        raise RuntimeError("counterfactual arm did not expose its algebraic identity")
    coverage = _metric(counterfactual, "intervention_effect_90_coverage")
    compared_parameters = {
        int(results[arm]["parameters"])
        for arm in {
            H008Arm.COUNTERFACTUAL_MIXED,
            H008Arm.DIRECT_MIXED,
            H008Arm.COUNTERFACTUAL_RANDOM,
            H008Arm.DIRECT_RANDOM,
        }
    }
    numeric_gate_values = [
        effect_ratio,
        rollout_ratio,
        random_effect_ratio,
        identity_residual,
        coverage,
    ]
    all_metrics_finite = all(math.isfinite(value) for value in numeric_gate_values)
    integrity = _integrity_evidence(suite)
    passed = (
        effect_ratio <= suite.effect_ratio_maximum
        and rollout_ratio <= suite.rollout_ratio_maximum
        and coverage >= suite.coverage_minimum
        and identity_residual <= suite.identity_residual_maximum
        and len(compared_parameters) == 1
        and all_metrics_finite
    )
    random_baseline = evaluate_legal_random_interventions(
        H004DatasetConfig.from_yaml(suite.generator_config).worlds,
        samples=suite.random_samples,
        candidates_per_sample=suite.random_candidates,
    ).to_dict()
    first = counterfactual
    report: dict[str, Any] = {
        "schema_version": 1,
        "preflight_id": "CHM-W-H008-DEVELOPMENT-001",
        "hypothesis_id": suite.hypothesis_id,
        "trial_id": suite.trial_id,
        "status": "completed_development_preflight",
        "scientific_result": False,
        "registered_trial_executed": False,
        "seed": suite.seed,
        "arms": {
            arm.value: _arm_summary(suite.arms[arm], results[arm])
            for arm in H008Arm
        },
        "legal_random_intervention": random_baseline,
        "comparisons": {
            "counterfactual_vs_direct_mixed": {
                "intervention_effect_nrmse_ratio": effect_ratio,
                "four_step_rollout_nrmse_ratio": rollout_ratio,
            },
            "counterfactual_vs_direct_random": {
                "intervention_effect_nrmse_ratio": random_effect_ratio,
            },
        },
        "development_gate": {
            "intervention_effect_nrmse_ratio": effect_ratio,
            "intervention_effect_nrmse_ratio_maximum": suite.effect_ratio_maximum,
            "four_step_rollout_nrmse_ratio": rollout_ratio,
            "four_step_rollout_nrmse_ratio_maximum": suite.rollout_ratio_maximum,
            "intervention_effect_90_coverage": coverage,
            "intervention_effect_90_coverage_minimum": suite.coverage_minimum,
            "counterfactual_identity_maximum_absolute_residual": identity_residual,
            "counterfactual_identity_maximum_absolute_residual_maximum": (
                suite.identity_residual_maximum
            ),
            "parameter_count_matched": len(compared_parameters) == 1,
            "all_metrics_finite": all_metrics_finite,
            "deterministic_replay_rate": integrity["deterministic_replay_rate"],
            "split_leakage_findings": integrity["split_leakage_findings"],
            "test_metrics_opened": False,
            "passed": passed,
        },
        "dataset_integrity": integrity,
        "decision": (
            "freeze_H008_hyperparameters_and_open_registered_validation_seeds"
            if passed
            else "do_not_open_H008_frozen_validation"
        ),
        "checkpoint_promoted": False,
        "opened_splits": ["train", "validation"],
        "frozen_validation_seeds_opened": False,
        "test_metrics_opened": False,
        "environment": first["environment"],
        "claim_boundary": (
            "Development-only generated-simulator evidence. Frozen validation "
            "seeds and every test split remained sealed; no real-world causal, "
            "business-utility, language-independence or production claim."
        ),
    }
    report_file = Path(report_path)
    report_file.parent.mkdir(parents=True, exist_ok=True)
    report_file.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return report
