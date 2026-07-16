"""Deterministic multi-arm development gate for CHM-W-H013."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import yaml

from chimera.meta_world.generators import GeneratedWorldDatasetConfig
from chimera.meta_world.h012.baselines import evaluate_legal_random_interventions
from chimera.meta_world.h013.config import H013Arm, H013RunConfig
from chimera.meta_world.h013.preflight import run_h013_preflight


def _mapping(value: object, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{name} must be a mapping")
    return cast(Mapping[str, Any], value)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


@dataclass(frozen=True)
class H013SuiteConfig:
    """Frozen three-model comparison and dataset evidence boundary."""

    hypothesis_id: str
    trial_id: str
    mode: str
    seed: int
    frozen_validation_seeds: tuple[int, ...]
    test_access: str
    generator_config: Path
    dataset_integrity_report: Path
    arms: Mapping[H013Arm, Path]
    delta_ratio_maximum: float
    rollout_ratio_maximum: float
    no_op_ratio_maximum: float
    effect_ratio_maximum: float
    identity_residual_maximum: float

    def __post_init__(self) -> None:
        if self.hypothesis_id != "CHM-W-H013" or self.trial_id != "CHM-W-T013":
            raise ValueError("H013 suite IDs are immutable")
        if self.mode != "development" or self.seed != 260942:
            raise ValueError("H013 development mode and seed are immutable")
        if self.frozen_validation_seeds != (260943, 260944, 260945):
            raise ValueError("H013 frozen validation seeds differ from registration")
        if self.test_access != "sealed":
            raise ValueError("H013 test access must remain sealed")
        if set(self.arms) != set(H013Arm):
            raise ValueError("H013 suite requires every registered trainable arm")
        if (
            self.delta_ratio_maximum != 0.90
            or self.rollout_ratio_maximum != 1.00
            or self.no_op_ratio_maximum != 1.00
            or self.effect_ratio_maximum != 1.00
            or self.identity_residual_maximum != 0.000001
        ):
            raise ValueError("H013 gate differs from preregistration")
        generator = GeneratedWorldDatasetConfig.from_yaml(self.generator_config)
        if generator.hypothesis_id != self.hypothesis_id or generator.dataset_id != "CHM-W-WG4":
            raise ValueError("H013 suite requires WG4")
        for arm, config_path in self.arms.items():
            run = H013RunConfig.from_yaml(config_path)
            if run.arm is not arm or run.runtime.training.seed != self.seed:
                raise ValueError(f"H013 run config disagrees with suite arm {arm.value}")

    @classmethod
    def from_yaml(cls, path: str | Path) -> H013SuiteConfig:
        with Path(path).open("r", encoding="utf-8") as handle:
            values = _mapping(yaml.safe_load(handle), "H013 suite")
        expected = {
            "hypothesis_id",
            "trial_id",
            "mode",
            "development_seed",
            "frozen_validation_seeds",
            "test_access",
            "generator_config",
            "dataset_integrity_report",
            "arms",
            "primary_metrics",
            "development_gate",
        }
        if set(values) != expected:
            raise ValueError("H013 suite fields must exactly match the frozen schema")
        primary_metrics = tuple(str(item) for item in values["primary_metrics"])
        if primary_metrics != (
            "intervention_state_delta_nrmse",
            "four_step_rollout_nrmse",
        ):
            raise ValueError("H013 primary metrics differ from preregistration")
        gate = _mapping(values["development_gate"], "H013 gate")
        expected_gate = {
            "intervention_state_delta_nrmse_ratio_vs_matched_direct_maximum",
            "four_step_rollout_nrmse_ratio_vs_matched_direct_maximum",
            "no_op_state_nrmse_ratio_vs_matched_direct_maximum",
            "intervention_effect_nrmse_ratio_vs_matched_direct_maximum",
            "factorized_identity_maximum_absolute_residual",
            "deterministic_replay_rate",
            "split_leakage_findings",
            "all_metrics_finite",
            "test_metrics_opened",
        }
        if set(gate) != expected_gate:
            raise ValueError("H013 development gate fields are invalid")
        if (
            float(gate["deterministic_replay_rate"]) != 1.0
            or int(gate["split_leakage_findings"]) != 0
            or not bool(gate["all_metrics_finite"])
            or bool(gate["test_metrics_opened"])
        ):
            raise ValueError("H013 non-metric gate invariants are invalid")
        arm_values = _mapping(values["arms"], "H013 arms")
        return cls(
            hypothesis_id=str(values["hypothesis_id"]),
            trial_id=str(values["trial_id"]),
            mode=str(values["mode"]),
            seed=int(values["development_seed"]),
            frozen_validation_seeds=tuple(
                int(seed) for seed in values["frozen_validation_seeds"]
            ),
            test_access=str(values["test_access"]),
            generator_config=Path(str(values["generator_config"])),
            dataset_integrity_report=Path(str(values["dataset_integrity_report"])),
            arms={
                H013Arm(str(name)): Path(str(config))
                for name, config in arm_values.items()
            },
            delta_ratio_maximum=float(
                gate[
                    "intervention_state_delta_nrmse_ratio_vs_matched_direct_maximum"
                ]
            ),
            rollout_ratio_maximum=float(
                gate["four_step_rollout_nrmse_ratio_vs_matched_direct_maximum"]
            ),
            no_op_ratio_maximum=float(
                gate["no_op_state_nrmse_ratio_vs_matched_direct_maximum"]
            ),
            effect_ratio_maximum=float(
                gate["intervention_effect_nrmse_ratio_vs_matched_direct_maximum"]
            ),
            identity_residual_maximum=float(
                gate["factorized_identity_maximum_absolute_residual"]
            ),
        )


def _metric(result: Mapping[str, Any], name: str) -> float:
    metrics = _mapping(result["best_validation"], "H013 validation metrics")
    value = metrics[name]
    if value is None:
        raise RuntimeError(f"H013 arm did not expose required metric {name}")
    return float(value)


def _integrity(config: H013SuiteConfig) -> dict[str, Any]:
    evidence = json.loads(config.dataset_integrity_report.read_text(encoding="utf-8"))
    if evidence.get("status") != "passed":
        raise ValueError("WG4 integrity evidence is not passing")
    if evidence.get("dataset_config_sha256") != _sha256(config.generator_config):
        raise ValueError("WG4 integrity evidence does not match the generator config")
    checks = _mapping(evidence["checks"], "WG4 checks")
    leakage_names = (
        "mechanism_id_isolation",
        "world_instance_isolation",
        "seed_isolation",
        "exact_configuration_isolation",
    )
    leakage = sum(not bool(checks[name]) for name in leakage_names)
    return {
        "source": config.dataset_integrity_report.as_posix(),
        "source_sha256": _sha256(config.dataset_integrity_report),
        "manifest_sha256": evidence["manifest_sha256"],
        "deterministic_replay_rate": 1.0 if checks["deterministic_replay"] else 0.0,
        "split_leakage_findings": leakage,
        "paired_counterfactual_transition_present": checks[
            "paired_counterfactual_transition_present"
        ],
        "revalidated": False,
    }


def _arm_summary(
    config_path: Path,
    result: Mapping[str, Any],
    output_dir: Path,
) -> dict[str, Any]:
    checkpoint = json.loads(
        (output_dir / "checkpoint_manifest.json").read_text(encoding="utf-8")
    )
    return {
        "run_id": result["run_id"],
        "config": config_path.as_posix(),
        "config_sha256": _sha256(config_path),
        "selected_step": result["best_step"],
        "model_class": result["model_class"],
        "parameters": result["parameters"],
        "metrics": result["best_validation"],
        "checkpoint": {
            "sha256": checkpoint["checkpoint_sha256"],
            "weights_kind": checkpoint["weights_kind"],
            "promoted": False,
        },
        "runtime_seconds": result["runtime_seconds"],
        "peak_memory_bytes": result["peak_memory_bytes"],
    }


def run_h013_development_suite(
    config_path: str | Path,
    output_dir: str | Path,
    report_path: str | Path,
) -> dict[str, Any]:
    """Run all H013 development arms and apply the registered gate once."""

    suite = H013SuiteConfig.from_yaml(config_path)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    results: dict[H013Arm, dict[str, Any]] = {}
    arm_outputs: dict[H013Arm, Path] = {}
    for arm, arm_config in suite.arms.items():
        arm_output = output / arm.value
        results[arm] = run_h013_preflight(arm_config, arm_output)
        arm_outputs[arm] = arm_output
    factorized = results[H013Arm.FACTORIZED]
    direct = results[H013Arm.DIRECT]
    delta_ratio = _metric(
        factorized, "intervention_state_delta_nrmse"
    ) / _metric(direct, "intervention_state_delta_nrmse")
    rollout_ratio = _metric(factorized, "four_step_rollout_nrmse") / _metric(
        direct, "four_step_rollout_nrmse"
    )
    no_op_ratio = _metric(factorized, "no_op_state_nrmse") / _metric(
        direct, "no_op_state_nrmse"
    )
    effect_ratio = _metric(factorized, "intervention_effect_nrmse") / _metric(
        direct, "intervention_effect_nrmse"
    )
    identity_residual = _metric(
        factorized, "factorized_identity_maximum_absolute_residual"
    )
    compared_parameters = {
        int(factorized["parameters"]),
        int(direct["parameters"]),
    }
    gate_values = [
        delta_ratio,
        rollout_ratio,
        no_op_ratio,
        effect_ratio,
        identity_residual,
    ]
    all_metrics_finite = all(math.isfinite(value) for value in gate_values)
    integrity = _integrity(suite)
    passed = (
        delta_ratio <= suite.delta_ratio_maximum
        and rollout_ratio <= suite.rollout_ratio_maximum
        and no_op_ratio <= suite.no_op_ratio_maximum
        and effect_ratio <= suite.effect_ratio_maximum
        and identity_residual <= suite.identity_residual_maximum
        and len(compared_parameters) == 1
        and all_metrics_finite
        and integrity["deterministic_replay_rate"] == 1.0
        and integrity["split_leakage_findings"] == 0
        and bool(integrity["paired_counterfactual_transition_present"])
    )
    generator = GeneratedWorldDatasetConfig.from_yaml(suite.generator_config)
    random_baseline = evaluate_legal_random_interventions(generator).to_dict()
    report: dict[str, Any] = {
        "schema_version": 1,
        "preflight_id": "CHM-W-H013-DEVELOPMENT-001",
        "hypothesis_id": suite.hypothesis_id,
        "trial_id": suite.trial_id,
        "status": "completed_development_preflight",
        "scientific_result": False,
        "registered_trial_executed": False,
        "seed": suite.seed,
        "arms": {
            arm.value: _arm_summary(suite.arms[arm], results[arm], arm_outputs[arm])
            for arm in H013Arm
        },
        "legal_random_intervention": random_baseline,
        "comparisons": {
            "factorized_vs_matched_direct": {
                "intervention_state_delta_nrmse_ratio": delta_ratio,
                "four_step_rollout_nrmse_ratio": rollout_ratio,
                "no_op_state_nrmse_ratio": no_op_ratio,
                "intervention_effect_nrmse_ratio": effect_ratio,
            },
            "factorized_vs_factual_only_reference": {
                "intervention_state_delta_nrmse_ratio": _metric(
                    factorized, "intervention_state_delta_nrmse"
                )
                / _metric(
                    results[H013Arm.FACTUAL_ONLY],
                    "intervention_state_delta_nrmse",
                ),
                "four_step_rollout_nrmse_ratio": _metric(
                    factorized, "four_step_rollout_nrmse"
                )
                / _metric(
                    results[H013Arm.FACTUAL_ONLY],
                    "four_step_rollout_nrmse",
                ),
            },
        },
        "development_gate": {
            "intervention_state_delta_nrmse_ratio": delta_ratio,
            "intervention_state_delta_nrmse_ratio_maximum": suite.delta_ratio_maximum,
            "four_step_rollout_nrmse_ratio": rollout_ratio,
            "four_step_rollout_nrmse_ratio_maximum": suite.rollout_ratio_maximum,
            "no_op_state_nrmse_ratio": no_op_ratio,
            "no_op_state_nrmse_ratio_maximum": suite.no_op_ratio_maximum,
            "intervention_effect_nrmse_ratio": effect_ratio,
            "intervention_effect_nrmse_ratio_maximum": suite.effect_ratio_maximum,
            "factorized_identity_maximum_absolute_residual": identity_residual,
            "factorized_identity_maximum_absolute_residual_maximum": (
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
            "freeze_H013_hyperparameters_and_open_registered_validation_seeds"
            if passed
            else "do_not_open_H013_frozen_validation"
        ),
        "checkpoint_promoted": False,
        "opened_splits": ["train", "validation"],
        "frozen_validation_seeds_opened": False,
        "test_metrics_opened": False,
        "environment": factorized["environment"],
        "claim_boundary": (
            "Development-only generated-simulator evidence. Frozen validation "
            "and every test split remained sealed; no real-world causal, business, "
            "language-independence or production claim."
        ),
    }
    report_file = Path(report_path)
    report_file.parent.mkdir(parents=True, exist_ok=True)
    report_file.write_bytes(
        (json.dumps(report, indent=2, sort_keys=True) + "\n").encode("utf-8")
    )
    return report
