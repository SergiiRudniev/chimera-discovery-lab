"""Parameter-matched H014 development gate."""

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
from chimera.meta_world.h014.config import H014Arm, H014RunConfig
from chimera.meta_world.h014.preflight import run_h014_preflight


def _mapping(value: object, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{name} must be a mapping")
    return cast(Mapping[str, Any], value)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


@dataclass(frozen=True)
class H014SuiteConfig:
    """Frozen effect-head comparison and reused WG4 evidence."""

    seed: int
    frozen_validation_seeds: tuple[int, ...]
    generator_config: Path
    integrity_report: Path
    arms: Mapping[H014Arm, Path]
    effect_ratio_maximum: float
    rollout_ratio_maximum: float
    delta_ratio_maximum: float
    no_op_ratio_maximum: float
    coverage_minimum: float
    identity_maximum: float

    @classmethod
    def from_yaml(cls, path: str | Path) -> H014SuiteConfig:
        with Path(path).open("r", encoding="utf-8") as handle:
            values = _mapping(yaml.safe_load(handle), "H014 suite")
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
            "primary_metric",
            "development_gate",
        }
        if set(values) != expected:
            raise ValueError("H014 suite fields differ from the frozen schema")
        if (
            values["hypothesis_id"] != "CHM-W-H014"
            or values["trial_id"] != "CHM-W-T014"
            or values["mode"] != "development"
            or values["test_access"] != "sealed"
            or values["primary_metric"] != "intervention_effect_nrmse"
        ):
            raise ValueError("H014 suite identity or access boundary is invalid")
        seed = int(values["development_seed"])
        frozen = tuple(int(item) for item in values["frozen_validation_seeds"])
        if seed != 260946 or frozen != (260947, 260948, 260949):
            raise ValueError("H014 registered seeds are immutable")
        arms_raw = _mapping(values["arms"], "H014 arms")
        arms = {H014Arm(str(name)): Path(str(value)) for name, value in arms_raw.items()}
        if set(arms) != set(H014Arm):
            raise ValueError("H014 suite requires both matched arms")
        for arm, config_path in arms.items():
            run = H014RunConfig.from_yaml(config_path)
            if run.arm is not arm:
                raise ValueError("H014 arm config disagrees with suite")
        gate = _mapping(values["development_gate"], "H014 gate")
        expected_gate = {
            "intervention_effect_nrmse_ratio_vs_matched_control_maximum",
            "four_step_rollout_nrmse_ratio_vs_matched_control_maximum",
            "intervention_state_delta_nrmse_ratio_vs_matched_control_maximum",
            "no_op_state_nrmse_ratio_vs_matched_control_maximum",
            "intervention_effect_90_coverage_minimum",
            "outcome_counterfactual_identity_maximum_absolute_residual",
            "deterministic_replay_rate",
            "split_leakage_findings",
            "all_metrics_finite",
            "test_metrics_opened",
        }
        if set(gate) != expected_gate:
            raise ValueError("H014 gate fields differ from preregistration")
        if (
            float(gate["deterministic_replay_rate"]) != 1.0
            or int(gate["split_leakage_findings"]) != 0
            or not bool(gate["all_metrics_finite"])
            or bool(gate["test_metrics_opened"])
        ):
            raise ValueError("H014 invariant gates are invalid")
        result = cls(
            seed=seed,
            frozen_validation_seeds=frozen,
            generator_config=Path(str(values["generator_config"])),
            integrity_report=Path(str(values["dataset_integrity_report"])),
            arms=arms,
            effect_ratio_maximum=float(
                gate["intervention_effect_nrmse_ratio_vs_matched_control_maximum"]
            ),
            rollout_ratio_maximum=float(
                gate["four_step_rollout_nrmse_ratio_vs_matched_control_maximum"]
            ),
            delta_ratio_maximum=float(
                gate[
                    "intervention_state_delta_nrmse_ratio_vs_matched_control_maximum"
                ]
            ),
            no_op_ratio_maximum=float(
                gate["no_op_state_nrmse_ratio_vs_matched_control_maximum"]
            ),
            coverage_minimum=float(gate["intervention_effect_90_coverage_minimum"]),
            identity_maximum=float(
                gate["outcome_counterfactual_identity_maximum_absolute_residual"]
            ),
        )
        if (
            result.effect_ratio_maximum != 0.90
            or result.rollout_ratio_maximum != 1.00
            or result.delta_ratio_maximum != 1.00
            or result.no_op_ratio_maximum != 1.00
            or result.coverage_minimum != 0.85
            or result.identity_maximum != 0.000001
        ):
            raise ValueError("H014 numeric gates are immutable")
        return result


def _metric(result: Mapping[str, Any], name: str) -> float:
    value = _mapping(result["best_validation"], "H014 metrics")[name]
    if value is None:
        raise RuntimeError(f"H014 metric {name} is unavailable")
    return float(value)


def _integrity(config: H014SuiteConfig) -> dict[str, Any]:
    evidence = json.loads(config.integrity_report.read_text(encoding="utf-8"))
    if evidence.get("status") != "passed":
        raise ValueError("reused WG4 integrity evidence is not passing")
    if evidence.get("dataset_config_sha256") != _sha256(config.generator_config):
        raise ValueError("reused WG4 evidence does not match its generator")
    checks = _mapping(evidence["checks"], "WG4 checks")
    isolation = (
        "mechanism_id_isolation",
        "world_instance_isolation",
        "seed_isolation",
        "exact_configuration_isolation",
    )
    return {
        "source": config.integrity_report.as_posix(),
        "source_sha256": _sha256(config.integrity_report),
        "deterministic_replay_rate": 1.0 if checks["deterministic_replay"] else 0.0,
        "split_leakage_findings": sum(not bool(checks[name]) for name in isolation),
        "paired_counterfactual_transition_present": checks[
            "paired_counterfactual_transition_present"
        ],
        "revalidated": False,
    }


def _summary(path: Path, output: Path, result: Mapping[str, Any]) -> dict[str, Any]:
    checkpoint = json.loads(
        (output / "checkpoint_manifest.json").read_text(encoding="utf-8")
    )
    return {
        "run_id": result["run_id"],
        "config": path.as_posix(),
        "config_sha256": _sha256(path),
        "response_source": result["response_source"],
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


def run_h014_development_suite(
    config_path: str | Path,
    output_dir: str | Path,
    report_path: str | Path,
) -> dict[str, Any]:
    """Run both response sources and apply the H014 gate once."""

    suite = H014SuiteConfig.from_yaml(config_path)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    results: dict[H014Arm, dict[str, Any]] = {}
    outputs: dict[H014Arm, Path] = {}
    for arm, path in suite.arms.items():
        outputs[arm] = output / arm.value
        results[arm] = run_h014_preflight(path, outputs[arm])
    response = results[H014Arm.RESPONSE]
    control = results[H014Arm.CONTROL]
    effect_ratio = _metric(response, "intervention_effect_nrmse") / _metric(
        control, "intervention_effect_nrmse"
    )
    rollout_ratio = _metric(response, "four_step_rollout_nrmse") / _metric(
        control, "four_step_rollout_nrmse"
    )
    delta_ratio = _metric(response, "intervention_state_delta_nrmse") / _metric(
        control, "intervention_state_delta_nrmse"
    )
    no_op_ratio = _metric(response, "no_op_state_nrmse") / _metric(
        control, "no_op_state_nrmse"
    )
    coverage = _metric(response, "intervention_effect_90_coverage")
    identity = _metric(
        response,
        "outcome_counterfactual_identity_maximum_absolute_residual",
    )
    parameter_matched = int(response["parameters"]) == int(control["parameters"])
    finite = all(
        math.isfinite(value)
        for value in (effect_ratio, rollout_ratio, delta_ratio, no_op_ratio, coverage, identity)
    )
    integrity = _integrity(suite)
    passed = (
        effect_ratio <= suite.effect_ratio_maximum
        and rollout_ratio <= suite.rollout_ratio_maximum
        and delta_ratio <= suite.delta_ratio_maximum
        and no_op_ratio <= suite.no_op_ratio_maximum
        and coverage >= suite.coverage_minimum
        and identity <= suite.identity_maximum
        and parameter_matched
        and finite
        and integrity["deterministic_replay_rate"] == 1.0
        and integrity["split_leakage_findings"] == 0
    )
    generator = GeneratedWorldDatasetConfig.from_yaml(suite.generator_config)
    report: dict[str, Any] = {
        "schema_version": 1,
        "preflight_id": "CHM-W-H014-DEVELOPMENT-001",
        "hypothesis_id": "CHM-W-H014",
        "trial_id": "CHM-W-T014",
        "status": "completed_development_preflight",
        "scientific_result": False,
        "registered_trial_executed": False,
        "seed": suite.seed,
        "arms": {
            arm.value: _summary(suite.arms[arm], outputs[arm], results[arm])
            for arm in H014Arm
        },
        "legal_random_intervention": evaluate_legal_random_interventions(
            generator
        ).to_dict(),
        "comparison": {
            "intervention_effect_nrmse_ratio": effect_ratio,
            "four_step_rollout_nrmse_ratio": rollout_ratio,
            "intervention_state_delta_nrmse_ratio": delta_ratio,
            "no_op_state_nrmse_ratio": no_op_ratio,
        },
        "development_gate": {
            "intervention_effect_nrmse_ratio": effect_ratio,
            "intervention_effect_nrmse_ratio_maximum": suite.effect_ratio_maximum,
            "four_step_rollout_nrmse_ratio": rollout_ratio,
            "four_step_rollout_nrmse_ratio_maximum": suite.rollout_ratio_maximum,
            "intervention_state_delta_nrmse_ratio": delta_ratio,
            "intervention_state_delta_nrmse_ratio_maximum": suite.delta_ratio_maximum,
            "no_op_state_nrmse_ratio": no_op_ratio,
            "no_op_state_nrmse_ratio_maximum": suite.no_op_ratio_maximum,
            "intervention_effect_90_coverage": coverage,
            "intervention_effect_90_coverage_minimum": suite.coverage_minimum,
            "outcome_counterfactual_identity_maximum_absolute_residual": identity,
            "outcome_counterfactual_identity_maximum_absolute_residual_maximum": (
                suite.identity_maximum
            ),
            "parameter_count_matched": parameter_matched,
            "all_metrics_finite": finite,
            "deterministic_replay_rate": integrity["deterministic_replay_rate"],
            "split_leakage_findings": integrity["split_leakage_findings"],
            "test_metrics_opened": False,
            "passed": passed,
        },
        "dataset_integrity": integrity,
        "decision": (
            "freeze_H014_hyperparameters_and_open_registered_validation_seeds"
            if passed
            else "do_not_open_H014_frozen_validation"
        ),
        "checkpoint_promoted": False,
        "opened_splits": ["train", "validation"],
        "frozen_validation_seeds_opened": False,
        "test_metrics_opened": False,
        "environment": response["environment"],
        "claim_boundary": (
            "Development-only simulator evidence; frozen validation and all "
            "model test metrics remain sealed."
        ),
    }
    report_file = Path(report_path)
    report_file.parent.mkdir(parents=True, exist_ok=True)
    report_file.write_bytes(
        (json.dumps(report, indent=2, sort_keys=True) + "\n").encode("utf-8")
    )
    return report
