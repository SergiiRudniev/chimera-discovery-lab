"""End-to-end development candidate-generation gate for CHM-W-H015."""

from __future__ import annotations

import hashlib
import json
import math
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from numpy.typing import NDArray

from chimera.meta_world.generators import (
    GeneratedWorldDatasetConfig,
    SplitName,
    WorldGenerationPipeline,
    WorldTrajectory,
)
from chimera.meta_world.h002.windows import (
    make_transition_window,
    materialize_sequence_sample,
)
from chimera.meta_world.h014.model import (
    ResponseConditionedEffectWorldModel,
    ResponseSource,
)
from chimera.meta_world.h015.config import (
    H015BackboneConfig,
    H015SuiteConfig,
)
from chimera.meta_world.h015.evaluation import (
    load_candidate_predictor,
    realized_candidate_effect,
    slice_sequence_sample,
    uniform_legal_pool,
)
from chimera.meta_world.h015.preflight import run_h015_backbone_preflight
from chimera.meta_world.h015.search import (
    InterventionCandidate,
    PredictionFunction,
    SearchResult,
    quality_diversity_search,
)
from chimera.meta_world.trainer import resolve_device


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _safe_ratio(numerator: float, denominator: float) -> float:
    if denominator > 1e-12:
        return numerator / denominator
    return 0.0 if numerator <= 1e-12 else math.inf


def _prediction_replay_audit(
    predict: PredictionFunction,
) -> tuple[PredictionFunction, PredictionFunction]:
    """Record model outputs once and require replay to issue identical queries."""

    records: list[
        tuple[
            tuple[InterventionCandidate, ...],
            NDArray[np.float64],
            NDArray[np.float64],
        ]
    ] = []
    replay_index = 0

    def record(
        candidates: tuple[InterventionCandidate, ...],
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        means, deviations = predict(candidates)
        records.append((candidates, means.copy(), deviations.copy()))
        return means, deviations

    def replay(
        candidates: tuple[InterventionCandidate, ...],
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        nonlocal replay_index
        if replay_index >= len(records):
            raise RuntimeError("H015 replay issued more scoring calls than the search")
        expected, means, deviations = records[replay_index]
        replay_index += 1
        if candidates != expected:
            raise RuntimeError("H015 search replay generated different candidates")
        return means.copy(), deviations.copy()

    return record, replay


def _integrity(config: H015SuiteConfig) -> dict[str, Any]:
    evidence = json.loads(config.integrity_report.read_text(encoding="utf-8"))
    if evidence.get("status") != "passed":
        raise ValueError("H015 reused WG4 evidence is not passing")
    if evidence.get("dataset_config_sha256") != _sha256(config.generator_config):
        raise ValueError("H015 WG4 evidence does not match its generator")
    checks = evidence["checks"]
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
        "revalidated": False,
    }


def _realized_selected(
    config: GeneratedWorldDatasetConfig,
    trajectory: WorldTrajectory,
    *,
    prediction_step: int,
    result: SearchResult,
) -> tuple[list[dict[str, Any]], list[float]]:
    records: list[dict[str, Any]] = []
    effects: list[float] = []
    for item in result.selected:
        effect = realized_candidate_effect(
            config,
            trajectory,
            prediction_step=prediction_step,
            candidate=item.candidate,
        )
        records.append({**item.to_dict(), "realized_effect": effect})
        effects.append(effect)
    return records, effects


def _backbone_summary(
    config_path: Path,
    output: Path,
    result: dict[str, Any],
) -> dict[str, Any]:
    manifest = json.loads(
        (output / "checkpoint_manifest.json").read_text(encoding="utf-8")
    )
    return {
        "run_id": result["run_id"],
        "config": config_path.as_posix(),
        "config_sha256": _sha256(config_path),
        "selected_step": result["best_step"],
        "parameters": result["parameters"],
        "metrics": result["best_validation"],
        "checkpoint": {
            "sha256": manifest["checkpoint_sha256"],
            "weights_kind": manifest["weights_kind"],
            "promoted": False,
        },
        "runtime_seconds": result["runtime_seconds"],
        "peak_memory_bytes": result["peak_memory_bytes"],
    }


def run_h015_development_suite(
    config_path: str | Path,
    output_dir: str | Path,
    report_path: str | Path,
) -> dict[str, Any]:
    """Train the backbone, generate candidates and apply the regret gate."""

    suite = H015SuiteConfig.from_yaml(config_path)
    output = Path(output_dir)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("H015 development output must be empty")
    output.mkdir(parents=True, exist_ok=True)
    backbone_output = output / "backbone"
    backbone_result = run_h015_backbone_preflight(
        suite.backbone_config,
        backbone_output,
    )
    backbone_config = H015BackboneConfig.from_yaml(suite.backbone_config)
    runtime = backbone_config.paired_runtime.runtime
    device = resolve_device(runtime.training.device)
    model = ResponseConditionedEffectWorldModel(
        runtime.model,
        response_source=ResponseSource.FACTUAL_RESIDUAL,
    )
    predictor = load_candidate_predictor(
        model,
        str(backbone_output / "checkpoint.pt"),
        device=device,
        use_autocast=(
            runtime.training.precision == "bfloat16" and device.type == "cuda"
        ),
    )
    generator = GeneratedWorldDatasetConfig.from_yaml(suite.generator_config)
    pipeline = WorldGenerationPipeline(generator)
    prediction_step = runtime.model.context_steps - 1
    uncertainty_regrets: list[float] = []
    mean_regrets: list[float] = []
    random_regrets: list[float] = []
    uncertainty_effects: list[float] = []
    mean_effects: list[float] = []
    random_effects: list[float] = []
    archive_coverages: list[float] = []
    unique_pairs: list[int] = []
    replay_matches = 0
    model_budget_matches = 0
    simulator_budget_matches = 0
    legal_candidates = 0
    total_candidates = 0
    state_records: list[dict[str, Any]] = []
    search_started = time.perf_counter()
    for state_index in range(suite.evaluation_states):
        trajectory_index = state_index * generator.views_per_mechanism
        trajectory = pipeline.materialize(SplitName.VALIDATION, trajectory_index)
        grouped = materialize_sequence_sample(
            pipeline,
            SplitName.VALIDATION,
            start_index=trajectory_index,
            batch_size=generator.views_per_mechanism,
        )
        sample = slice_sequence_sample(grouped, 0)
        window = make_transition_window(
            sample,
            prediction_step=prediction_step,
            context_steps=runtime.model.context_steps,
        )
        final_step = int(window.time_mask[0].sum().item()) - 1
        objects = int(window.slot_mask[0, final_step].sum().item())
        pool = uniform_legal_pool(
            objects=objects,
            count=suite.oracle_pool_candidates_per_state,
            seed=suite.seed + state_index * 1_000_003 + 11,
        )
        pool_effects = [
            realized_candidate_effect(
                generator,
                trajectory,
                prediction_step=prediction_step,
                candidate=candidate,
            )
            for candidate in pool
        ]
        oracle_best = max(pool_effects)
        random_selected = pool[: suite.random_candidates_per_state]
        random_selected_effects = pool_effects[: suite.random_candidates_per_state]
        search_seed = suite.seed + state_index * 1_000_003 + 29

        def predict(
            candidates: tuple[InterventionCandidate, ...],
            state_window: Any = window,
        ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
            return predictor.predict(state_window, candidates)

        uncertainty_predict, uncertainty_replay_predict = _prediction_replay_audit(
            predict
        )
        uncertainty = quality_diversity_search(
            objects=objects,
            seed=search_seed,
            config=suite.search,
            uncertainty_beta=suite.uncertainty_beta,
            predict=uncertainty_predict,
        )
        uncertainty_replay = quality_diversity_search(
            objects=objects,
            seed=search_seed,
            config=suite.search,
            uncertainty_beta=suite.uncertainty_beta,
            predict=uncertainty_replay_predict,
        )
        mean_predict, mean_replay_predict = _prediction_replay_audit(predict)
        mean_only = quality_diversity_search(
            objects=objects,
            seed=search_seed,
            config=suite.search,
            uncertainty_beta=suite.mean_only_beta,
            predict=mean_predict,
        )
        mean_replay = quality_diversity_search(
            objects=objects,
            seed=search_seed,
            config=suite.search,
            uncertainty_beta=suite.mean_only_beta,
            predict=mean_replay_predict,
        )
        replay_matches += int(
            uncertainty.to_dict() == uncertainty_replay.to_dict()
            and mean_only.to_dict() == mean_replay.to_dict()
        )
        uncertainty_records, uncertainty_selected_effects = _realized_selected(
            generator,
            trajectory,
            prediction_step=prediction_step,
            result=uncertainty,
        )
        mean_records, mean_selected_effects = _realized_selected(
            generator,
            trajectory,
            prediction_step=prediction_step,
            result=mean_only,
        )
        model_budget_matches += int(
            uncertainty.model_scores == suite.search.model_scores_per_state
            and mean_only.model_scores == suite.search.model_scores_per_state
        )
        simulator_budget_matches += int(
            len(uncertainty_selected_effects)
            == suite.search.simulator_executions_per_state
            and len(mean_selected_effects)
            == suite.search.simulator_executions_per_state
            and len(random_selected_effects) == suite.random_candidates_per_state
        )
        uncertainty_best = max(uncertainty_selected_effects)
        mean_best = max(mean_selected_effects)
        random_best = max(random_selected_effects)
        uncertainty_regrets.append(max(oracle_best - uncertainty_best, 0.0))
        mean_regrets.append(max(oracle_best - mean_best, 0.0))
        random_regrets.append(max(oracle_best - random_best, 0.0))
        uncertainty_effects.append(uncertainty_best)
        mean_effects.append(mean_best)
        random_effects.append(random_best)
        maximum_cells = objects * (objects - 1) * 4
        archive_coverages.append(uncertainty.archive_cells / maximum_cells)
        unique_pairs.append(uncertainty.unique_source_target_pairs)
        selected_candidates = (
            [item.candidate for item in uncertainty.selected]
            + [item.candidate for item in mean_only.selected]
            + list(random_selected)
        )
        total_candidates += len(selected_candidates)
        legal_candidates += sum(
            item.source_slot != item.target_slot
            and 0.0 <= item.magnitude <= 1.0
            and -1.0 <= item.control <= 1.0
            for item in selected_candidates
        )
        state_records.append(
            {
                "state_index": state_index,
                "trajectory_index": trajectory_index,
                "objects": objects,
                "oracle_best_effect": oracle_best,
                "uncertainty_aware": uncertainty_records,
                "mean_only": mean_records,
                "legal_random": [
                    {**candidate.to_dict(), "realized_effect": effect}
                    for candidate, effect in zip(
                        random_selected,
                        random_selected_effects,
                        strict=True,
                    )
                ],
            }
        )
    uncertainty_regret = float(np.mean(uncertainty_regrets))
    mean_regret = float(np.mean(mean_regrets))
    random_regret = float(np.mean(random_regrets))
    ratio_random = _safe_ratio(uncertainty_regret, random_regret)
    ratio_mean = _safe_ratio(uncertainty_regret, mean_regret)
    legal_rate = legal_candidates / max(total_candidates, 1)
    replay_rate = replay_matches / suite.evaluation_states
    model_budget_match_rate = model_budget_matches / suite.evaluation_states
    simulator_budget_match_rate = simulator_budget_matches / suite.evaluation_states
    numeric = [
        uncertainty_regret,
        mean_regret,
        random_regret,
        ratio_random,
        ratio_mean,
        legal_rate,
        replay_rate,
        model_budget_match_rate,
        simulator_budget_match_rate,
        *uncertainty_effects,
        *mean_effects,
        *random_effects,
    ]
    finite = all(math.isfinite(value) for value in numeric)
    integrity = _integrity(suite)
    passed = (
        ratio_random <= suite.random_regret_ratio_maximum
        and ratio_mean <= suite.mean_regret_ratio_maximum
        and legal_rate == 1.0
        and replay_rate == 1.0
        and model_budget_match_rate == 1.0
        and simulator_budget_match_rate == 1.0
        and finite
        and integrity["deterministic_replay_rate"] == 1.0
        and integrity["split_leakage_findings"] == 0
    )
    report: dict[str, Any] = {
        "schema_version": 1,
        "preflight_id": "CHM-W-H015-DEVELOPMENT-001",
        "hypothesis_id": "CHM-W-H015",
        "trial_id": "CHM-W-T015",
        "status": "completed_development_preflight",
        "scientific_result": False,
        "registered_trial_executed": False,
        "seed": suite.seed,
        "backbone": _backbone_summary(
            suite.backbone_config,
            backbone_output,
            backbone_result,
        ),
        "candidate_generation": {
            "evaluation_states": suite.evaluation_states,
            "oracle_pool_candidates_per_state": suite.oracle_pool_candidates_per_state,
            "model_scores_per_state_each_search_arm": (
                suite.search.model_scores_per_state
            ),
            "simulator_executions_per_state_each_arm": (
                suite.search.simulator_executions_per_state
            ),
            "search_replay_additional_model_scores_per_state": 0,
            "runtime_seconds": time.perf_counter() - search_started,
            "states": state_records,
        },
        "metrics": {
            "uncertainty_aware_mean_regret": uncertainty_regret,
            "mean_only_mean_regret": mean_regret,
            "legal_random_mean_regret": random_regret,
            "uncertainty_regret_ratio_vs_random": ratio_random,
            "uncertainty_regret_ratio_vs_mean_only": ratio_mean,
            "uncertainty_mean_best_effect": float(np.mean(uncertainty_effects)),
            "mean_only_mean_best_effect": float(np.mean(mean_effects)),
            "legal_random_mean_best_effect": float(np.mean(random_effects)),
            "mean_archive_cell_coverage": float(np.mean(archive_coverages)),
            "mean_unique_source_target_pairs": float(np.mean(unique_pairs)),
        },
        "development_gate": {
            "regret_ratio_vs_legal_random": ratio_random,
            "regret_ratio_vs_legal_random_maximum": (
                suite.random_regret_ratio_maximum
            ),
            "regret_ratio_vs_mean_only_search": ratio_mean,
            "regret_ratio_vs_mean_only_search_maximum": (
                suite.mean_regret_ratio_maximum
            ),
            "legal_action_rate": legal_rate,
            "simulator_executions_per_state_each_arm": (
                suite.search.simulator_executions_per_state
            ),
            "model_scores_per_state_each_search_arm": (
                suite.search.model_scores_per_state
            ),
            "model_score_budget_match_rate": model_budget_match_rate,
            "simulator_execution_budget_match_rate": simulator_budget_match_rate,
            "deterministic_search_replay_rate": replay_rate,
            "deterministic_dataset_replay_rate": integrity[
                "deterministic_replay_rate"
            ],
            "split_leakage_findings": integrity["split_leakage_findings"],
            "all_metrics_finite": finite,
            "test_metrics_opened": False,
            "passed": passed,
        },
        "dataset_integrity": integrity,
        "decision": (
            "freeze_H015_search_and_open_registered_validation_seeds"
            if passed
            else "do_not_open_H015_frozen_validation"
        ),
        "checkpoint_promoted": False,
        "opened_splits": ["train", "validation"],
        "frozen_validation_seeds_opened": False,
        "test_metrics_opened": False,
        "environment": backbone_result["environment"],
        "claim_boundary": (
            "Development-only generated-world candidate-search evidence; no "
            "real-world causal, creative, business or production claim."
        ),
    }
    report_file = Path(report_path)
    report_file.parent.mkdir(parents=True, exist_ok=True)
    report_file.write_bytes(
        (json.dumps(report, indent=2, sort_keys=True) + "\n").encode("utf-8")
    )
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return report
