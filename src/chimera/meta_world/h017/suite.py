"""End-to-end support-preserving candidate-generation gate for CHM-W-H017."""

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
from chimera.meta_world.h015.evaluation import (
    realized_candidate_effect,
    slice_sequence_sample,
    uniform_legal_pool,
)
from chimera.meta_world.h015.search import (
    InterventionCandidate,
    PredictionFunction,
    SearchResult,
    quality_diversity_search,
)
from chimera.meta_world.h016.config import H016BackboneConfig, H016SuiteConfig
from chimera.meta_world.h016.evaluation import H016CandidatePredictor
from chimera.meta_world.h016.preflight import run_h016_backbone_preflight
from chimera.meta_world.h016.run import run_h016_ranking_training
from chimera.meta_world.h017.config import H017SuiteConfig
from chimera.meta_world.h017.pool import (
    SupportPoolDiagnostics,
    balanced_support_pool,
    support_pool_diagnostics,
)
from chimera.meta_world.h017.rerank import one_pass_qd_rerank
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
            raise RuntimeError("H017 replay issued more score calls than search")
        expected, means, deviations = records[replay_index]
        replay_index += 1
        if candidates != expected:
            raise RuntimeError("H017 adaptive replay generated different candidates")
        return means.copy(), deviations.copy()

    return record, replay


def _integrity(config: H017SuiteConfig) -> dict[str, Any]:
    evidence = json.loads(config.integrity_report.read_text(encoding="utf-8"))
    if evidence.get("status") != "passed":
        raise ValueError("H017 reused WG4 evidence is not passing")
    if evidence.get("dataset_config_sha256") != _sha256(config.generator_config):
        raise ValueError("H017 WG4 evidence does not match its generator")
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
        record = item.to_dict()
        record["rank_logit"] = record.pop("predicted_effect_mean")
        record.pop("predicted_effect_std")
        record["realized_effect"] = effect
        records.append(record)
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


def _boundary_rate(candidates: list[InterventionCandidate]) -> float:
    if not candidates:
        return 0.0
    return sum(
        item.magnitude in {0.0, 1.0} or item.control in {-1.0, 1.0}
        for item in candidates
    ) / len(candidates)


def run_h017_development_suite(
    config_path: str | Path,
    output_dir: str | Path,
    report_path: str | Path,
) -> dict[str, Any]:
    """Retrain the H016 critic and compare finite-pool versus adaptive search."""

    suite = H017SuiteConfig.from_yaml(config_path)
    critic_suite = H016SuiteConfig.from_yaml(suite.critic_suite_config)
    output = Path(output_dir)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("H017 development output must be empty")
    output.mkdir(parents=True, exist_ok=True)
    backbone_output = output / "backbone"
    backbone_result = run_h016_backbone_preflight(
        critic_suite.backbone_config,
        backbone_output,
    )
    backbone_config = H016BackboneConfig.from_yaml(critic_suite.backbone_config)
    runtime = backbone_config.paired_runtime.runtime
    device = resolve_device(runtime.training.device)
    use_autocast = runtime.training.precision == "bfloat16" and device.type == "cuda"
    ranker, ranking_result = run_h016_ranking_training(
        critic_suite,
        backbone_checkpoint=backbone_output / "checkpoint.pt",
        output_dir=output / "ranking",
        device=device,
        use_autocast=use_autocast,
    )
    predictor = H016CandidatePredictor(
        model=ranker,
        device=device,
        use_autocast=use_autocast,
    )
    generator = GeneratedWorldDatasetConfig.from_yaml(suite.generator_config)
    pipeline = WorldGenerationPipeline(generator)
    prediction_step = critic_suite.ranking.prediction_step
    pool_regrets: list[float] = []
    adaptive_regrets: list[float] = []
    random_regrets: list[float] = []
    pool_effects_best: list[float] = []
    adaptive_effects_best: list[float] = []
    random_effects_best: list[float] = []
    pool_archive_coverages: list[float] = []
    pool_unique_pairs: list[int] = []
    adaptive_archive_coverages: list[float] = []
    adaptive_unique_pairs: list[int] = []
    pool_diagnostics: list[SupportPoolDiagnostics] = []
    pool_replay_matches = 0
    search_replay_matches = 0
    model_budget_matches = 0
    simulator_budget_matches = 0
    legal_candidates = 0
    total_candidates = 0
    pool_selected_candidates: list[InterventionCandidate] = []
    adaptive_selected_candidates: list[InterventionCandidate] = []
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
            context_steps=critic_suite.ranking.context_steps,
        )
        final_step = int(window.time_mask[0].sum().item()) - 1
        objects = int(window.slot_mask[0, final_step].sum().item())
        support_seed = (
            suite.seed
            + state_index * suite.support_pool.seed_stride
            + suite.support_pool.seed_offset
        )
        support = balanced_support_pool(
            objects=objects,
            count=suite.support_pool.candidates_per_state,
            seed=support_seed,
        )
        replay_support = balanced_support_pool(
            objects=objects,
            count=suite.support_pool.candidates_per_state,
            seed=support_seed,
        )
        pool_replay_matches += int(support == replay_support)
        diagnostics = support_pool_diagnostics(support)
        pool_diagnostics.append(diagnostics)
        pool_scores, _ = predictor.predict_rank(window, support)
        pool_search = one_pass_qd_rerank(
            support,
            pool_scores,
            executions=suite.pool_reranking.simulator_executions_per_state,
        )
        pool_replay = one_pass_qd_rerank(
            replay_support,
            pool_scores.copy(),
            executions=suite.pool_reranking.simulator_executions_per_state,
        )

        def predict_adaptive(
            candidates: tuple[InterventionCandidate, ...],
            state_window: Any = window,
        ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
            return predictor.predict_rank(state_window, candidates)

        adaptive_predict, adaptive_replay_predict = _prediction_replay_audit(
            predict_adaptive
        )
        adaptive_seed = (
            suite.seed
            + state_index * suite.support_pool.seed_stride
            + suite.adaptive_seed_offset
        )
        adaptive_search = quality_diversity_search(
            objects=objects,
            seed=adaptive_seed,
            config=suite.adaptive_search,
            uncertainty_beta=0.0,
            predict=adaptive_predict,
        )
        adaptive_replay = quality_diversity_search(
            objects=objects,
            seed=adaptive_seed,
            config=suite.adaptive_search,
            uncertainty_beta=0.0,
            predict=adaptive_replay_predict,
        )
        search_replay_matches += int(
            pool_search.to_dict() == pool_replay.to_dict()
            and adaptive_search.to_dict() == adaptive_replay.to_dict()
        )
        oracle = uniform_legal_pool(
            objects=objects,
            count=suite.oracle_pool_candidates_per_state,
            seed=(
                suite.seed
                + state_index * suite.support_pool.seed_stride
                + suite.oracle_seed_offset
            ),
        )
        oracle_effects = [
            realized_candidate_effect(
                generator,
                trajectory,
                prediction_step=prediction_step,
                candidate=candidate,
            )
            for candidate in oracle
        ]
        oracle_best = max(oracle_effects)
        random_selected = support[: suite.random_candidates_per_state]
        random_selected_effects = [
            realized_candidate_effect(
                generator,
                trajectory,
                prediction_step=prediction_step,
                candidate=candidate,
            )
            for candidate in random_selected
        ]
        pool_records, selected_pool_effects = _realized_selected(
            generator,
            trajectory,
            prediction_step=prediction_step,
            result=pool_search,
        )
        adaptive_records, selected_adaptive_effects = _realized_selected(
            generator,
            trajectory,
            prediction_step=prediction_step,
            result=adaptive_search,
        )
        model_budget_matches += int(
            pool_search.model_scores == suite.support_pool.candidates_per_state
            and adaptive_search.model_scores
            == suite.adaptive_search.model_scores_per_state
        )
        simulator_budget_matches += int(
            len(selected_pool_effects)
            == suite.pool_reranking.simulator_executions_per_state
            and len(selected_adaptive_effects)
            == suite.adaptive_search.simulator_executions_per_state
            and len(random_selected_effects) == suite.random_candidates_per_state
        )
        pool_best = max(selected_pool_effects)
        adaptive_best = max(selected_adaptive_effects)
        random_best = max(random_selected_effects)
        pool_regrets.append(max(oracle_best - pool_best, 0.0))
        adaptive_regrets.append(max(oracle_best - adaptive_best, 0.0))
        random_regrets.append(max(oracle_best - random_best, 0.0))
        pool_effects_best.append(pool_best)
        adaptive_effects_best.append(adaptive_best)
        random_effects_best.append(random_best)
        maximum_cells = objects * (objects - 1) * 4
        pool_archive_coverages.append(pool_search.archive_cells / maximum_cells)
        pool_unique_pairs.append(pool_search.unique_source_target_pairs)
        adaptive_archive_coverages.append(
            adaptive_search.archive_cells / maximum_cells
        )
        adaptive_unique_pairs.append(adaptive_search.unique_source_target_pairs)
        selected_pool = [item.candidate for item in pool_search.selected]
        selected_adaptive = [item.candidate for item in adaptive_search.selected]
        pool_selected_candidates.extend(selected_pool)
        adaptive_selected_candidates.extend(selected_adaptive)
        selected = selected_pool + selected_adaptive + list(random_selected)
        total_candidates += len(selected)
        legal_candidates += sum(
            item.source_slot != item.target_slot
            and 0.0 <= item.magnitude <= 1.0
            and -1.0 <= item.control <= 1.0
            for item in selected
        )
        state_records.append(
            {
                "state_index": state_index,
                "trajectory_index": trajectory_index,
                "objects": objects,
                "support_seed": support_seed,
                "support_pool": diagnostics.to_dict(),
                "oracle_best_effect": oracle_best,
                "pool_reranking": pool_records,
                "adaptive_CEM": adaptive_records,
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
    pool_regret = float(np.mean(pool_regrets))
    adaptive_regret = float(np.mean(adaptive_regrets))
    random_regret = float(np.mean(random_regrets))
    ratio_random = _safe_ratio(pool_regret, random_regret)
    ratio_adaptive = _safe_ratio(pool_regret, adaptive_regret)
    legal_rate = legal_candidates / max(total_candidates, 1)
    pool_replay_rate = pool_replay_matches / suite.evaluation_states
    search_replay_rate = search_replay_matches / suite.evaluation_states
    model_budget_match_rate = model_budget_matches / suite.evaluation_states
    simulator_budget_match_rate = simulator_budget_matches / suite.evaluation_states
    pool_boundary_rate = float(
        np.mean([item.exact_continuous_boundary_rate for item in pool_diagnostics])
    )
    pool_unique_rate = float(
        np.mean([item.unique_vector_rate for item in pool_diagnostics])
    )
    pair_discrepancy = max(item.pair_count_discrepancy for item in pool_diagnostics)
    selected_pool_boundary_rate = _boundary_rate(pool_selected_candidates)
    selected_adaptive_boundary_rate = _boundary_rate(adaptive_selected_candidates)
    selected_pool_magnitude = float(
        np.mean([item.magnitude for item in pool_selected_candidates])
    )
    selected_adaptive_magnitude = float(
        np.mean([item.magnitude for item in adaptive_selected_candidates])
    )
    training_replay = float(
        ranking_result["deterministic_training_candidate_replay_rate"]
    )
    integrity = _integrity(suite)
    numeric = [
        pool_regret,
        adaptive_regret,
        random_regret,
        ratio_random,
        ratio_adaptive,
        legal_rate,
        pool_replay_rate,
        search_replay_rate,
        model_budget_match_rate,
        simulator_budget_match_rate,
        pool_boundary_rate,
        pool_unique_rate,
        selected_pool_boundary_rate,
        selected_adaptive_boundary_rate,
        selected_pool_magnitude,
        selected_adaptive_magnitude,
        *pool_effects_best,
        *adaptive_effects_best,
        *random_effects_best,
    ]
    finite = all(math.isfinite(value) for value in numeric)
    passed = (
        ratio_random <= suite.random_regret_ratio_maximum
        and ratio_adaptive <= suite.adaptive_regret_ratio_maximum
        and legal_rate == 1.0
        and pool_replay_rate == 1.0
        and pool_boundary_rate == 0.0
        and pool_unique_rate == 1.0
        and pair_discrepancy <= suite.support_pool.maximum_pair_count_discrepancy
        and search_replay_rate == 1.0
        and model_budget_match_rate == 1.0
        and simulator_budget_match_rate == 1.0
        and training_replay == 1.0
        and bool(ranking_result["backbone_unchanged"])
        and finite
        and integrity["deterministic_replay_rate"] == 1.0
        and integrity["split_leakage_findings"] == 0
    )
    report: dict[str, Any] = {
        "schema_version": 1,
        "preflight_id": "CHM-W-H017-DEVELOPMENT-001",
        "hypothesis_id": "CHM-W-H017",
        "trial_id": "CHM-W-T017",
        "status": "completed_development_preflight",
        "scientific_result": False,
        "registered_trial_executed": False,
        "seed": suite.seed,
        "critic": {
            "source_protocol": suite.critic_suite_config.as_posix(),
            "source_protocol_sha256": _sha256(suite.critic_suite_config),
            "weights_shared_between_search_arms": True,
            "backbone": _backbone_summary(
                critic_suite.backbone_config,
                backbone_output,
                backbone_result,
            ),
            "ranking_training": ranking_result,
        },
        "candidate_generation": {
            "evaluation_states": suite.evaluation_states,
            "support_pool_candidates_per_state": (
                suite.support_pool.candidates_per_state
            ),
            "oracle_pool_candidates_per_state": (
                suite.oracle_pool_candidates_per_state
            ),
            "model_scores_per_state_each_learned_arm": 256,
            "simulator_executions_per_state_each_arm": 8,
            "search_replay_additional_model_scores_per_state": 0,
            "runtime_seconds": time.perf_counter() - search_started,
            "states": state_records,
        },
        "metrics": {
            "support_pool_mean_regret": pool_regret,
            "adaptive_CEM_mean_regret": adaptive_regret,
            "legal_random_mean_regret": random_regret,
            "support_pool_regret_ratio_vs_random": ratio_random,
            "support_pool_regret_ratio_vs_adaptive_CEM": ratio_adaptive,
            "support_pool_mean_best_effect": float(np.mean(pool_effects_best)),
            "adaptive_CEM_mean_best_effect": float(
                np.mean(adaptive_effects_best)
            ),
            "legal_random_mean_best_effect": float(np.mean(random_effects_best)),
            "support_pool_selected_magnitude_mean": selected_pool_magnitude,
            "adaptive_CEM_selected_magnitude_mean": selected_adaptive_magnitude,
            "support_pool_selected_exact_boundary_rate": (
                selected_pool_boundary_rate
            ),
            "adaptive_CEM_selected_exact_boundary_rate": (
                selected_adaptive_boundary_rate
            ),
            "support_pool_mean_archive_cell_coverage": float(
                np.mean(pool_archive_coverages)
            ),
            "adaptive_CEM_mean_archive_cell_coverage": float(
                np.mean(adaptive_archive_coverages)
            ),
            "support_pool_mean_unique_source_target_pairs": float(
                np.mean(pool_unique_pairs)
            ),
            "adaptive_CEM_mean_unique_source_target_pairs": float(
                np.mean(adaptive_unique_pairs)
            ),
        },
        "development_gate": {
            "regret_ratio_vs_legal_random": ratio_random,
            "regret_ratio_vs_legal_random_maximum": (
                suite.random_regret_ratio_maximum
            ),
            "regret_ratio_vs_H016_adaptive_CEM": ratio_adaptive,
            "regret_ratio_vs_H016_adaptive_CEM_maximum": (
                suite.adaptive_regret_ratio_maximum
            ),
            "legal_action_rate": legal_rate,
            "support_pool_replay_rate": pool_replay_rate,
            "support_pool_exact_boundary_rate": pool_boundary_rate,
            "support_pool_unique_vector_rate": pool_unique_rate,
            "support_pool_pair_count_discrepancy": pair_discrepancy,
            "model_score_budget_match_rate": model_budget_match_rate,
            "simulator_execution_budget_match_rate": simulator_budget_match_rate,
            "deterministic_training_candidate_replay_rate": training_replay,
            "deterministic_search_replay_rate": search_replay_rate,
            "deterministic_dataset_replay_rate": integrity[
                "deterministic_replay_rate"
            ],
            "backbone_unchanged_during_ranking": bool(
                ranking_result["backbone_unchanged"]
            ),
            "split_leakage_findings": integrity["split_leakage_findings"],
            "all_metrics_finite": finite,
            "test_metrics_opened": False,
            "passed": passed,
        },
        "dataset_integrity": integrity,
        "decision": (
            "freeze_H017_generator_and_open_registered_validation_seeds"
            if passed
            else "do_not_open_H017_frozen_validation"
        ),
        "checkpoint_promoted": False,
        "opened_splits": ["train", "validation"],
        "frozen_validation_seeds_opened": False,
        "test_metrics_opened": False,
        "environment": {
            **backbone_result["environment"],
            "ranking_peak_memory_bytes": ranking_result["peak_memory_bytes"],
        },
        "claim_boundary": (
            "Development-only generated-world support-pool evidence; no "
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
