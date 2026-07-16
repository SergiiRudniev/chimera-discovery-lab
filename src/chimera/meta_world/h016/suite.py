"""End-to-end development action-ranking gate for CHM-W-H016."""

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
from chimera.meta_world.h016.evaluation import (
    H016CandidatePredictor,
    ndcg_at_k,
    spearman_rank_correlation,
)
from chimera.meta_world.h016.preflight import run_h016_backbone_preflight
from chimera.meta_world.h016.run import run_h016_ranking_training
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
            raise RuntimeError("H016 replay issued more score calls than search")
        expected, means, deviations = records[replay_index]
        replay_index += 1
        if candidates != expected:
            raise RuntimeError("H016 search replay generated different candidates")
        return means.copy(), deviations.copy()

    return record, replay


def _integrity(config: H016SuiteConfig) -> dict[str, Any]:
    evidence = json.loads(config.integrity_report.read_text(encoding="utf-8"))
    if evidence.get("status") != "passed":
        raise ValueError("H016 reused WG4 evidence is not passing")
    if evidence.get("dataset_config_sha256") != _sha256(config.generator_config):
        raise ValueError("H016 WG4 evidence does not match its generator")
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
    rank_semantics: bool,
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
        if rank_semantics:
            record["rank_logit"] = record.pop("predicted_effect_mean")
            record.pop("predicted_effect_std")
        record["realized_effect"] = effect
        records.append(record)
        effects.append(effect)
    return records, effects


def _backbone_summary(
    config_path: Path,
    output: Path,
    result: dict[str, object],
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


def run_h016_development_suite(
    config_path: str | Path,
    output_dir: str | Path,
    report_path: str | Path,
) -> dict[str, Any]:
    """Train the frozen-backbone critic and apply the H016 regret gate."""

    suite = H016SuiteConfig.from_yaml(config_path)
    output = Path(output_dir)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("H016 development output must be empty")
    output.mkdir(parents=True, exist_ok=True)
    backbone_output = output / "backbone"
    backbone_result = run_h016_backbone_preflight(
        suite.backbone_config,
        backbone_output,
    )
    backbone_config = H016BackboneConfig.from_yaml(suite.backbone_config)
    runtime = backbone_config.paired_runtime.runtime
    device = resolve_device(runtime.training.device)
    use_autocast = runtime.training.precision == "bfloat16" and device.type == "cuda"
    ranker, ranking_result = run_h016_ranking_training(
        suite,
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
    prediction_step = suite.ranking.prediction_step
    rank_regrets: list[float] = []
    pointwise_regrets: list[float] = []
    random_regrets: list[float] = []
    rank_effects: list[float] = []
    pointwise_effects: list[float] = []
    random_effects: list[float] = []
    rank_spearman: list[float] = []
    pointwise_spearman: list[float] = []
    rank_ndcg: list[float] = []
    pointwise_ndcg: list[float] = []
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
            context_steps=suite.ranking.context_steps,
        )
        final_step = int(window.time_mask[0].sum().item()) - 1
        objects = int(window.slot_mask[0, final_step].sum().item())
        pool = uniform_legal_pool(
            objects=objects,
            count=suite.oracle_pool_candidates_per_state,
            seed=suite.seed + state_index * 1_000_003 + 11,
        )
        pool_effects = np.asarray(
            [
                realized_candidate_effect(
                    generator,
                    trajectory,
                    prediction_step=prediction_step,
                    candidate=candidate,
                )
                for candidate in pool
            ],
            dtype=np.float64,
        )
        oracle_best = float(pool_effects.max())
        random_selected = pool[: suite.random_candidates_per_state]
        random_selected_effects = pool_effects[: suite.random_candidates_per_state]
        search_seed = suite.seed + state_index * 1_000_003 + 29

        def predict_rank(
            candidates: tuple[InterventionCandidate, ...],
            state_window: Any = window,
        ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
            return predictor.predict_rank(state_window, candidates)

        def predict_pointwise(
            candidates: tuple[InterventionCandidate, ...],
            state_window: Any = window,
        ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
            return predictor.predict_pointwise(state_window, candidates)

        rank_predict, rank_replay_predict = _prediction_replay_audit(predict_rank)
        rank_search = quality_diversity_search(
            objects=objects,
            seed=search_seed,
            config=suite.search,
            uncertainty_beta=0.0,
            predict=rank_predict,
        )
        rank_replay = quality_diversity_search(
            objects=objects,
            seed=search_seed,
            config=suite.search,
            uncertainty_beta=0.0,
            predict=rank_replay_predict,
        )
        point_predict, point_replay_predict = _prediction_replay_audit(
            predict_pointwise
        )
        pointwise_search = quality_diversity_search(
            objects=objects,
            seed=search_seed,
            config=suite.search,
            uncertainty_beta=0.0,
            predict=point_predict,
        )
        pointwise_replay = quality_diversity_search(
            objects=objects,
            seed=search_seed,
            config=suite.search,
            uncertainty_beta=0.0,
            predict=point_replay_predict,
        )
        replay_matches += int(
            rank_search.to_dict() == rank_replay.to_dict()
            and pointwise_search.to_dict() == pointwise_replay.to_dict()
        )
        rank_records, rank_selected_effects = _realized_selected(
            generator,
            trajectory,
            prediction_step=prediction_step,
            result=rank_search,
            rank_semantics=True,
        )
        point_records, point_selected_effects = _realized_selected(
            generator,
            trajectory,
            prediction_step=prediction_step,
            result=pointwise_search,
            rank_semantics=False,
        )
        model_budget_matches += int(
            rank_search.model_scores == suite.search.model_scores_per_state
            and pointwise_search.model_scores == suite.search.model_scores_per_state
        )
        simulator_budget_matches += int(
            len(rank_selected_effects) == suite.search.simulator_executions_per_state
            and len(point_selected_effects)
            == suite.search.simulator_executions_per_state
            and len(random_selected_effects) == suite.random_candidates_per_state
        )
        rank_best = max(rank_selected_effects)
        pointwise_best = max(point_selected_effects)
        random_best = float(random_selected_effects.max())
        rank_regrets.append(max(oracle_best - rank_best, 0.0))
        pointwise_regrets.append(max(oracle_best - pointwise_best, 0.0))
        random_regrets.append(max(oracle_best - random_best, 0.0))
        rank_effects.append(rank_best)
        pointwise_effects.append(pointwise_best)
        random_effects.append(random_best)
        maximum_cells = objects * (objects - 1) * 4
        archive_coverages.append(rank_search.archive_cells / maximum_cells)
        unique_pairs.append(rank_search.unique_source_target_pairs)
        selected_candidates = (
            [item.candidate for item in rank_search.selected]
            + [item.candidate for item in pointwise_search.selected]
            + list(random_selected)
        )
        total_candidates += len(selected_candidates)
        legal_candidates += sum(
            item.source_slot != item.target_slot
            and 0.0 <= item.magnitude <= 1.0
            and -1.0 <= item.control <= 1.0
            for item in selected_candidates
        )

        diagnostic_pool = pool[: suite.ranking_diagnostic_candidates_per_state]
        rank_scores, _ = predictor.predict_rank(window, diagnostic_pool)
        point_scores, _ = predictor.predict_pointwise(window, diagnostic_pool)
        rank_spearman.append(spearman_rank_correlation(rank_scores, pool_effects))
        pointwise_spearman.append(
            spearman_rank_correlation(point_scores, pool_effects)
        )
        rank_ndcg.append(ndcg_at_k(rank_scores, pool_effects, k=8))
        pointwise_ndcg.append(ndcg_at_k(point_scores, pool_effects, k=8))
        state_records.append(
            {
                "state_index": state_index,
                "trajectory_index": trajectory_index,
                "objects": objects,
                "oracle_best_effect": oracle_best,
                "ranking_critic": rank_records,
                "pointwise_control": point_records,
                "legal_random": [
                    {**candidate.to_dict(), "realized_effect": float(effect)}
                    for candidate, effect in zip(
                        random_selected,
                        random_selected_effects,
                        strict=True,
                    )
                ],
                "diagnostics": {
                    "ranking_spearman": rank_spearman[-1],
                    "pointwise_spearman": pointwise_spearman[-1],
                    "ranking_ndcg_at_8": rank_ndcg[-1],
                    "pointwise_ndcg_at_8": pointwise_ndcg[-1],
                },
            }
        )
    rank_regret = float(np.mean(rank_regrets))
    pointwise_regret = float(np.mean(pointwise_regrets))
    random_regret = float(np.mean(random_regrets))
    ratio_random = _safe_ratio(rank_regret, random_regret)
    ratio_pointwise = _safe_ratio(rank_regret, pointwise_regret)
    legal_rate = legal_candidates / max(total_candidates, 1)
    replay_rate = replay_matches / suite.evaluation_states
    model_budget_match_rate = model_budget_matches / suite.evaluation_states
    simulator_budget_match_rate = simulator_budget_matches / suite.evaluation_states
    integrity = _integrity(suite)
    numeric = [
        rank_regret,
        pointwise_regret,
        random_regret,
        ratio_random,
        ratio_pointwise,
        legal_rate,
        replay_rate,
        model_budget_match_rate,
        simulator_budget_match_rate,
        *rank_effects,
        *pointwise_effects,
        *random_effects,
        *rank_spearman,
        *pointwise_spearman,
        *rank_ndcg,
        *pointwise_ndcg,
    ]
    finite = all(math.isfinite(value) for value in numeric)
    training_replay = float(
        ranking_result["deterministic_training_candidate_replay_rate"]
    )
    backbone_unchanged = bool(ranking_result["backbone_unchanged"])
    passed = (
        ratio_random <= suite.random_regret_ratio_maximum
        and ratio_pointwise <= suite.pointwise_regret_ratio_maximum
        and legal_rate == 1.0
        and replay_rate == 1.0
        and model_budget_match_rate == 1.0
        and simulator_budget_match_rate == 1.0
        and training_replay == 1.0
        and backbone_unchanged
        and finite
        and integrity["deterministic_replay_rate"] == 1.0
        and integrity["split_leakage_findings"] == 0
    )
    report: dict[str, Any] = {
        "schema_version": 1,
        "preflight_id": "CHM-W-H016-DEVELOPMENT-001",
        "hypothesis_id": "CHM-W-H016",
        "trial_id": "CHM-W-T016",
        "status": "completed_development_preflight",
        "scientific_result": False,
        "registered_trial_executed": False,
        "seed": suite.seed,
        "backbone": _backbone_summary(
            suite.backbone_config,
            backbone_output,
            backbone_result,
        ),
        "ranking_training": ranking_result,
        "candidate_generation": {
            "evaluation_states": suite.evaluation_states,
            "oracle_pool_candidates_per_state": (
                suite.oracle_pool_candidates_per_state
            ),
            "model_scores_per_state_each_search_arm": (
                suite.search.model_scores_per_state
            ),
            "simulator_executions_per_state_each_arm": (
                suite.search.simulator_executions_per_state
            ),
            "search_replay_additional_model_scores_per_state": 0,
            "ranking_diagnostic_scores_per_state_each_arm": (
                suite.ranking_diagnostic_candidates_per_state
            ),
            "runtime_seconds": time.perf_counter() - search_started,
            "states": state_records,
        },
        "metrics": {
            "ranking_critic_mean_regret": rank_regret,
            "pointwise_control_mean_regret": pointwise_regret,
            "legal_random_mean_regret": random_regret,
            "ranking_regret_ratio_vs_random": ratio_random,
            "ranking_regret_ratio_vs_pointwise": ratio_pointwise,
            "ranking_mean_best_effect": float(np.mean(rank_effects)),
            "pointwise_mean_best_effect": float(np.mean(pointwise_effects)),
            "legal_random_mean_best_effect": float(np.mean(random_effects)),
            "ranking_mean_spearman": float(np.mean(rank_spearman)),
            "pointwise_mean_spearman": float(np.mean(pointwise_spearman)),
            "ranking_mean_ndcg_at_8": float(np.mean(rank_ndcg)),
            "pointwise_mean_ndcg_at_8": float(np.mean(pointwise_ndcg)),
            "mean_archive_cell_coverage": float(np.mean(archive_coverages)),
            "mean_unique_source_target_pairs": float(np.mean(unique_pairs)),
        },
        "development_gate": {
            "regret_ratio_vs_legal_random": ratio_random,
            "regret_ratio_vs_legal_random_maximum": (
                suite.random_regret_ratio_maximum
            ),
            "regret_ratio_vs_H015_pointwise": ratio_pointwise,
            "regret_ratio_vs_H015_pointwise_maximum": (
                suite.pointwise_regret_ratio_maximum
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
            "deterministic_training_candidate_replay_rate": training_replay,
            "deterministic_search_replay_rate": replay_rate,
            "deterministic_dataset_replay_rate": integrity[
                "deterministic_replay_rate"
            ],
            "backbone_unchanged_during_ranking": backbone_unchanged,
            "split_leakage_findings": integrity["split_leakage_findings"],
            "all_metrics_finite": finite,
            "test_metrics_opened": False,
            "passed": passed,
        },
        "dataset_integrity": integrity,
        "decision": (
            "freeze_H016_ranker_and_open_registered_validation_seeds"
            if passed
            else "do_not_open_H016_frozen_validation"
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
            "Development-only generated-world action-ranking evidence; no "
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
