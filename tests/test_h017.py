from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from chimera.meta_world.h015.search import InterventionCandidate
from chimera.meta_world.h017 import suite as h017_suite_module
from chimera.meta_world.h017.config import H017SuiteConfig
from chimera.meta_world.h017.pool import (
    balanced_support_pool,
    support_pool_diagnostics,
)
from chimera.meta_world.h017.rerank import one_pass_qd_rerank


def test_h017_config_freezes_critic_pool_and_budgets() -> None:
    suite = H017SuiteConfig.from_yaml("configs/meta_world/world_h017_suite.yaml")
    assert suite.seed == 260958
    assert suite.frozen_validation_seeds == (260959, 260960, 260961)
    assert suite.support_pool.candidates_per_state == 256
    assert suite.adaptive_search.model_scores_per_state == 256
    assert suite.pool_reranking.simulator_executions_per_state == 8
    assert suite.critic_suite_config == Path(
        "configs/meta_world/world_h016_suite.yaml"
    )


def test_h017_support_pool_is_replay_exact_balanced_and_interior() -> None:
    candidates = balanced_support_pool(objects=5, count=256, seed=260999)
    replay = balanced_support_pool(objects=5, count=256, seed=260999)
    diagnostics = support_pool_diagnostics(candidates)
    assert candidates == replay
    assert diagnostics.legal_action_rate == 1.0
    assert diagnostics.exact_continuous_boundary_rate == 0.0
    assert diagnostics.unique_vector_rate == 1.0
    assert diagnostics.pair_count_discrepancy <= 1
    magnitudes = np.sort(np.asarray([item.magnitude for item in candidates]))
    lower = np.arange(256) / 256
    upper = np.arange(1, 257) / 256
    assert np.all(magnitudes > lower)
    assert np.all(magnitudes < upper)


def test_h017_support_pool_changes_with_seed_without_changing_contract() -> None:
    first = balanced_support_pool(objects=7, count=256, seed=1)
    second = balanced_support_pool(objects=7, count=256, seed=2)
    assert first != second
    for candidates in (first, second):
        diagnostics = support_pool_diagnostics(candidates)
        assert diagnostics.pair_count_discrepancy <= 1
        assert diagnostics.exact_continuous_boundary_rate == 0.0


def test_h017_one_pass_reranking_is_budget_exact_and_cell_diverse() -> None:
    candidates = balanced_support_pool(objects=5, count=256, seed=260958)
    scores = np.asarray(
        [item.magnitude + 0.01 * item.control for item in candidates],
        dtype=np.float64,
    )
    result = one_pass_qd_rerank(candidates, scores, executions=8)
    replay = one_pass_qd_rerank(candidates, scores.copy(), executions=8)
    assert result == replay
    assert result.model_scores == 256
    assert len(result.selected) == 8
    assert len({item.archive_cell for item in result.selected}) == 8
    assert result.archive_cells >= 8
    with pytest.raises(ValueError, match="invalid"):
        one_pass_qd_rerank(candidates, scores[:-1], executions=8)


def test_h017_reranking_retains_maximum_score_per_archive_cell() -> None:
    candidates = (
        InterventionCandidate(0, 1, 0.10, 0.0),
        InterventionCandidate(0, 1, 0.20, 0.0),
        InterventionCandidate(0, 1, 0.40, 0.0),
        InterventionCandidate(1, 0, 0.60, 0.0),
    )
    result = one_pass_qd_rerank(
        candidates,
        np.asarray([0.1, 0.9, 0.5, 0.4]),
        executions=3,
    )
    assert candidates[1] in [item.candidate for item in result.selected]
    assert candidates[0] not in [item.candidate for item in result.selected]


def test_h017_suite_compares_shared_critic_without_revalidating_wg4(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fake_backbone(config_path: object, output_dir: object) -> dict[str, object]:
        output = Path(str(output_dir))
        output.mkdir(parents=True)
        (output / "checkpoint.pt").write_bytes(b"checkpoint")
        (output / "checkpoint_manifest.json").write_text(
            json.dumps(
                {
                    "checkpoint_sha256": "e" * 64,
                    "weights_kind": "ema",
                }
            ),
            encoding="utf-8",
        )
        return {
            "run_id": "h017-test",
            "best_step": 2,
            "parameters": 100,
            "best_validation": {"intervention_effect_nrmse": 1.0},
            "runtime_seconds": 1.0,
            "peak_memory_bytes": 1,
            "environment": {"device": "cpu"},
        }

    ranking_result = {
        "deterministic_training_candidate_replay_rate": 1.0,
        "backbone_unchanged": True,
        "peak_memory_bytes": 1,
    }

    class FakePredictor:
        def predict_rank(
            self,
            window: object,
            candidates: tuple[InterventionCandidate, ...],
        ) -> tuple[np.ndarray, np.ndarray]:
            del window
            scores = np.asarray([item.magnitude for item in candidates])
            return scores, np.zeros_like(scores)

    monkeypatch.setattr(h017_suite_module, "run_h016_backbone_preflight", fake_backbone)
    monkeypatch.setattr(
        h017_suite_module,
        "run_h016_ranking_training",
        lambda *args, **kwargs: (SimpleNamespace(), ranking_result),
    )
    monkeypatch.setattr(
        h017_suite_module,
        "H016CandidatePredictor",
        lambda *args, **kwargs: FakePredictor(),
    )
    monkeypatch.setattr(
        h017_suite_module,
        "resolve_device",
        lambda value: torch.device("cpu"),
    )
    monkeypatch.setattr(
        h017_suite_module,
        "realized_candidate_effect",
        lambda config, trajectory, *, prediction_step, candidate: candidate.magnitude,
    )
    report_path = tmp_path / "reports" / "h017.json"
    report = h017_suite_module.run_h017_development_suite(
        "configs/meta_world/world_h017_suite.yaml",
        tmp_path / "runs",
        report_path,
    )
    gate = report["development_gate"]
    assert report["critic"]["weights_shared_between_search_arms"] is True
    assert gate["support_pool_replay_rate"] == 1.0
    assert gate["support_pool_exact_boundary_rate"] == 0.0
    assert gate["support_pool_unique_vector_rate"] == 1.0
    assert gate["model_score_budget_match_rate"] == 1.0
    assert report["dataset_integrity"]["revalidated"] is False
    assert report["test_metrics_opened"] is False
    assert json.loads(report_path.read_text(encoding="utf-8"))["checkpoint_promoted"] is False
