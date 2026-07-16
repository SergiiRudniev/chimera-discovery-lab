from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from chimera.meta_world.config import MetaWorldModelConfig
from chimera.meta_world.generators import (
    GeneratedWorldDatasetConfig,
    SplitName,
    WorldGenerationPipeline,
)
from chimera.meta_world.h014.model import (
    ResponseConditionedEffectWorldModel,
    ResponseSource,
)
from chimera.meta_world.h015.evaluation import candidate_batch
from chimera.meta_world.h015.search import InterventionCandidate
from chimera.meta_world.h016 import preflight as h016_preflight_module
from chimera.meta_world.h016 import suite as h016_suite_module
from chimera.meta_world.h016.config import (
    H016BackboneConfig,
    H016SuiteConfig,
)
from chimera.meta_world.h016.dataset import materialize_ranking_group
from chimera.meta_world.h016.evaluation import ndcg_at_k, spearman_rank_correlation
from chimera.meta_world.h016.model import WithinStateActionRanker
from chimera.meta_world.h016.objectives import h016_ranking_loss
from chimera.meta_world.h016.trainer import H016RankingTrainer

GENERATOR = Path("configs/meta_world/world_generators_h013.yaml")


def _small_model_config() -> MetaWorldModelConfig:
    return MetaWorldModelConfig(
        observation_features=8,
        relation_features=4,
        intervention_types=1,
        intervention_parameters=3,
        effect_dimensions=4,
        domain_count=1,
        mechanism_count=8,
        hidden_dim=32,
        num_heads=4,
        spatial_layers=1,
        temporal_layers=1,
        transition_layers=1,
        feedforward_multiplier=2,
        max_slots=10,
        context_steps=4,
        dropout=0.0,
    )


def _groups() -> tuple[object, object, H016SuiteConfig]:
    suite = H016SuiteConfig.from_yaml("configs/meta_world/world_h016_suite.yaml")
    pipeline = WorldGenerationPipeline(GeneratedWorldDatasetConfig.from_yaml(GENERATOR))
    first = materialize_ranking_group(
        pipeline,
        SplitName.TRAIN,
        state_ordinal=0,
        seed=suite.seed,
        config=suite.ranking,
        audit_replay=True,
    )
    second = materialize_ranking_group(
        pipeline,
        SplitName.TRAIN,
        state_ordinal=1,
        seed=suite.seed,
        config=suite.ranking,
        audit_replay=True,
    )
    return first, second, suite


def test_h016_configs_freeze_ranking_and_data_boundary() -> None:
    backbone = H016BackboneConfig.from_yaml(
        "configs/meta_world/world_h016_development_backbone.yaml"
    )
    suite = H016SuiteConfig.from_yaml("configs/meta_world/world_h016_suite.yaml")
    assert backbone.paired_runtime.runtime.training.seed == 260954
    assert backbone.paired_runtime.runtime.training.steps == 600
    assert suite.ranking.steps == 600
    assert suite.ranking.candidates_per_state == 16
    assert suite.search.model_scores_per_state == 256
    assert suite.frozen_validation_seeds == (260955, 260956, 260957)


def test_h016_ranking_groups_are_shared_state_replay_exact() -> None:
    first, replay, suite = _groups()
    assert first.deterministic_replay  # type: ignore[attr-defined]
    assert replay.deterministic_replay  # type: ignore[attr-defined]
    assert len(first.candidates) == 16  # type: ignore[attr-defined]
    assert first.candidate_seed == suite.seed + 101  # type: ignore[attr-defined]
    assert replay.renderer_view == 1  # type: ignore[attr-defined]
    again, _, _ = _groups()
    assert first.candidates == again.candidates  # type: ignore[attr-defined]
    assert np.array_equal(  # type: ignore[attr-defined]
        first.realized_effects,
        again.realized_effects,
    )


def test_h016_ranker_cannot_read_evaluator_targets() -> None:
    first, _, _ = _groups()
    ranker = WithinStateActionRanker(
        ResponseConditionedEffectWorldModel(
            _small_model_config(),
            response_source=ResponseSource.FACTUAL_RESIDUAL,
        )
    ).eval()
    candidates = first.candidates[:4]  # type: ignore[attr-defined]
    original = candidate_batch(first.window, candidates)  # type: ignore[attr-defined]
    altered_window = replace(
        first.window,  # type: ignore[attr-defined]
        next_observations=torch.full_like(first.window.next_observations, 999.0),  # type: ignore[attr-defined]
        effect_targets=torch.full_like(first.window.effect_targets, -999.0),  # type: ignore[attr-defined]
        counterfactual_no_op_observations=torch.full_like(
            first.window.counterfactual_no_op_observations,  # type: ignore[attr-defined]
            777.0,
        ),
    )
    altered = candidate_batch(altered_window, candidates)
    with torch.no_grad():
        first_logits = ranker(original).rank_logits
        altered_logits = ranker(altered).rank_logits
    assert torch.equal(first_logits, altered_logits)
    assert not any(parameter.requires_grad for parameter in ranker.backbone.parameters())


def test_h016_objective_prefers_correct_ordering() -> None:
    suite = H016SuiteConfig.from_yaml("configs/meta_world/world_h016_suite.yaml")
    effects = torch.tensor([-0.02, -0.01, 0.0, 0.03])
    aligned = h016_ranking_loss(effects * 20.0, effects, suite.ranking)
    reversed_order = h016_ranking_loss(-effects * 20.0, effects, suite.ranking)
    assert aligned["loss"] < reversed_order["loss"]
    assert aligned["retained_pairs"] == 6
    assert torch.isfinite(torch.stack(list(aligned.values()))).all()


def test_h016_trainer_updates_only_rank_head() -> None:
    first, second, suite = _groups()
    ranker = WithinStateActionRanker(
        ResponseConditionedEffectWorldModel(
            _small_model_config(),
            response_source=ResponseSource.FACTUAL_RESIDUAL,
        )
    )
    before_backbone = {
        name: value.detach().clone() for name, value in ranker.backbone.state_dict().items()
    }
    before_head = {
        name: value.detach().clone() for name, value in ranker.rank_head.state_dict().items()
    }
    trainer = H016RankingTrainer(
        ranker,
        suite.ranking,
        device=torch.device("cpu"),
        use_autocast=False,
    )
    metrics = trainer.train_step([first, second])  # type: ignore[list-item]
    assert metrics["backbone_gradient_tensors"] == 0.0
    assert all(
        torch.equal(before_backbone[name], value)
        for name, value in ranker.backbone.state_dict().items()
    )
    assert any(
        not torch.equal(before_head[name], value)
        for name, value in ranker.rank_head.state_dict().items()
    )


def test_h016_ranking_diagnostics_have_exact_endpoints() -> None:
    targets = np.asarray([-1.0, 0.0, 2.0, 5.0], dtype=np.float64)
    assert spearman_rank_correlation(targets, targets) == pytest.approx(1.0)
    assert spearman_rank_correlation(-targets, targets) == pytest.approx(-1.0)
    assert ndcg_at_k(targets, targets, k=3) == pytest.approx(1.0)
    assert ndcg_at_k(-targets, targets, k=3) < 1.0


def test_h016_preflight_routes_registered_backbone(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}

    def fake_execute(
        config_path: object,
        output_dir: object,
        **kwargs: object,
    ) -> dict[str, object]:
        captured.update(kwargs)
        model = kwargs["model_factory"](kwargs["run_config"])  # type: ignore[operator]
        assert isinstance(model, ResponseConditionedEffectWorldModel)
        return {"status": "completed_preflight", **kwargs["result_metadata"]}  # type: ignore[dict-item]

    monkeypatch.setattr(
        h016_preflight_module,
        "execute_paired_transition_preflight",
        fake_execute,
    )
    result = h016_preflight_module.run_h016_backbone_preflight(
        "configs/meta_world/world_h016_development_smoke_backbone.yaml",
        tmp_path / "run",
    )
    assert result["response_source"] == "predicted_factual_minus_final_observation"
    assert captured["selection_metrics"] == ("intervention_effect_nrmse",)


def test_h016_suite_applies_rank_gate_without_revalidating_wg4(
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
                    "checkpoint_sha256": "d" * 64,
                    "weights_kind": "ema",
                }
            ),
            encoding="utf-8",
        )
        return {
            "run_id": "h016-test",
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

        def predict_pointwise(
            self,
            window: object,
            candidates: tuple[InterventionCandidate, ...],
        ) -> tuple[np.ndarray, np.ndarray]:
            del window
            scores = np.asarray([item.control for item in candidates])
            return scores, np.zeros_like(scores)

    monkeypatch.setattr(h016_suite_module, "run_h016_backbone_preflight", fake_backbone)
    monkeypatch.setattr(
        h016_suite_module,
        "run_h016_ranking_training",
        lambda *args, **kwargs: (SimpleNamespace(), ranking_result),
    )
    monkeypatch.setattr(
        h016_suite_module,
        "H016CandidatePredictor",
        lambda *args, **kwargs: FakePredictor(),
    )
    monkeypatch.setattr(
        h016_suite_module,
        "resolve_device",
        lambda value: torch.device("cpu"),
    )
    monkeypatch.setattr(
        h016_suite_module,
        "realized_candidate_effect",
        lambda config, trajectory, *, prediction_step, candidate: candidate.magnitude,
    )
    report_path = tmp_path / "reports" / "h016.json"
    report = h016_suite_module.run_h016_development_suite(
        "configs/meta_world/world_h016_suite.yaml",
        tmp_path / "runs",
        report_path,
    )
    assert report["development_gate"]["deterministic_training_candidate_replay_rate"] == 1.0
    assert report["development_gate"]["model_score_budget_match_rate"] == 1.0
    assert report["dataset_integrity"]["revalidated"] is False
    assert report["test_metrics_opened"] is False
    assert json.loads(report_path.read_text(encoding="utf-8"))["checkpoint_promoted"] is False
