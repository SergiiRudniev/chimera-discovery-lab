from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from chimera.meta_world.generators import (
    GeneratedWorldDatasetConfig,
    SplitName,
    WorldGenerationPipeline,
)
from chimera.meta_world.h002.windows import (
    make_transition_window,
    materialize_sequence_sample,
)
from chimera.meta_world.h015 import preflight as h015_preflight_module
from chimera.meta_world.h015 import suite as h015_suite_module
from chimera.meta_world.h015.config import (
    H015BackboneConfig,
    H015SearchConfig,
    H015SuiteConfig,
)
from chimera.meta_world.h015.evaluation import (
    candidate_batch,
    realized_candidate_effect,
    slice_sequence_sample,
    uniform_legal_pool,
)
from chimera.meta_world.h015.search import (
    InterventionCandidate,
    quality_diversity_search,
)

GENERATOR = Path("configs/meta_world/world_generators_h013.yaml")


def _single_window() -> tuple[object, object, GeneratedWorldDatasetConfig]:
    config = GeneratedWorldDatasetConfig.from_yaml(GENERATOR)
    pipeline = WorldGenerationPipeline(config)
    trajectory = pipeline.materialize(SplitName.VALIDATION, 0)
    grouped = materialize_sequence_sample(
        pipeline,
        SplitName.VALIDATION,
        start_index=0,
        batch_size=config.views_per_mechanism,
    )
    sample = slice_sequence_sample(grouped, 0)
    return (
        trajectory,
        make_transition_window(sample, prediction_step=3, context_steps=4),
        config,
    )


def test_h015_configs_freeze_search_and_sealed_data() -> None:
    backbone = H015BackboneConfig.from_yaml(
        "configs/meta_world/world_h015_development_backbone.yaml"
    )
    suite = H015SuiteConfig.from_yaml("configs/meta_world/world_h015_suite.yaml")
    assert backbone.paired_runtime.runtime.training.seed == 260950
    assert suite.search.model_scores_per_state == 256
    assert suite.search.simulator_executions_per_state == 8
    assert suite.oracle_pool_candidates_per_state == 256
    assert suite.frozen_validation_seeds == (260951, 260952, 260953)


def test_h015_search_is_legal_deterministic_and_budget_exact() -> None:
    config = H015SearchConfig(
        rounds=4,
        candidates_per_round=64,
        elite_candidates_per_round=8,
        simulator_executions_per_state=8,
        archive_descriptor=("source_slot", "target_slot", "magnitude_quartile"),
    )

    def predict(
        candidates: tuple[InterventionCandidate, ...],
    ) -> tuple[np.ndarray, np.ndarray]:
        means = np.asarray(
            [item.magnitude + 0.1 * item.control for item in candidates],
            dtype=np.float64,
        )
        deviations = np.asarray(
            [0.05 + item.magnitude for item in candidates],
            dtype=np.float64,
        )
        return means, deviations

    first = quality_diversity_search(
        objects=5,
        seed=260950,
        config=config,
        uncertainty_beta=1.0,
        predict=predict,
    )
    replay = quality_diversity_search(
        objects=5,
        seed=260950,
        config=config,
        uncertainty_beta=1.0,
        predict=predict,
    )
    mean_only = quality_diversity_search(
        objects=5,
        seed=260950,
        config=config,
        uncertainty_beta=0.0,
        predict=predict,
    )
    assert first == replay
    assert first.model_scores == 256
    assert len(first.selected) == 8
    assert first != mean_only
    assert all(
        item.candidate.source_slot != item.candidate.target_slot
        for item in first.selected
    )
    with pytest.raises(ValueError, match="must differ"):
        InterventionCandidate(0, 0, 0.5, 0.0)


def test_h015_candidate_batch_replaces_only_intervention_inputs() -> None:
    _, window, _ = _single_window()
    candidates = (
        InterventionCandidate(0, 1, 0.25, -0.5),
        InterventionCandidate(1, 2, 0.75, 0.5),
    )
    scored = candidate_batch(window, candidates)  # type: ignore[arg-type]
    assert scored.batch_size == 2
    assert scored.source_slots.tolist() == [0, 1]
    assert scored.target_slots.tolist() == [1, 2]
    assert torch.equal(scored.observations[0], scored.observations[1])
    assert torch.equal(scored.next_observations[0], scored.next_observations[1])
    assert torch.allclose(
        scored.intervention_parameters[:, :2],
        torch.tensor([[0.25, -0.5], [0.75, 0.5]]),
    )


def test_h015_simulator_replay_recovers_recorded_effect_exactly() -> None:
    trajectory, _, config = _single_window()
    transition = trajectory.transitions[3]  # type: ignore[attr-defined]
    action = transition.action
    candidate = InterventionCandidate(
        action.source,
        action.target,
        action.magnitude,
        action.control,
    )
    first = realized_candidate_effect(
        config,
        trajectory,  # type: ignore[arg-type]
        prediction_step=3,
        candidate=candidate,
    )
    second = realized_candidate_effect(
        config,
        trajectory,  # type: ignore[arg-type]
        prediction_step=3,
        candidate=candidate,
    )
    assert first == second == float(transition.outcome[3])
    pool = uniform_legal_pool(objects=5, count=256, seed=260950)
    assert pool == uniform_legal_pool(objects=5, count=256, seed=260950)
    assert all(item.source_slot != item.target_slot for item in pool)


def test_h015_preflight_routes_factual_residual_backbone(
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
        assert model.__class__.__name__ == "ResponseConditionedEffectWorldModel"
        return {
            "run_id": "h015-smoke",
            "status": "completed_preflight",
            "arm": kwargs["reported_arm"],
            **kwargs["result_metadata"],  # type: ignore[arg-type]
        }

    monkeypatch.setattr(
        h015_preflight_module,
        "execute_paired_transition_preflight",
        fake_execute,
    )
    result = h015_preflight_module.run_h015_backbone_preflight(
        "configs/meta_world/world_h015_development_smoke.yaml",
        tmp_path / "run",
    )
    assert result["response_source"] == "predicted_factual_minus_final_observation"
    assert captured["selection_metrics"] == ("intervention_effect_nrmse",)


def test_h015_suite_generates_candidates_without_revalidating_wg4(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fake_preflight(config_path: object, output_dir: object) -> dict[str, object]:
        output = Path(str(output_dir))
        output.mkdir(parents=True)
        (output / "checkpoint_manifest.json").write_text(
            json.dumps(
                {
                    "checkpoint_sha256": "c" * 64,
                    "weights_kind": "ema",
                }
            ),
            encoding="utf-8",
        )
        return {
            "run_id": "h015-test",
            "best_step": 2,
            "parameters": 100,
            "best_validation": {"intervention_effect_nrmse": 1.0},
            "runtime_seconds": 1.0,
            "peak_memory_bytes": 1,
            "environment": {"device": "cpu"},
        }

    class FakePredictor:
        def predict(
            self,
            window: object,
            candidates: tuple[InterventionCandidate, ...],
        ) -> tuple[np.ndarray, np.ndarray]:
            del window
            return (
                np.asarray([item.magnitude for item in candidates], dtype=np.float64),
                np.asarray(
                    [0.01 + 0.02 * abs(item.control) for item in candidates],
                    dtype=np.float64,
                ),
            )

    monkeypatch.setattr(h015_suite_module, "run_h015_backbone_preflight", fake_preflight)
    monkeypatch.setattr(
        h015_suite_module,
        "ResponseConditionedEffectWorldModel",
        lambda *args, **kwargs: SimpleNamespace(),
    )
    monkeypatch.setattr(
        h015_suite_module,
        "load_candidate_predictor",
        lambda *args, **kwargs: FakePredictor(),
    )
    monkeypatch.setattr(
        h015_suite_module,
        "realized_candidate_effect",
        lambda config, trajectory, *, prediction_step, candidate: candidate.magnitude,
    )
    report_path = tmp_path / "reports" / "h015.json"
    report = h015_suite_module.run_h015_development_suite(
        "configs/meta_world/world_h015_suite.yaml",
        tmp_path / "runs",
        report_path,
    )
    assert report["candidate_generation"]["evaluation_states"] == 32
    assert report["development_gate"]["legal_action_rate"] == 1.0
    assert report["development_gate"]["deterministic_search_replay_rate"] == 1.0
    assert report["dataset_integrity"]["revalidated"] is False
    assert report["test_metrics_opened"] is False
    assert json.loads(report_path.read_text(encoding="utf-8"))["checkpoint_promoted"] is False
