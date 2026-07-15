from __future__ import annotations

import json
from pathlib import Path

from chimera.config import (
    ModelConfig,
    ProposalPolicyConfig,
    ProposalTrialConfig,
    TrainingConfig,
    TrialEvaluationConfig,
    VentureTrialConfig,
)
from chimera.data.synthetic import make_synthetic_batch
from chimera.models.venture import ChimeraVenture
from chimera.trials.proposal import (
    _aggregate_policy_metrics,
    _evaluate_policy,
    _historical_candidate_summary,
    _reconstruction_guardrail,
    _select_policy,
)


def _proposal_config(*, seeds: tuple[int, ...] = (17,)) -> ProposalTrialConfig:
    return ProposalTrialConfig(
        trial_id="CHM-V-T002",
        hypothesis_id="CHM-V-H001",
        checkpoint_path="checkpoint.pt",
        checkpoint_sha256="0" * 64,
        reconstruction_config="reconstruction.yaml",
        reconstruction_result="result.json",
        corpus_manifest="manifest.json",
        policies=(
            ProposalPolicyConfig("model-only", 0.75, 0.0),
            ProposalPolicyConfig("explore-50", 0.75, 0.5),
        ),
        baseline_policy_id="model-only",
        device="cpu",
        seeds=seeds,
        candidates_per_case=2,
        max_edits=2,
        archive_bins=(2, 2),
        unique_graph_rate_min=0.5,
        changed_candidate_rate_min=0.5,
    )


def test_policy_metrics_aggregate_and_select() -> None:
    seed_metrics = [
        {
            "unique_graph_rate": 0.75,
            "changed_candidate_rate": 1.0,
            "invalid_candidate_rate": 0.0,
            "archive_coverage": 0.5,
            "operation_counts": {"CONNECT": 2},
            "reproducible": True,
        },
        {
            "unique_graph_rate": 1.0,
            "changed_candidate_rate": 1.0,
            "invalid_candidate_rate": 0.0,
            "archive_coverage": 0.75,
            "operation_counts": {"MERGE": 1},
            "reproducible": True,
        },
    ]
    candidates = [
        {"proxy_scores": {"feasibility": 0.6}},
        {"proxy_scores": {"feasibility": 0.8}},
    ]
    explored = _aggregate_policy_metrics(
        ProposalPolicyConfig("explore-50", 0.75, 0.5),
        seed_metrics,
        candidates,
    )
    baseline = dict(explored)
    baseline["unique_graph_rate_mean"] = 0.25
    baseline["policy"] = {
        "policy_id": "model-only",
        "temperature": 0.75,
        "exploration_rate": 0.0,
    }
    selected, eligible = _select_policy(
        {"model-only": baseline, "explore-50": explored},
        _proposal_config(),
    )
    assert explored["operation_coverage"] == 2
    assert explored["feasibility_median"] == 0.7
    assert selected == "explore-50"
    assert eligible == ["explore-50"]


def test_historical_candidate_summary(tmp_path: Path) -> None:
    path = tmp_path / "candidates.jsonl"
    records = [
        {
            "graph_signature": "graph-a",
            "program_signature": "program-a",
            "source_case_id": "case-a",
            "changed": True,
            "edits": [{"operation": "CONNECT"}, {"operation": "STOP"}],
        },
        {
            "graph_signature": "graph-a",
            "program_signature": "program-a",
            "source_case_id": "case-a",
            "changed": False,
            "edits": [{"operation": "CONNECT"}, {"operation": "STOP"}],
        },
    ]
    path.write_text(
        "".join(json.dumps(record) + "\n" for record in records),
        encoding="utf-8",
    )
    summary = _historical_candidate_summary(path)
    assert summary["unique_graph_rate"] == 0.5
    assert summary["changed_candidate_rate"] == 0.5
    assert summary["top_program_sequence"] == "CONNECT>STOP"
    assert summary["per_case"]["case-a"]["unique_graphs"] == 1


def test_reconstruction_guardrail_reports_exact_rate(
    small_model_config: ModelConfig,
) -> None:
    model = ChimeraVenture(small_model_config)

    class SyntheticShard:
        def __len__(self) -> int:
            return 2

        def batch(self, indices: list[int]):  # type: ignore[no-untyped-def]
            return make_synthetic_batch(
                small_model_config,
                batch_size=len(indices),
                seed=41,
            )

    metrics = _reconstruction_guardrail(model, SyntheticShard())  # type: ignore[arg-type]
    assert metrics["examples"] == 2.0
    assert 0.0 <= metrics["exact_graph_rate"] <= 1.0


def test_policy_evaluation_generates_valid_candidates(
    small_model_config: ModelConfig,
) -> None:
    model = ChimeraVenture(small_model_config)
    base = VentureTrialConfig(
        trial_id="CHM-V-T000",
        hypothesis_id="CHM-V-H001",
        model=small_model_config,
        training=TrainingConfig(device="cpu", steps=1),
        evaluation=TrialEvaluationConfig(
            corpus_manifest="manifest.json",
            candidates_per_case=2,
            min_edits=1,
            max_edits=2,
            archive_bins=(2, 2),
        ),
    )
    graph = make_synthetic_batch(small_model_config, batch_size=1, seed=43).next_graph
    config = _proposal_config()
    candidates, metrics = _evaluate_policy(
        model,
        [("case-a", graph)],
        base,
        config,
        config.policies[1],
        split="train",
    )
    assert len(candidates) == 2
    assert all(candidate["valid"] for candidate in candidates)
    assert metrics["candidates"] == 2
    assert metrics["reproducible"]
