from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from chimera.data.ai_review import assert_ai_review_gate_passed, evaluate_ai_review_gate

DATA = Path("datasets/venture_corpus_c1")


def _evaluate(protocol: Path, status: Path | None = None) -> dict[str, object]:
    return evaluate_ai_review_gate(
        DATA / "manifest.json",
        DATA / "reviewer_packet.json",
        protocol,
        source_path=DATA / "source_cases.yaml",
        status_path=status,
    )


def _temporary_protocol(tmp_path: Path, review: dict[str, object] | None) -> Path:
    protocol = yaml.safe_load((DATA / "ai_review_protocol.yaml").read_text(encoding="utf-8"))
    protocol_path = tmp_path / "ai_review_protocol.yaml"
    if review is not None:
        review_dir = tmp_path / "ai_reviews"
        review_dir.mkdir()
        (review_dir / "multi_lens_ai_review.json").write_text(
            json.dumps(review), encoding="utf-8"
        )
    protocol_path.write_text(yaml.safe_dump(protocol), encoding="utf-8")
    return protocol_path


def test_committed_ai_review_gate_passes() -> None:
    result = _evaluate(DATA / "ai_review_protocol.yaml")
    assert result["status"] == "passed"
    assert result["generation_allowed"] is True
    assert result["human_review_required"] is False
    assert result["coverage"]["total_item_decisions"] == 1191
    assert_ai_review_gate_passed(result)


def test_missing_ai_review_blocks_gate(tmp_path: Path) -> None:
    result = _evaluate(_temporary_protocol(tmp_path, None))
    assert result["status"] == "blocked"
    assert result["generation_allowed"] is False
    with pytest.raises(RuntimeError, match="not passed"):
        assert_ai_review_gate_passed(result)


def test_rejected_ai_review_fails_gate(tmp_path: Path) -> None:
    review = json.loads(
        (DATA / "ai_reviews" / "multi_lens_ai_review.json").read_text(encoding="utf-8")
    )
    review["overall_verdict"] = "needs_change"
    result = _evaluate(_temporary_protocol(tmp_path, review))
    assert result["status"] == "failed"
    assert result["generation_allowed"] is False


def test_stale_ai_review_hash_is_rejected(tmp_path: Path) -> None:
    review = json.loads(
        (DATA / "ai_reviews" / "multi_lens_ai_review.json").read_text(encoding="utf-8")
    )
    review["input_integrity"]["source_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="input hash is stale"):
        _evaluate(_temporary_protocol(tmp_path, review))


def test_future_dataset_policy_requires_three_distinct_roles() -> None:
    policy = yaml.safe_load(Path("datasets/ai_review_policy.yaml").read_text(encoding="utf-8"))
    defaults = policy["default_for_new_datasets"]
    assert defaults["minimum_independent_subagents"] == 3
    assert defaults["unanimous_accept_required"] is True
    assert defaults["full_snapshot_coverage_per_subagent"] is True
    assert defaults["required_roles"] == [
        "source_diligence",
        "semantic_integrity",
        "commercial_challenge",
    ]
