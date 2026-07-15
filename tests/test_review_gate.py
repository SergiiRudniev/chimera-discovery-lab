from __future__ import annotations

import json
from pathlib import Path

import pytest

from chimera.data.review import (
    assert_review_gate_passed,
    build_review_packet,
    build_review_template,
    evaluate_review_gate,
)

DATA = Path("datasets/venture_corpus_c1")


def _complete_review() -> dict[str, object]:
    review = json.loads((DATA / "review_template.json").read_text(encoding="utf-8"))
    review["review_id"] = "independent-review-001"
    review["reviewer"] = {
        "id": "external-reviewer-001",
        "independent_of_annotation_author": True,
        "conflicts_disclosed": "none",
    }
    review["attestation"] = {
        "registered_source_documents_opened": True,
        "no_candidate_outputs_seen": True,
        "no_model_arm_outputs_seen": True,
    }
    for case in review["cases"]:
        case["filing_identity"] = "verified"
        for field in ("evidence_notes", "nodes", "edges", "objective_nodes", "constraint_nodes"):
            for decision in case[field]:
                decision["decision"] = "verified"
        case["overall"] = "accept"
    return review


def _write_review(tmp_path: Path, review: object) -> Path:
    reviews = tmp_path / "reviews"
    reviews.mkdir()
    (reviews / "independent-review-001.review.json").write_text(
        json.dumps(review), encoding="utf-8"
    )
    return reviews


def test_committed_review_gate_is_explicitly_blocked() -> None:
    result = evaluate_review_gate(
        DATA / "manifest.json",
        DATA / "reviewer_packet.json",
        DATA / "review_protocol.yaml",
        DATA / "reviews",
    )
    assert result["status"] == "blocked"
    assert result["generation_allowed"] is False
    with pytest.raises(RuntimeError, match="not passed"):
        assert_review_gate_passed(result)


def test_ai_review_cannot_satisfy_human_gate() -> None:
    review = json.loads(
        (DATA / "ai_reviews" / "multi_lens_ai_review.json").read_text(encoding="utf-8")
    )
    assert review["review_type"] == "ai_assisted_internal_review"
    assert review["reviewer"]["human_independent"] is False
    assert review["reviewer"]["satisfies_human_gate"] is False
    assert review["generation_allowed"] is False
    assert len(review["cases"]) == 10
    assert {case["verdict"] for case in review["cases"]} == {"needs_change"}


def test_review_packet_build_is_deterministic(tmp_path: Path) -> None:
    packet_path = tmp_path / "packet.json"
    template_path = tmp_path / "template.json"
    packet = build_review_packet(
        DATA / "manifest.json",
        DATA / "source_cases.yaml",
        DATA / "internal_source_audit.yaml",
        DATA / "review_protocol.yaml",
        packet_path,
    )
    template = build_review_template(packet_path, template_path)
    assert packet == json.loads((DATA / "reviewer_packet.json").read_text(encoding="utf-8"))
    assert template == json.loads((DATA / "review_template.json").read_text(encoding="utf-8"))
    assert b"\r\n" not in packet_path.read_bytes()
    assert b"\r\n" not in template_path.read_bytes()


def test_complete_independent_review_passes_gate(tmp_path: Path) -> None:
    reviews = _write_review(tmp_path, _complete_review())
    result = evaluate_review_gate(
        DATA / "manifest.json",
        DATA / "reviewer_packet.json",
        DATA / "review_protocol.yaml",
        reviews,
    )
    assert result["status"] == "passed"
    assert result["generation_allowed"] is True
    assert_review_gate_passed(result)


def test_review_requesting_change_fails_gate(tmp_path: Path) -> None:
    review = _complete_review()
    review["cases"][0]["nodes"][0]["decision"] = "needs_change"
    review["cases"][0]["overall"] = "needs_change"
    reviews = _write_review(tmp_path, review)
    result = evaluate_review_gate(
        DATA / "manifest.json",
        DATA / "reviewer_packet.json",
        DATA / "review_protocol.yaml",
        reviews,
    )
    assert result["status"] == "failed"
    assert result["changed_reviews"] == 1
    assert result["generation_allowed"] is False


def test_changed_protocol_invalidates_review_packet(tmp_path: Path) -> None:
    protocol = tmp_path / "review_protocol.yaml"
    protocol.write_text(
        (DATA / "review_protocol.yaml").read_text(encoding="utf-8") + "\n# changed\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="protocol hash is stale"):
        evaluate_review_gate(
            DATA / "manifest.json",
            DATA / "reviewer_packet.json",
            protocol,
            DATA / "reviews",
        )


def test_annotation_author_cannot_self_approve(tmp_path: Path) -> None:
    review = json.loads((DATA / "review_template.json").read_text(encoding="utf-8"))
    review["review_id"] = "self-review"
    review["reviewer"] = {
        "id": "chimera-corpus-author-001",
        "independent_of_annotation_author": True,
        "conflicts_disclosed": "none",
    }
    reviews = tmp_path / "reviews"
    reviews.mkdir()
    (reviews / "self-review.review.json").write_text(json.dumps(review), encoding="utf-8")
    with pytest.raises(ValueError, match="annotation author"):
        evaluate_review_gate(
            DATA / "manifest.json",
            DATA / "reviewer_packet.json",
            DATA / "review_protocol.yaml",
            reviews,
        )
