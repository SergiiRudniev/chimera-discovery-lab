"""AI-subagent dataset review gate."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import yaml

from chimera.data.evaluation import validate_evaluation_corpus

AI_REVIEW_GATE_SCHEMA_VERSION = 1


def evaluate_ai_review_gate(
    manifest_path: str | Path,
    packet_path: str | Path,
    protocol_path: str | Path,
    *,
    source_path: str | Path | None = None,
    status_path: str | Path | None = None,
) -> dict[str, Any]:
    """Validate configured AI review artifacts against one immutable dataset snapshot."""

    manifest_file = Path(manifest_path)
    packet_file = Path(packet_path)
    protocol_file = Path(protocol_path)
    source_file = (
        Path(source_path) if source_path is not None else manifest_file.parent / "source_cases.yaml"
    )
    validate_evaluation_corpus(manifest_file)
    manifest = _load_json(manifest_file, "manifest")
    packet = _load_json(packet_file, "review packet")
    protocol = _load_yaml(protocol_file, "AI review protocol")
    if protocol.get("schema_version") != AI_REVIEW_GATE_SCHEMA_VERSION:
        raise ValueError("unsupported AI review protocol schema version")
    corpus_id = _required_string(manifest, "corpus_id")
    if _required_string(packet, "corpus_id") != corpus_id:
        raise ValueError("review packet corpus does not match manifest")
    if _required_string(protocol, "corpus_id") != corpus_id:
        raise ValueError("AI review protocol corpus does not match manifest")
    if _required_string(packet, "manifest_sha256") != _sha256(manifest_file):
        raise ValueError("review packet manifest hash is stale")
    if _required_string(packet, "source_sha256") != _sha256(source_file):
        raise ValueError("review packet source hash is stale")

    expected_coverage = _expected_full_coverage(packet)
    configured = _mapping_list(protocol.get("required_reviews"), "required AI reviews")
    minimum = int(protocol.get("minimum_accepted_reviews", len(configured)))
    if minimum <= 0 or minimum > len(configured):
        raise ValueError("minimum accepted AI reviews is invalid")

    accepted: list[str] = []
    rejected: list[str] = []
    missing: list[str] = []
    artifacts: list[dict[str, str]] = []
    seen_review_ids: set[str] = set()
    seen_reviewer_ids: set[str] = set()
    seen_roles: set[str] = set()
    for specification in configured:
        relative_path = Path(_required_string(specification, "path"))
        review_file = protocol_file.parent / relative_path
        expected_review_id = _required_string(specification, "review_id")
        role = _required_string(specification, "role")
        if expected_review_id in seen_review_ids or role in seen_roles:
            raise ValueError("AI review IDs and roles must be unique")
        seen_review_ids.add(expected_review_id)
        seen_roles.add(role)
        if not review_file.is_file():
            missing.append(expected_review_id)
            continue
        review = _load_json(review_file, "AI review")
        review_id = _required_string(review, "review_id")
        if review_id != expected_review_id:
            raise ValueError(f"unexpected AI review ID: {review_id}")
        if _required_string(review, "corpus_id") != corpus_id:
            raise ValueError(f"AI review corpus does not match: {review_id}")
        if review.get("review_type") not in {
            "ai_assisted_internal_review",
            "ai_ensemble_specialist_review",
        }:
            raise ValueError(f"unsupported AI review type: {review_id}")
        reviewer = _mapping(review.get("reviewer"), f"reviewer for {review_id}")
        if reviewer.get("type") != "codex_subagent":
            raise ValueError(f"AI review must be produced by a subagent: {review_id}")
        reviewer_id = _required_string(reviewer, "reviewer_id")
        if reviewer_id in seen_reviewer_ids:
            raise ValueError("AI ensemble reviewers must be distinct")
        seen_reviewer_ids.add(reviewer_id)
        integrity = _mapping(review.get("input_integrity"), f"input integrity for {review_id}")
        expected_hashes = {
            "corpus_manifest_sha256": _sha256(manifest_file),
            "source_sha256": _sha256(source_file),
            "reviewer_packet_sha256": _sha256(packet_file),
        }
        for field, expected_hash in expected_hashes.items():
            if _required_string(integrity, field) != expected_hash:
                raise ValueError(f"AI review input hash is stale: {review_id}/{field}")
        artifacts.append(
            {
                "review_id": review_id,
                "role": role,
                "path": relative_path.as_posix(),
                "sha256": _sha256(review_file),
            }
        )
        if review.get("overall_verdict") != "accept":
            rejected.append(review_id)
            continue
        _validate_full_acceptance(review, expected_coverage, review_id)
        accepted.append(review_id)

    if rejected:
        status = "failed"
        reason = "at_least_one_required_ai_review_rejected_snapshot"
    elif missing:
        status = "blocked"
        reason = "required_ai_review_artifact_missing"
    elif len(accepted) >= minimum:
        status = "passed"
        reason = "configured_ai_review_policy_accepted_snapshot"
    else:
        status = "blocked"
        reason = "minimum_accepted_ai_reviews_not_met"
    result = {
        "schema_version": AI_REVIEW_GATE_SCHEMA_VERSION,
        "corpus_id": corpus_id,
        "gate_authority": "ai_subagent_review",
        "policy_mode": _required_string(protocol, "policy_mode"),
        "status": status,
        "reason": reason,
        "generation_allowed": status == "passed",
        "human_review_required": False,
        "minimum_accepted_reviews": minimum,
        "accepted_ai_reviews": len(accepted),
        "required_ai_reviews": len(configured),
        "accepted_review_ids": accepted,
        "rejected_review_ids": rejected,
        "missing_review_ids": missing,
        "manifest_sha256": _sha256(manifest_file),
        "source_sha256": _sha256(source_file),
        "packet_sha256": _sha256(packet_file),
        "review_artifacts": artifacts,
        "coverage": expected_coverage,
    }
    if status_path is not None:
        _write_json(Path(status_path), result)
    return result


def assert_ai_review_gate_passed(result: Mapping[str, Any]) -> None:
    """Prevent downstream generation unless the configured AI review policy passed."""

    if result.get("status") != "passed" or result.get("generation_allowed") is not True:
        raise RuntimeError("AI dataset review gate is not passed")


def _expected_full_coverage(packet: Mapping[str, Any]) -> dict[str, int]:
    cases = _mapping_list(packet.get("cases"), "review packet cases")
    coverage = {
        "registered_cases": len(cases),
        "filing_identity_decisions": len(cases),
        "source_locator_decisions": len(cases),
        "evidence_note_decisions": 0,
        "node_decisions": 0,
        "node_rating_axis_decisions": 0,
        "edge_decisions": 0,
        "objective_node_decisions": 0,
        "constraint_node_decisions": 0,
    }
    for case in cases:
        evidence = _mapping_list(case.get("evidence_notes"), "evidence notes")
        nodes = _mapping_list(case.get("nodes"), "nodes")
        edges = _mapping_list(case.get("edges"), "edges")
        coverage["evidence_note_decisions"] += len(evidence)
        coverage["node_decisions"] += len(nodes)
        coverage["node_rating_axis_decisions"] += sum(
            len(_mapping_list(node.get("ratings"), "node ratings")) for node in nodes
        )
        coverage["edge_decisions"] += len(edges)
        coverage["objective_node_decisions"] += len(
            _string_list(case.get("objective_nodes"), "objective nodes")
        )
        coverage["constraint_node_decisions"] += len(
            _string_list(case.get("constraint_nodes"), "constraint nodes")
        )
    coverage["total_item_decisions"] = sum(
        value for field, value in coverage.items() if field != "registered_cases"
    )
    return coverage


def _validate_full_acceptance(
    review: Mapping[str, Any], expected_coverage: Mapping[str, int], review_id: str
) -> None:
    coverage = _mapping(review.get("coverage"), f"coverage for {review_id}")
    for field, expected in expected_coverage.items():
        if int(coverage.get(field, -1)) != expected:
            raise ValueError(f"AI review coverage mismatch: {review_id}/{field}")
    if coverage.get("item_level_decision_ledger_complete") is not True:
        raise ValueError(f"AI review ledger is incomplete: {review_id}")
    decisions = _mapping(review.get("decision_counts"), f"decisions for {review_id}")
    if dict(decisions) != {"verified": expected_coverage["total_item_decisions"]}:
        raise ValueError(f"AI review contains unresolved decisions: {review_id}")
    case_counts = _mapping(review.get("case_verdict_counts"), f"case verdicts for {review_id}")
    if dict(case_counts) != {"accept": expected_coverage["registered_cases"]}:
        raise ValueError(f"AI review did not accept every case: {review_id}")
    remaining = review.get("remaining_issues")
    if not isinstance(remaining, list) or remaining:
        raise ValueError(f"AI review has remaining issues: {review_id}")


def _load_json(path: Path, name: str) -> Mapping[str, Any]:
    return _mapping(json.loads(path.read_text(encoding="utf-8")), name)


def _load_yaml(path: Path, name: str) -> Mapping[str, Any]:
    return _mapping(yaml.safe_load(path.read_text(encoding="utf-8")), name)


def _mapping(value: object, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{name} must be a mapping")
    return value


def _mapping_list(value: object, name: str) -> list[Mapping[str, Any]]:
    if not isinstance(value, list) or not all(isinstance(item, Mapping) for item in value):
        raise TypeError(f"{name} must be a list of mappings")
    return list(value)


def _string_list(value: object, name: str) -> list[str]:
    if not isinstance(value, list) or not value or not all(
        isinstance(item, str) and item for item in value
    ):
        raise ValueError(f"{name} must be a non-empty string list")
    return list(value)


def _required_string(values: Mapping[str, Any], name: str) -> str:
    value = values.get(name)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(value, indent=2, sort_keys=True) + "\n")
