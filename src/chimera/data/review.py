"""Independent source-to-graph review gate for Venture evaluation corpora."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import yaml

from chimera.data.evaluation import validate_evaluation_corpus
from chimera.data.semantics import ANNOTATED_FEATURE_NAMES

REVIEW_SCHEMA_VERSION = 2
REVIEW_DECISIONS = frozenset({"verified", "needs_change", "cannot_verify"})


def build_review_packet(
    manifest_path: str | Path,
    source_path: str | Path,
    audit_path: str | Path,
    protocol_path: str | Path,
    output_path: str | Path,
) -> dict[str, Any]:
    """Build the deterministic packet an independent reviewer must inspect."""

    manifest_file = Path(manifest_path)
    source_file = Path(source_path)
    audit_file = Path(audit_path)
    protocol_file = Path(protocol_path)
    manifest = _load_json_mapping(manifest_file, "manifest")
    source = _load_yaml_mapping(source_file, "source document")
    audit = _load_yaml_mapping(audit_file, "internal source audit")
    protocol = _load_yaml_mapping(protocol_file, "review protocol")
    cases = _read_jsonl(manifest_file.parent / "cases.jsonl")
    corpus_id = _required_string(manifest, "corpus_id")
    author_id = _required_string(source, "annotation_author_id")
    if _required_string(audit, "corpus_id") != corpus_id:
        raise ValueError("internal audit corpus does not match manifest")
    if _required_string(protocol, "corpus_id") != corpus_id:
        raise ValueError("review protocol corpus does not match manifest")
    if _required_string(audit, "auditor_id") != author_id or audit.get("independent") is not False:
        raise ValueError("internal audit must be attributed to the annotation author")
    if _required_string(protocol, "annotation_author_id") != author_id:
        raise ValueError("review protocol annotation author does not match source")
    audit_cases_value = audit.get("cases")
    if not isinstance(audit_cases_value, list):
        raise TypeError("internal audit cases must be a list")
    audit_cases = {
        _required_string(_mapping(value, "internal audit case"), "case_id"): _mapping(
            value, "internal audit case"
        )
        for value in audit_cases_value
    }
    case_ids = [_required_string(case, "case_id") for case in cases]
    if set(audit_cases) != set(case_ids):
        raise ValueError("internal audit case coverage does not match corpus")

    packet_cases: list[dict[str, Any]] = []
    for case in cases:
        case_id = _required_string(case, "case_id")
        audit_case = audit_cases[case_id]
        if audit_case.get("filing_identity") != "verified":
            raise ValueError(f"internal filing identity is not verified: {case_id}")
        if audit_case.get("primary_source_support") != "verified":
            raise ValueError(f"internal source support is not verified: {case_id}")
        nodes = _mapping_list(case.get("nodes"), f"nodes for {case_id}")
        edges = _mapping_list(case.get("edges"), f"edges for {case_id}")
        evidence = _string_list(case.get("evidence"), f"evidence for {case_id}")
        challenge = _mapping(case.get("challenge"), f"challenge for {case_id}")
        packet_cases.append(
            {
                "case_id": case_id,
                "partition": _required_string(case, "partition"),
                "source": dict(_mapping(case.get("source"), f"source for {case_id}")),
                "source_locator": {
                    "section": _required_string(audit_case, "section"),
                    "search_terms": _string_list(
                        audit_case.get("search_terms"), f"search terms for {case_id}"
                    ),
                },
                "evidence_notes": [
                    {"index": index, "text": text} for index, text in enumerate(evidence)
                ],
                "nodes": [_packet_node(node, case_id) for node in nodes],
                "edges": [
                    {
                        "index": index,
                        "source": edge["source"],
                        "relation": edge["relation"],
                        "target": edge["target"],
                    }
                    for index, edge in enumerate(edges)
                ],
                "objective_nodes": list(challenge["objective_nodes"]),
                "constraint_nodes": list(challenge["constraint_nodes"]),
            }
        )
    packet = {
        "schema_version": REVIEW_SCHEMA_VERSION,
        "packet_id": "CHM-V-C1-REVIEW-PACKET",
        "corpus_id": corpus_id,
        "annotation_author_id": author_id,
        "manifest_sha256": _sha256(manifest_file),
        "source_sha256": _sha256(source_file),
        "internal_audit_sha256": _sha256(audit_file),
        "review_protocol_sha256": _sha256(protocol_file),
        "case_count": len(packet_cases),
        "decision_values": sorted(REVIEW_DECISIONS),
        "node_rating_axes": list(ANNOTATED_FEATURE_NAMES),
        "node_rating_semantics": "docs/BUSINESS_GRAPH_SEMANTICS.md#numeric-features",
        "cases": packet_cases,
    }
    output = Path(output_path)
    _write_json(output, packet)
    return packet


def build_review_template(packet_path: str | Path, output_path: str | Path) -> dict[str, Any]:
    """Build a blank, case-aligned review document from a frozen packet."""

    packet_file = Path(packet_path)
    packet = _load_json_mapping(packet_file, "review packet")
    cases = _mapping_list(packet.get("cases"), "review packet cases")
    review_cases: list[dict[str, Any]] = []
    for case in cases:
        review_cases.append(
            {
                "case_id": case["case_id"],
                "filing_identity": None,
                "evidence_notes": [
                    {"index": item["index"], "decision": None}
                    for item in _mapping_list(case.get("evidence_notes"), "evidence notes")
                ],
                "nodes": [
                    {
                        "id": item["id"],
                        "decision": None,
                        "ratings": [
                            {
                                "axis": rating["axis"],
                                "value": rating["value"],
                                "decision": None,
                            }
                            for rating in _mapping_list(
                                item.get("ratings"), "node ratings"
                            )
                        ],
                    }
                    for item in _mapping_list(case.get("nodes"), "nodes")
                ],
                "edges": [
                    {"index": item["index"], "decision": None}
                    for item in _mapping_list(case.get("edges"), "edges")
                ],
                "objective_nodes": [
                    {"id": node_id, "decision": None}
                    for node_id in _string_list(case.get("objective_nodes"), "objective nodes")
                ],
                "constraint_nodes": [
                    {"id": node_id, "decision": None}
                    for node_id in _string_list(case.get("constraint_nodes"), "constraint nodes")
                ],
                "overall": None,
                "notes": "",
            }
        )
    template = {
        "schema_version": REVIEW_SCHEMA_VERSION,
        "review_id": "",
        "corpus_id": packet["corpus_id"],
        "manifest_sha256": packet["manifest_sha256"],
        "packet_sha256": _sha256(packet_file),
        "reviewer": {
            "id": "",
            "independent_of_annotation_author": None,
            "conflicts_disclosed": "",
        },
        "attestation": {
            "registered_source_documents_opened": None,
            "no_candidate_outputs_seen": None,
            "no_model_arm_outputs_seen": None,
        },
        "cases": review_cases,
    }
    output = Path(output_path)
    _write_json(output, template)
    return template


def evaluate_review_gate(
    manifest_path: str | Path,
    packet_path: str | Path,
    protocol_path: str | Path,
    reviews_dir: str | Path,
    *,
    source_path: str | Path | None = None,
    audit_path: str | Path | None = None,
    status_path: str | Path | None = None,
) -> dict[str, Any]:
    """Validate independent reviews and return the generation-gate decision."""

    manifest_file = Path(manifest_path)
    packet_file = Path(packet_path)
    protocol_file = Path(protocol_path)
    source_file = (
        Path(source_path) if source_path is not None else manifest_file.parent / "source_cases.yaml"
    )
    audit_file = (
        Path(audit_path)
        if audit_path is not None
        else manifest_file.parent / "internal_source_audit.yaml"
    )
    validate_evaluation_corpus(manifest_file)
    protocol = _load_yaml_mapping(protocol_file, "review protocol")
    packet = _load_json_mapping(packet_file, "review packet")
    manifest = _load_json_mapping(manifest_file, "manifest")
    corpus_id = _required_string(manifest, "corpus_id")
    if packet.get("schema_version") != REVIEW_SCHEMA_VERSION:
        raise ValueError("unsupported review packet schema version")
    if _required_string(packet, "corpus_id") != corpus_id:
        raise ValueError("review packet corpus does not match manifest")
    if _required_string(packet, "manifest_sha256") != _sha256(manifest_file):
        raise ValueError("review packet manifest hash is stale")
    if _required_string(packet, "source_sha256") != _sha256(source_file):
        raise ValueError("review packet source hash is stale")
    if _required_string(packet, "internal_audit_sha256") != _sha256(audit_file):
        raise ValueError("review packet internal audit hash is stale")
    if _required_string(packet, "review_protocol_sha256") != _sha256(protocol_file):
        raise ValueError("review packet protocol hash is stale")
    packet_cases = _mapping_list(packet.get("cases"), "review packet cases")
    counts = _mapping(manifest.get("counts"), "manifest counts")
    if int(packet.get("case_count", -1)) != len(packet_cases) or len(packet_cases) != int(
        counts["cases"]
    ):
        raise ValueError("review packet case count does not match manifest")
    expected_cases = {_required_string(case, "case_id"): case for case in packet_cases}
    author_id = _required_string(packet, "annotation_author_id")
    minimum_reviewers = int(protocol.get("minimum_independent_reviewers", 1))
    review_files = sorted(Path(reviews_dir).glob("*.review.json"))
    accepted_reviews = 0
    changed_reviews = 0
    review_ids: list[str] = []
    for review_file in review_files:
        review = _load_json_mapping(review_file, "independent review")
        review_id = _required_string(review, "review_id")
        reviewer = _mapping(review.get("reviewer"), f"reviewer for {review_id}")
        reviewer_id = _required_string(reviewer, "id")
        if reviewer_id == author_id:
            raise ValueError("annotation author cannot satisfy independent review")
        if reviewer.get("independent_of_annotation_author") is not True:
            raise ValueError(f"reviewer independence is not attested: {review_id}")
        _required_string(reviewer, "conflicts_disclosed")
        attestation = _mapping(review.get("attestation"), f"attestation for {review_id}")
        for field in (
            "registered_source_documents_opened",
            "no_candidate_outputs_seen",
            "no_model_arm_outputs_seen",
        ):
            if attestation.get(field) is not True:
                raise ValueError(f"review attestation is incomplete: {review_id}/{field}")
        if _required_string(review, "corpus_id") != corpus_id:
            raise ValueError(f"review corpus does not match: {review_id}")
        if _required_string(review, "manifest_sha256") != _sha256(manifest_file):
            raise ValueError(f"review manifest hash is stale: {review_id}")
        if _required_string(review, "packet_sha256") != _sha256(packet_file):
            raise ValueError(f"review packet hash is stale: {review_id}")
        review_cases = _mapping_list(review.get("cases"), f"cases for {review_id}")
        if {_required_string(case, "case_id") for case in review_cases} != set(expected_cases):
            raise ValueError(f"review case coverage is incomplete: {review_id}")
        requested_change = False
        for review_case in review_cases:
            case_id = _required_string(review_case, "case_id")
            requested_change |= _validate_case_review(review_case, expected_cases[case_id])
        if requested_change:
            changed_reviews += 1
        else:
            accepted_reviews += 1
        review_ids.append(review_id)
    if changed_reviews:
        status = "failed"
        reason = "at_least_one_independent_review_requested_changes"
    elif accepted_reviews >= minimum_reviewers:
        status = "passed"
        reason = "minimum_complete_independent_reviews_accepted_all_cases"
    else:
        status = "blocked"
        reason = "no_complete_independent_review"
    result = {
        "schema_version": REVIEW_SCHEMA_VERSION,
        "corpus_id": corpus_id,
        "status": status,
        "reason": reason,
        "generation_allowed": status == "passed",
        "minimum_independent_reviewers": minimum_reviewers,
        "review_files": len(review_files),
        "accepted_reviews": accepted_reviews,
        "changed_reviews": changed_reviews,
        "review_ids": review_ids,
        "manifest_sha256": _sha256(manifest_file),
        "packet_sha256": _sha256(packet_file),
    }
    if status_path is not None:
        _write_json(Path(status_path), result)
    return result


def assert_review_gate_passed(result: Mapping[str, Any]) -> None:
    """Prevent any evidence-bearing generation while source review is incomplete."""

    if result.get("status") != "passed" or result.get("generation_allowed") is not True:
        raise RuntimeError("Venture Corpus C1 independent review gate is not passed")


def _validate_case_review(review: Mapping[str, Any], expected: Mapping[str, Any]) -> bool:
    decisions: list[str] = [_decision(review.get("filing_identity"), "filing identity")]
    decisions.extend(
        _indexed_decisions(review.get("evidence_notes"), expected.get("evidence_notes"), "index")
    )
    decisions.extend(_node_review_decisions(review.get("nodes"), expected.get("nodes")))
    decisions.extend(_indexed_decisions(review.get("edges"), expected.get("edges"), "index"))
    decisions.extend(
        _node_decisions(review.get("objective_nodes"), expected.get("objective_nodes"))
    )
    decisions.extend(
        _node_decisions(review.get("constraint_nodes"), expected.get("constraint_nodes"))
    )
    requested_change = any(decision != "verified" for decision in decisions)
    overall = review.get("overall")
    expected_overall = "needs_change" if requested_change else "accept"
    if overall != expected_overall:
        raise ValueError(f"case review overall decision must be {expected_overall}")
    return requested_change


def _indexed_decisions(value: object, expected_value: object, key: str) -> list[str]:
    records = _mapping_list(value, "review decision records")
    expected = _mapping_list(expected_value, "expected records")
    expected_keys = [record[key] for record in expected]
    observed_keys = [record.get(key) for record in records]
    if observed_keys != expected_keys:
        raise ValueError("review decision records are not aligned with packet")
    return [_decision(record.get("decision"), "review decision") for record in records]


def _node_review_decisions(value: object, expected_value: object) -> list[str]:
    records = _mapping_list(value, "review node decisions")
    expected = _mapping_list(expected_value, "expected nodes")
    if [record.get("id") for record in records] != [record["id"] for record in expected]:
        raise ValueError("review node decisions are not aligned with packet")
    decisions: list[str] = []
    for record, expected_node in zip(records, expected, strict=True):
        decisions.append(_decision(record.get("decision"), "review node decision"))
        ratings = _mapping_list(record.get("ratings"), "review node rating decisions")
        expected_ratings = _mapping_list(expected_node.get("ratings"), "expected node ratings")
        if [rating.get("axis") for rating in ratings] != [
            rating["axis"] for rating in expected_ratings
        ]:
            raise ValueError("review node rating decisions are not aligned with packet")
        if [rating.get("value") for rating in ratings] != [
            rating["value"] for rating in expected_ratings
        ]:
            raise ValueError("review node rating values do not match packet")
        decisions.extend(
            _decision(rating.get("decision"), "review node rating decision")
            for rating in ratings
        )
    return decisions


def _node_decisions(value: object, expected_value: object) -> list[str]:
    records = _mapping_list(value, "review node decisions")
    expected = _string_list(expected_value, "expected node IDs")
    if [record.get("id") for record in records] != expected:
        raise ValueError("review node decisions are not aligned with packet")
    return [_decision(record.get("decision"), "review node decision") for record in records]


def _decision(value: object, name: str) -> str:
    if not isinstance(value, str) or value not in REVIEW_DECISIONS:
        raise ValueError(f"{name} must use a registered review decision")
    return value


def _packet_node(node: Mapping[str, Any], case_id: str) -> dict[str, Any]:
    features = _mapping(node.get("annotated_features"), f"annotated features for {case_id}")
    if set(features) != set(ANNOTATED_FEATURE_NAMES):
        raise ValueError(f"annotated feature axes do not match semantics: {case_id}/{node['id']}")
    ratings: list[dict[str, Any]] = []
    for axis in ANNOTATED_FEATURE_NAMES:
        value = features[axis]
        if not isinstance(value, (int, float)):
            raise TypeError(f"annotated feature must be numeric: {case_id}/{node['id']}/{axis}")
        ratings.append({"axis": axis, "value": float(value)})
    return {
        "id": node["id"],
        "type": node["type"],
        "label": node["label"],
        "ratings": ratings,
    }


def _load_json_mapping(path: Path, name: str) -> Mapping[str, Any]:
    return _mapping(json.loads(path.read_text(encoding="utf-8")), name)


def _load_yaml_mapping(path: Path, name: str) -> Mapping[str, Any]:
    return _mapping(yaml.safe_load(path.read_text(encoding="utf-8")), name)


def _read_jsonl(path: Path) -> list[Mapping[str, Any]]:
    return [
        _mapping(json.loads(line), path.name)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line
    ]


def _mapping(value: object, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{name} must be a mapping")
    return value


def _mapping_list(value: object, name: str) -> list[Mapping[str, Any]]:
    if not isinstance(value, list) or not all(isinstance(item, Mapping) for item in value):
        raise TypeError(f"{name} must be a list of mappings")
    return list(value)


def _string_list(value: object, name: str) -> list[str]:
    if not isinstance(value, Sequence) or isinstance(value, str):
        raise TypeError(f"{name} must be a string list")
    normalized = list(value)
    if not normalized or not all(isinstance(item, str) and item for item in normalized):
        raise ValueError(f"{name} must be a non-empty string list")
    return normalized


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
