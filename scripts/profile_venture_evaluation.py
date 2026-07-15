"""Profile Venture Corpus C1 and write its decision-oriented quality report."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from chimera.data.evaluation import EvaluationCorpus, validate_evaluation_corpus
from chimera.data.review import evaluate_review_gate
from chimera.data.semantics import FEATURE_NAMES


def _signature(*arrays: np.ndarray[Any, Any]) -> str:
    digest = hashlib.sha256()
    for array in arrays:
        digest.update(np.ascontiguousarray(array).tobytes())
    return digest.hexdigest()


def build_report(manifest_path: Path) -> dict[str, Any]:
    validation = validate_evaluation_corpus(manifest_path)
    corpus = EvaluationCorpus(manifest_path)
    manifest = corpus.manifest
    base = manifest_path.parent
    with np.load(base / "graphs.npz", allow_pickle=False) as archive:
        arrays = {name: archive[name].copy() for name in archive.files}

    active_nodes = arrays["graph_node_mask"].sum(axis=1)
    active_edges = (arrays["graph_edge_types"] > 0).sum(axis=(1, 2))
    active_features = arrays["graph_node_features"][arrays["graph_node_mask"]]
    topology_signatures = {
        _signature(
            arrays["graph_node_types"][index],
            arrays["graph_edge_types"][index],
            arrays["graph_node_mask"][index],
        )
        for index in range(len(corpus))
    }
    source_records = [record["source"] for record in corpus.cases]
    review_statuses = Counter(str(record["annotation_review_status"]) for record in corpus.cases)
    node_type_counts = Counter(
        int(node_type)
        for index in range(len(corpus))
        for node_type in arrays["graph_node_types"][index][arrays["graph_node_mask"][index]]
    )
    edge_type_counts = Counter(
        int(edge_type)
        for edge_type in arrays["graph_edge_types"].reshape(-1)
        if int(edge_type) > 0
    )
    audit = yaml.safe_load(
        (base / "internal_source_audit.yaml").read_text(encoding="utf-8")
    )
    audit_cases = audit["cases"]
    review_gate = evaluate_review_gate(
        manifest_path,
        base / "reviewer_packet.json",
        base / "review_protocol.yaml",
        base / "reviews",
    )
    review_passed = review_gate["status"] == "passed"
    findings = [
        {
            "id": "C1-Q001",
            "severity": "high",
            "confidence": "high",
            "status": "closed" if review_passed else "open",
            "finding": (
                "Internal primary-source checks cover all ten cases; "
                f"independent reviews accepted: {review_gate['accepted_reviews']}/"
                f"{review_gate['minimum_independent_reviewers']}."
            ),
            "impact": "Source interpretation errors could affect both experiment arms.",
            "remediation": (
                "A second reviewer must verify every evidence note, node, edge and "
                "challenge mask before generation."
            ),
        },
        {
            "id": "C1-Q002",
            "severity": "high",
            "confidence": "high",
            "status": "accepted_limitation",
            "finding": "C1 contains public-company SEC filings only.",
            "impact": (
                "Results cannot be generalized to startups, private firms or "
                "non-business ideation."
            ),
            "remediation": (
                "Create later Venture Corpus lines with source-compatible private "
                "and early-stage cases."
            ),
        },
        {
            "id": "C1-Q003",
            "severity": "high",
            "confidence": "high",
            "status": "expected_by_design",
            "finding": (
                "The corpus contains no creativity, feasibility or commercial "
                "outcome labels."
            ),
            "impact": "C1 alone cannot support any creativity claim.",
            "remediation": (
                "Collect preregistered blind human ratings; keep them outside model "
                "input and calibration."
            ),
        },
        {
            "id": "C1-Q004",
            "severity": "medium",
            "confidence": "high",
            "status": "accepted_limitation",
            "finding": "The primary analysis has eight independent evaluation cases.",
            "impact": "Case-level uncertainty will be wide even with many candidate ratings.",
            "remediation": (
                "Treat H001 as an initial falsification test and preregister C2 "
                "before expanding claims."
            ),
        },
    ]
    boundary = manifest["temporal_boundary"]
    return {
        "report_id": "CHM-VENTURE-C1-QUALITY",
        "generated_at": manifest["source_accessed_at"],
        "corpus_id": manifest["corpus_id"],
        "release_status": manifest["release_status"],
        "intended_grain": "one organization filing, one challenge and one typed numeric graph",
        "intended_use": "preregistered CHM-V-H001 generation and blind evaluation",
        "validation": validation,
        "dimensions": {
            "completeness": {
                "cases_with_source": sum(bool(record.get("source")) for record in corpus.cases),
                "cases_with_evidence": sum(bool(record.get("evidence")) for record in corpus.cases),
                "cases_with_challenge": sum(
                    bool(record.get("challenge")) for record in corpus.cases
                ),
                "review_statuses": dict(sorted(review_statuses.items())),
            },
            "uniqueness": {
                "unique_case_ids": len({record["case_id"] for record in corpus.cases}),
                "unique_organizations": len({record["organization"] for record in source_records}),
                "unique_ciks": len({record["cik"] for record in source_records}),
                "unique_accessions": len({record["accession"] for record in source_records}),
                "unique_numeric_graphs": len(set(corpus.numeric_signatures())),
                "unique_topologies": len(topology_signatures),
            },
            "validity": {
                "feature_names": list(FEATURE_NAMES),
                "feature_min": float(active_features.min()),
                "feature_max": float(active_features.max()),
                "finite_features": bool(np.isfinite(active_features).all()),
                "numeric_archive_has_text_or_objects": any(
                    array.dtype.kind in {"O", "S", "U"} for array in arrays.values()
                ),
            },
            "consistency": {
                "objective_nodes_per_case": arrays["objective_mask"].sum(axis=1).tolist(),
                "constraint_nodes_per_case": arrays["constraint_mask"].sum(axis=1).tolist(),
                "matched_brief_count": len(corpus.briefs),
                "case_aligned_briefs": [record["case_id"] for record in corpus.cases]
                == [record["case_id"] for record in corpus.briefs],
            },
            "integrity": {
                "manifest_hash_validation": "passed",
                "safe_npz_allow_pickle_false": "passed",
                "partition_counts": validation,
            },
            "timeliness": {
                "pretraining_max_period_end": boundary["pretraining_max_period_end"],
                "evaluation_min_period_end": boundary["evaluation_min_period_end"],
                "strict_time_boundary": "passed",
            },
            "volume_and_shape": {
                "node_count_min": int(active_nodes.min()),
                "node_count_max": int(active_nodes.max()),
                "node_count_median": float(np.median(active_nodes)),
                "edge_count_min": int(active_edges.min()),
                "edge_count_max": int(active_edges.max()),
                "edge_count_median": float(np.median(active_edges)),
                "node_type_counts": dict(sorted(node_type_counts.items())),
                "edge_type_counts": dict(sorted(edge_type_counts.items())),
            },
            "leakage": {
                "organization_overlap_count": boundary["organization_overlap_count"],
                "cik_overlap_count": boundary["cik_overlap_count"],
                "accession_overlap_count": boundary["accession_overlap_count"],
                "calibration_excluded_from_primary_analysis": True,
                "outcome_labels_present": False,
            },
            "source_review": {
                "internal_auditor_id": audit["auditor_id"],
                "internal_auditor_independent": audit["independent"],
                "internal_filing_identity_verified": sum(
                    record["filing_identity"] == "verified" for record in audit_cases
                ),
                "internal_primary_source_support_verified": sum(
                    record["primary_source_support"] == "verified" for record in audit_cases
                ),
                "semantic_mapping_pending_independent_review": sum(
                    record["semantic_mapping"] == "pending_independent_review"
                    for record in audit_cases
                ),
                "gate": review_gate,
            },
        },
        "findings": findings,
        "fitness_for_use": {
            "corpus_build_and_protocol_preregistration": "fit",
            "candidate_generation": (
                "fit" if review_passed else "blocked_pending_C1-Q001"
            ),
            "creativity_claim": "not_fit_until_blind_ratings_and_locked_analysis",
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("datasets/venture_corpus_c1/manifest.json"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("datasets/venture_corpus_c1/quality_report.json"),
    )
    arguments = parser.parse_args()
    report = build_report(arguments.manifest)
    with arguments.output.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report["fitness_for_use"], sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
