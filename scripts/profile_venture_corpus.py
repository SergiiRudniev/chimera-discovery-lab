"""Produce an inspectable data-quality profile for Venture Corpus C0."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

SPLITS = ("train", "validation", "test")
MODEL_ARRAYS = (
    "graph_node_types",
    "graph_node_features",
    "graph_edge_types",
    "graph_node_mask",
    "edit_operations",
    "edit_source_nodes",
    "edit_target_nodes",
    "edit_node_types",
    "edit_edge_types",
    "edit_step_mask",
    "next_node_types",
    "next_node_features",
    "next_edge_types",
    "next_node_mask",
    "scores",
)


def _summary(values: np.ndarray[Any, Any]) -> dict[str, float]:
    return {
        "min": float(np.min(values)),
        "median": float(np.median(values)),
        "max": float(np.max(values)),
        "mean": float(np.mean(values)),
    }


def _record_hash(archive: dict[str, np.ndarray[Any, Any]], index: int) -> str:
    digest = hashlib.sha256()
    for name in MODEL_ARRAYS:
        digest.update(name.encode("ascii"))
        digest.update(np.ascontiguousarray(archive[name][index]).tobytes())
    return digest.hexdigest()


def profile(manifest_path: Path) -> dict[str, Any]:
    base = manifest_path.parent
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    records = [
        json.loads(line)
        for line in (base / "records.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    metadata = {
        split: [record for record in records if record["split"] == split]
        for split in SPLITS
    }
    split_profiles: dict[str, Any] = {}
    all_hashes: list[str] = []
    all_cases: dict[str, set[str]] = {}

    for split in SPLITS:
        with np.load(base / f"{split}.npz", allow_pickle=False) as loaded:
            archive = {name: loaded[name].copy() for name in MODEL_ARRAYS}
        count = int(archive["graph_node_types"].shape[0])
        active_nodes = archive["graph_node_mask"].sum(axis=1)
        active_edges = (archive["graph_edge_types"] > 0).sum(axis=(1, 2))
        edit_steps = archive["edit_step_mask"].sum(axis=1)
        record_hashes = [_record_hash(archive, index) for index in range(count)]
        all_hashes.extend(record_hashes)
        case_ids = {str(record["case_id"]) for record in metadata[split]}
        all_cases[split] = case_ids
        corruption_counts = Counter(
            corruption
            for record in metadata[split]
            for corruption in record["corruptions"]
        )
        target_mask = archive["next_node_mask"]
        target_features = archive["next_node_features"][target_mask]
        split_profiles[split] = {
            "records": count,
            "cases": len(case_ids),
            "duplicate_numeric_records": count - len(set(record_hashes)),
            "active_nodes": _summary(active_nodes),
            "active_edges": _summary(active_edges),
            "edit_steps": _summary(edit_steps),
            "feature_min": float(target_features.min()),
            "feature_max": float(target_features.max()),
            "feature_zero_rate": float(np.mean(target_features == 0)),
            "score_mean": {
                name: float(archive["scores"][:, index].mean())
                for index, name in enumerate(manifest["score_names"])
            },
            "corruptions": dict(sorted(corruption_counts.items())),
            "all_arrays_numeric": all(
                np.issubdtype(archive[name].dtype, np.number)
                or np.issubdtype(archive[name].dtype, np.bool_)
                for name in MODEL_ARRAYS
            ),
            "nonfinite_values": sum(
                int(np.size(archive[name]) - np.isfinite(archive[name]).sum())
                for name in MODEL_ARRAYS
            ),
        }

    split_overlap = {
        "train_validation": sorted(all_cases["train"] & all_cases["validation"]),
        "train_test": sorted(all_cases["train"] & all_cases["test"]),
        "validation_test": sorted(all_cases["validation"] & all_cases["test"]),
    }
    duplicate_count = len(all_hashes) - len(set(all_hashes))
    score_mean_ranges = {
        name: float(
            max(split_profiles[split]["score_mean"][name] for split in SPLITS)
            - min(split_profiles[split]["score_mean"][name] for split in SPLITS)
        )
        for name in manifest["score_names"]
    }
    limitations = [
        {
            "severity": "high",
            "finding": "Only ten independent source graphs are present.",
            "impact": (
                "Augmented transition count must not be treated as independent "
                "business-case count."
            ),
        },
        {
            "severity": "high",
            "finding": (
                "Targets are deterministic structural proxies, not observed "
                "business outcomes."
            ),
            "impact": "Score heads cannot support utility or feasibility claims.",
        },
        {
            "severity": "medium",
            "finding": "Annotations were produced by one schema author.",
            "impact": "Inter-rater reliability and ontology bias are unknown.",
        },
    ]
    if max(score_mean_ranges.values()) > 0.1:
        limitations.append(
            {
                "severity": "medium",
                "finding": "Proxy-score distributions differ across source-isolated splits.",
                "impact": (
                    "The small number of source companies can dominate validation and "
                    "test metrics."
                ),
            }
        )
    return {
        "corpus_id": manifest["corpus_id"],
        "grain": "one corrupted graph, one inverse edit program, one registered target",
        "profiled_at": manifest["source_accessed_at"],
        "splits": split_profiles,
        "integrity": {
            "total_records": len(all_hashes),
            "duplicate_numeric_records": duplicate_count,
            "split_case_overlap": split_overlap,
            "hash_manifest_validation": "covered by chimera validate-corpus",
        },
        "distribution": {"score_mean_range_across_splits": score_mean_ranges},
        "fitness": {
            "engineering_pretraining": "usable",
            "generalization_claim": "not_usable",
            "creativity_claim": "not_usable",
        },
        "limitations": limitations,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("datasets/venture_corpus_c0/manifest.json"),
    )
    parser.add_argument("--output", type=Path)
    arguments = parser.parse_args()
    report = json.dumps(profile(arguments.manifest), indent=2, sort_keys=True) + "\n"
    if arguments.output is None:
        print(report, end="")
    else:
        arguments.output.write_text(report, encoding="utf-8")


if __name__ == "__main__":
    main()
