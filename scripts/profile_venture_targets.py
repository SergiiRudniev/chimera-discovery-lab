"""Profile target ambiguity and loss-field relevance in Venture Corpus C0."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import numpy as np

SPLITS = ("train", "validation", "test")
INPUT_ARRAYS = (
    "graph_node_types",
    "graph_node_features",
    "graph_edge_types",
    "graph_node_mask",
)
TARGET_GRAPH_ARRAYS = (
    "next_node_types",
    "next_node_features",
    "next_edge_types",
    "next_node_mask",
)
PROGRAM_ARRAYS = (
    "edit_operations",
    "edit_source_nodes",
    "edit_target_nodes",
    "edit_node_types",
    "edit_edge_types",
    "edit_step_mask",
)
OPERATION_NAMES = {
    0: "STOP",
    1: "ADD_NODE",
    2: "CONNECT",
    3: "REWIRE",
    4: "TRANSFER_ROLE",
    5: "REMOVE_CONSTRAINT",
    6: "INVERT_RELATION",
    7: "SUBSTITUTE",
    8: "MERGE",
}
RELEVANT_ARGUMENTS = {
    0: frozenset(),
    1: frozenset({"source", "node_type", "edge_type"}),
    2: frozenset({"source", "target", "edge_type"}),
    3: frozenset({"source", "target", "edge_type"}),
    4: frozenset({"target", "node_type"}),
    5: frozenset({"source"}),
    6: frozenset({"source", "target"}),
    7: frozenset({"source", "node_type"}),
    8: frozenset({"source", "target"}),
}


def _row_hash(archive: dict[str, np.ndarray[Any, Any]], names: Iterable[str], index: int) -> str:
    digest = hashlib.sha256()
    for name in names:
        digest.update(name.encode("ascii"))
        digest.update(np.ascontiguousarray(archive[name][index]).tobytes())
    return digest.hexdigest()


def _majority_upper_bound(input_signatures: list[str], target_signatures: list[str]) -> float:
    targets_by_input: defaultdict[str, Counter[str]] = defaultdict(Counter)
    for input_signature, target_signature in zip(input_signatures, target_signatures, strict=True):
        targets_by_input[input_signature][target_signature] += 1
    correct = sum(max(counts.values()) for counts in targets_by_input.values())
    return correct / len(input_signatures)


def _ambiguous_examples(
    input_signatures: list[str],
    target_signatures: list[str],
    record_ids: list[str],
) -> list[list[str]]:
    records_by_input: defaultdict[str, list[tuple[str, str]]] = defaultdict(list)
    for input_signature, target_signature, record_id in zip(
        input_signatures, target_signatures, record_ids, strict=True
    ):
        records_by_input[input_signature].append((target_signature, record_id))
    return [
        [record_id for _, record_id in records]
        for records in records_by_input.values()
        if len({target for target, _ in records}) > 1
    ][:10]


def profile(manifest_path: Path) -> dict[str, Any]:
    base = manifest_path.parent
    sidecar = [
        json.loads(line)
        for line in (base / "records.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    metadata = {
        split: [record for record in sidecar if record["split"] == split] for split in SPLITS
    }
    split_profiles: dict[str, Any] = {}
    signatures_by_split: dict[str, set[str]] = {}
    total_relevant = 0
    total_argument_slots = 0

    for split in SPLITS:
        with np.load(base / f"{split}.npz", allow_pickle=False) as loaded:
            names = INPUT_ARRAYS + TARGET_GRAPH_ARRAYS + PROGRAM_ARRAYS
            archive = {name: loaded[name].copy() for name in names}
        count = int(archive["graph_node_types"].shape[0])
        record_ids = [str(record["record_id"]) for record in metadata[split]]
        input_signatures = [_row_hash(archive, INPUT_ARRAYS, index) for index in range(count)]
        target_graph_signatures = [
            _row_hash(archive, TARGET_GRAPH_ARRAYS, index) for index in range(count)
        ]
        program_signatures = [_row_hash(archive, PROGRAM_ARRAYS, index) for index in range(count)]
        first_operations = [str(int(value)) for value in archive["edit_operations"][:, 0]]
        inputs: Counter[str] = Counter(input_signatures)
        repeated_groups = {key for key, value in inputs.items() if value > 1}
        signatures_by_split[split] = set(input_signatures)

        operation_counts: Counter[str] = Counter()
        relevant = 0
        argument_slots = 0
        for operations, mask in zip(
            archive["edit_operations"], archive["edit_step_mask"], strict=True
        ):
            for operation_value, active in zip(operations, mask, strict=True):
                if not bool(active):
                    continue
                operation = int(operation_value)
                operation_counts[OPERATION_NAMES[operation]] += 1
                relevant += len(RELEVANT_ARGUMENTS[operation])
                argument_slots += 4
        total_relevant += relevant
        total_argument_slots += argument_slots

        split_profiles[split] = {
            "records": count,
            "unique_input_graphs": len(inputs),
            "repeated_input_groups": len(repeated_groups),
            "records_in_repeated_input_groups": sum(
                count_value
                for signature, count_value in inputs.items()
                if signature in repeated_groups
            ),
            "registered_program_majority_upper_bound": _majority_upper_bound(
                input_signatures, program_signatures
            ),
            "target_graph_majority_upper_bound": _majority_upper_bound(
                input_signatures, target_graph_signatures
            ),
            "first_operation_majority_upper_bound": _majority_upper_bound(
                input_signatures, first_operations
            ),
            "conflicting_program_examples": _ambiguous_examples(
                input_signatures, program_signatures, record_ids
            ),
            "conflicting_target_graph_examples": _ambiguous_examples(
                input_signatures, target_graph_signatures, record_ids
            ),
            "operation_counts": dict(sorted(operation_counts.items())),
            "irrelevant_argument_target_rate": 1.0 - relevant / max(argument_slots, 1),
        }

    cross_split_input_overlap = {
        "train_validation": len(signatures_by_split["train"] & signatures_by_split["validation"]),
        "train_test": len(signatures_by_split["train"] & signatures_by_split["test"]),
        "validation_test": len(signatures_by_split["validation"] & signatures_by_split["test"]),
    }
    train = split_profiles["train"]
    limitations: list[dict[str, str]] = []
    if train["registered_program_majority_upper_bound"] < 1.0:
        limitations.append(
            {
                "severity": "high",
                "finding": "Identical training inputs have conflicting edit-program targets.",
                "impact": (
                    "Exact registered-program reconstruction is not fully identifiable "
                    "from the model input."
                ),
            }
        )
    if train["target_graph_majority_upper_bound"] < 1.0:
        limitations.append(
            {
                "severity": "critical",
                "finding": "Identical training inputs have conflicting target graphs.",
                "impact": "Exact graph reconstruction has an irreducible label conflict.",
            }
        )
    limitations.append(
        {
            "severity": "high",
            "finding": (
                f"{1.0 - total_relevant / max(total_argument_slots, 1):.1%} of raw "
                "argument targets are irrelevant to their edit operation."
            ),
            "impact": (
                "An all-fields loss spends capacity predicting placeholder zeros and "
                "dilutes operation-specific supervision."
            ),
        }
    )
    return {
        "corpus_id": "CHM-VENTURE-C0",
        "grain": "one corrupted input graph mapped to one registered inverse program",
        "intended_use": "exact structural reconstruction pretraining",
        "splits": split_profiles,
        "integrity": {"cross_split_input_overlap": cross_split_input_overlap},
        "fitness": {
            "exact_target_graph_reconstruction": (
                "identifiable"
                if train["target_graph_majority_upper_bound"] == 1.0
                else "not_fully_identifiable"
            ),
            "exact_registered_program_reconstruction": (
                "identifiable"
                if train["registered_program_majority_upper_bound"] == 1.0
                else "not_fully_identifiable"
            ),
            "all_fields_argument_loss": "not_recommended",
        },
        "limitations": limitations,
        "recommended_automated_tests": [
            "identical input graphs must map to one target graph within each split",
            "cross-split input graph overlap must remain zero",
            "argument loss masks must follow the edit-operation schema",
        ],
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
