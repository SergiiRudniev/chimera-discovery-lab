"""Evidence-bearing, language-isolated evaluation corpora for Chimera Venture."""

from __future__ import annotations

import hashlib
import json
import re
import shutil
from collections.abc import Mapping
from datetime import date
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from numpy.typing import NDArray

from chimera.config import ModelConfig
from chimera.data.contracts import GraphBatch
from chimera.data.corpus import graph_from_annotated_case
from chimera.data.semantics import FEATURE_NAMES

EVALUATION_SCHEMA_VERSION = 1
PARTITIONS = ("calibration", "evaluation")
_ARRAY_NAMES = (
    "graph_node_types",
    "graph_node_features",
    "graph_edge_types",
    "graph_node_mask",
    "objective_mask",
    "constraint_mask",
    "case_indices",
)
_FIXED_EDGE_SIGNATURES = {
    "HAS_NEED": frozenset({("ACTOR", "NEED")}),
    "REACHES": frozenset({("CHANNEL", "ACTOR")}),
    "DELIVERS": frozenset({("ACTION", "VALUE")}),
    "PAYS": frozenset({("ACTOR", "REVENUE")}),
    "COSTS": frozenset({("ACTION", "COST")}),
    "PRODUCES": frozenset({("VALUE", "OUTCOME"), ("ACTION", "OUTCOME")}),
    "FEEDS_BACK": frozenset({("OUTCOME", "FEEDBACK")}),
}
_FORBIDDEN_EDGE_SIGNATURES = frozenset(
    {
        ("NEED", "DEPENDS_ON", "VALUE"),
        ("COST", "REDUCES", "OUTCOME"),
        ("ACTOR", "TRANSFERS_TO", "RESOURCE"),
    }
)


class EvaluationCorpus:
    """Numeric graph tensors plus separately loaded human-readable audit records."""

    def __init__(self, manifest_path: str | Path) -> None:
        self.manifest_path = Path(manifest_path)
        manifest_value = json.loads(self.manifest_path.read_text(encoding="utf-8"))
        self.manifest = _mapping(manifest_value, "manifest")
        with np.load(self.manifest_path.parent / "graphs.npz", allow_pickle=False) as archive:
            missing = sorted(set(_ARRAY_NAMES) - set(archive.files))
            if missing:
                raise ValueError(f"evaluation archive is missing arrays: {', '.join(missing)}")
            self._arrays = {name: archive[name].copy() for name in _ARRAY_NAMES}
        self._validate_shapes()
        self.cases = tuple(_read_jsonl(self.manifest_path.parent / "cases.jsonl"))
        self.briefs = tuple(_read_jsonl(self.manifest_path.parent / "matched_briefs.jsonl"))
        if len(self.cases) != len(self) or len(self.briefs) != len(self):
            raise ValueError("evaluation sidecar length does not match numeric archive")

    def __len__(self) -> int:
        return int(self._arrays["graph_node_types"].shape[0])

    def graph(self, index: int) -> GraphBatch:
        if index < 0 or index >= len(self):
            raise IndexError("evaluation case index is out of range")
        selected = slice(index, index + 1)
        import torch

        graph = GraphBatch(
            node_types=torch.from_numpy(self._arrays["graph_node_types"][selected].copy()).long(),
            node_features=torch.from_numpy(
                self._arrays["graph_node_features"][selected].copy()
            ).float(),
            edge_types=torch.from_numpy(self._arrays["graph_edge_types"][selected].copy()).long(),
            node_mask=torch.from_numpy(self._arrays["graph_node_mask"][selected].copy()).bool(),
        )
        graph.validate(feature_dim=len(FEATURE_NAMES))
        return graph

    def challenge_masks(self, index: int) -> tuple[NDArray[np.bool_], NDArray[np.bool_]]:
        if index < 0 or index >= len(self):
            raise IndexError("evaluation case index is out of range")
        return (
            self._arrays["objective_mask"][index].copy(),
            self._arrays["constraint_mask"][index].copy(),
        )

    def numeric_signatures(self) -> tuple[str, ...]:
        signatures: list[str] = []
        for index in range(len(self)):
            digest = hashlib.sha256()
            for name in _ARRAY_NAMES[:-1]:
                digest.update(name.encode("ascii"))
                digest.update(np.ascontiguousarray(self._arrays[name][index]).tobytes())
            signatures.append(digest.hexdigest())
        return tuple(signatures)

    def _validate_shapes(self) -> None:
        count, nodes = self._arrays["graph_node_types"].shape
        expected = {
            "graph_node_features": (count, nodes, len(FEATURE_NAMES)),
            "graph_edge_types": (count, nodes, nodes),
            "graph_node_mask": (count, nodes),
            "objective_mask": (count, nodes),
            "constraint_mask": (count, nodes),
            "case_indices": (count,),
        }
        invalid = [name for name, shape in expected.items() if self._arrays[name].shape != shape]
        if invalid:
            raise ValueError(f"evaluation archive has invalid shapes: {', '.join(invalid)}")
        if not np.array_equal(self._arrays["case_indices"], np.arange(count)):
            raise ValueError("evaluation case indices are not contiguous")


def build_evaluation_corpus(
    source_path: str | Path,
    output_dir: str | Path,
    *,
    pretraining_manifest_path: str | Path,
    model_config: ModelConfig | None = None,
) -> dict[str, Any]:
    """Build deterministic C1 tensors and sidecars from registered source graphs."""

    source = Path(source_path)
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    config = model_config or ModelConfig()
    if config.node_numeric_features != len(FEATURE_NAMES):
        raise ValueError("model feature count does not match evaluation feature semantics")
    document = _load_yaml_mapping(source, "evaluation source")
    if document.get("schema_version") != EVALUATION_SCHEMA_VERSION:
        raise ValueError("unsupported evaluation source schema version")
    cases_value = document.get("cases")
    if not isinstance(cases_value, list) or not cases_value:
        raise ValueError("evaluation source must contain cases")
    pretraining_path = Path(pretraining_manifest_path)
    pretraining = _mapping(
        json.loads(pretraining_path.read_text(encoding="utf-8")), "pretraining manifest"
    )
    pretraining_cases = _read_jsonl(pretraining_path.parent / "canonical_graphs.jsonl")
    boundary = _validate_source_isolation(cases_value, pretraining_cases)

    canonical_records: list[dict[str, Any]] = []
    brief_records: list[dict[str, Any]] = []
    arrays: dict[str, list[NDArray[Any]]] = {name: [] for name in _ARRAY_NAMES}
    partition_case_ids: dict[str, list[str]] = {partition: [] for partition in PARTITIONS}
    for index, case_value in enumerate(cases_value):
        case = _mapping(case_value, "evaluation case")
        case_id = _required_string(case, "case_id")
        partition = _required_string(case, "partition")
        if partition not in partition_case_ids:
            raise ValueError(f"case {case_id} uses unknown partition {partition}")
        parser_case = dict(case)
        parser_case["split"] = partition
        graph, canonical = graph_from_annotated_case(parser_case, config)
        _validate_edge_semantics(canonical)
        challenge = _mapping(case.get("challenge"), f"challenge for {case_id}")
        objective_nodes = _string_list(challenge.get("objective_nodes"), "objective_nodes")
        constraint_nodes = _string_list(challenge.get("constraint_nodes"), "constraint_nodes")
        node_ids = {str(node["id"]) for node in canonical["nodes"]}
        if not set(objective_nodes + constraint_nodes) <= node_ids:
            raise ValueError(f"challenge for {case_id} references an unknown node")
        canonical["partition"] = canonical.pop("split")
        canonical["challenge"] = {
            "objective_nodes": objective_nodes,
            "constraint_nodes": constraint_nodes,
        }
        canonical["annotation_review_status"] = _required_string(
            case, "annotation_review_status"
        )
        canonical_records.append(canonical)
        brief_records.append(_matched_brief(canonical))
        partition_case_ids[partition].append(case_id)
        ordered_node_ids = [str(node["id"]) for node in canonical["nodes"]]
        objective_mask = np.zeros(config.max_nodes, dtype=np.bool_)
        constraint_mask = np.zeros(config.max_nodes, dtype=np.bool_)
        for node_id in objective_nodes:
            objective_mask[ordered_node_ids.index(node_id)] = True
        for node_id in constraint_nodes:
            constraint_mask[ordered_node_ids.index(node_id)] = True
        arrays["graph_node_types"].append(graph.node_types[0].numpy().astype(np.int16))
        arrays["graph_node_features"].append(
            graph.node_features[0].numpy().astype(np.float32)
        )
        arrays["graph_edge_types"].append(graph.edge_types[0].numpy().astype(np.int16))
        arrays["graph_node_mask"].append(graph.node_mask[0].numpy())
        arrays["objective_mask"].append(objective_mask)
        arrays["constraint_mask"].append(constraint_mask)
        arrays["case_indices"].append(np.asarray(index, dtype=np.int32))

    cases_path = destination / "cases.jsonl"
    briefs_path = destination / "matched_briefs.jsonl"
    _write_jsonl(cases_path, canonical_records)
    _write_jsonl(briefs_path, brief_records)
    graphs_path = destination / "graphs.npz"
    stacked_arrays = {name: np.stack(values) for name, values in arrays.items()}
    np.savez_compressed(graphs_path, **stacked_arrays)  # type: ignore[arg-type]
    protocol_paths: list[Path] = []
    for protocol_name in (
        "matched_baseline_protocol.yaml",
        "rating_protocol.yaml",
        "review_protocol.yaml",
        "internal_source_audit.yaml",
    ):
        registered_protocol = source.parent / protocol_name
        if not registered_protocol.is_file():
            raise FileNotFoundError(f"missing preregistered protocol: {registered_protocol}")
        output_protocol = destination / protocol_name
        if registered_protocol.resolve() != output_protocol.resolve():
            shutil.copyfile(registered_protocol, output_protocol)
        protocol_paths.append(output_protocol)
    files = {
        path.name: _file_metadata(path)
        for path in (cases_path, briefs_path, graphs_path, *protocol_paths)
    }
    counts = {partition: len(case_ids) for partition, case_ids in partition_case_ids.items()}
    manifest = {
        "schema_version": EVALUATION_SCHEMA_VERSION,
        "corpus_id": _required_string(document, "corpus_id"),
        "model_family": "Chimera Venture",
        "role": "preregistered_evaluation_only",
        "release_status": "provisional_pending_independent_annotation_review",
        "feature_schema": _required_string(document, "feature_schema"),
        "feature_names": list(FEATURE_NAMES),
        "max_nodes": config.max_nodes,
        "source_accessed_at": _required_string(document, "accessed_at"),
        "annotation_author_id": _required_string(document, "annotation_author_id"),
        "source_document": {"path": source.name, "sha256": _sha256(source)},
        "pretraining_corpus": {
            "corpus_id": _required_string(pretraining, "corpus_id"),
            "manifest_path": str(pretraining_path.as_posix()),
            "manifest_sha256": _sha256(pretraining_path),
        },
        "temporal_boundary": boundary,
        "counts": {"cases": len(canonical_records), **counts},
        "partition_case_ids": partition_case_ids,
        "files": files,
        "language_isolation": {
            "model_input": "graphs.npz numeric tensors only",
            "audit_sidecar": "cases.jsonl",
            "matched_baseline_input": "matched_briefs.jsonl",
        },
        "claim_boundary": (
            "source-grounded evaluation inputs; no creativity labels and no H001 outcomes"
        ),
    }
    manifest_path = destination / "manifest.json"
    with manifest_path.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    validate_evaluation_corpus(manifest_path)
    return manifest


def validate_evaluation_corpus(manifest_path: str | Path) -> dict[str, int]:
    """Validate file integrity, tensor ranges, partitions and source isolation evidence."""

    path = Path(manifest_path)
    manifest = _mapping(json.loads(path.read_text(encoding="utf-8")), "manifest")
    if manifest.get("schema_version") != EVALUATION_SCHEMA_VERSION:
        raise ValueError("unsupported evaluation corpus schema version")
    files = _mapping(manifest.get("files"), "manifest files")
    for name, metadata_value in files.items():
        metadata = _mapping(metadata_value, f"file metadata for {name}")
        file_path = path.parent / name
        if _sha256(file_path) != _required_string(metadata, "sha256"):
            raise ValueError(f"evaluation file hash mismatch: {name}")
        if file_path.stat().st_size != int(metadata["bytes"]):
            raise ValueError(f"evaluation file size mismatch: {name}")
    corpus = EvaluationCorpus(path)
    signatures = corpus.numeric_signatures()
    if len(set(signatures)) != len(signatures):
        raise ValueError("duplicate numeric evaluation graphs")
    case_ids = [_required_string(record, "case_id") for record in corpus.cases]
    if len(set(case_ids)) != len(case_ids):
        raise ValueError("duplicate evaluation case IDs")
    brief_case_ids = [_required_string(record, "case_id") for record in corpus.briefs]
    if brief_case_ids != case_ids:
        raise ValueError("matched baseline briefs are not case-aligned")
    observed_partitions = dict.fromkeys(PARTITIONS, 0)
    for index, record in enumerate(corpus.cases):
        _validate_edge_semantics(record)
        partition = _required_string(record, "partition")
        if partition not in observed_partitions:
            raise ValueError(f"unknown evaluation partition {partition}")
        observed_partitions[partition] += 1
        graph = corpus.graph(index)
        if not np.isfinite(graph.node_features.numpy()).all():
            raise ValueError(f"non-finite features in case {case_ids[index]}")
        if bool(((graph.node_features < 0) | (graph.node_features > 1)).any()):
            raise ValueError(f"features outside [0, 1] in case {case_ids[index]}")
        objective_mask, constraint_mask = corpus.challenge_masks(index)
        active_mask = graph.node_mask[0].numpy()
        if not objective_mask.any() or not constraint_mask.any():
            raise ValueError(f"empty challenge mask in case {case_ids[index]}")
        if np.any((objective_mask | constraint_mask) & ~active_mask):
            raise ValueError(f"challenge mask targets padding in case {case_ids[index]}")
    counts = _mapping(manifest.get("counts"), "counts")
    if int(counts["cases"]) != len(corpus):
        raise ValueError("manifest case count does not match archive")
    for partition, count in observed_partitions.items():
        if int(counts[partition]) != count:
            raise ValueError(f"manifest partition count does not match: {partition}")
    boundary = _mapping(manifest.get("temporal_boundary"), "temporal boundary")
    if int(boundary.get("organization_overlap_count", -1)) != 0:
        raise ValueError("evaluation organizations overlap pretraining corpus")
    if int(boundary.get("accession_overlap_count", -1)) != 0:
        raise ValueError("evaluation accessions overlap pretraining corpus")
    if int(boundary.get("cik_overlap_count", -1)) != 0:
        raise ValueError("evaluation CIKs overlap pretraining corpus")
    return {"cases": len(corpus), **observed_partitions}


def _validate_edge_semantics(case: Mapping[str, Any]) -> None:
    case_id = _required_string(case, "case_id")
    nodes = _mapping_list(case.get("nodes"), f"nodes for {case_id}")
    edges = _mapping_list(case.get("edges"), f"edges for {case_id}")
    node_types = {
        _required_string(node, "id"): _required_string(node, "type") for node in nodes
    }
    for edge in edges:
        source = _required_string(edge, "source")
        relation = _required_string(edge, "relation")
        target = _required_string(edge, "target")
        signature = (node_types[source], node_types[target])
        allowed = _FIXED_EDGE_SIGNATURES.get(relation)
        if allowed is not None and signature not in allowed:
            raise ValueError(
                f"edge role signature violates semantics: {case_id}/{source}/{relation}/{target}"
            )
        if (signature[0], relation, signature[1]) in _FORBIDDEN_EDGE_SIGNATURES:
            raise ValueError(
                f"known causal edge anti-pattern: {case_id}/{source}/{relation}/{target}"
            )
        if relation == "ENABLES" and signature[1] not in {"ACTION", "RESOURCE"}:
            raise ValueError(
                f"ENABLES target must be an action or resource: "
                f"{case_id}/{source}/{target}"
            )
        if relation == "BLOCKS" and signature[0] != "CONSTRAINT":
            raise ValueError(
                f"BLOCKS source must be a constraint: {case_id}/{source}/{target}"
            )


def _validate_source_isolation(
    cases: list[object], pretraining_cases: list[Mapping[str, Any]]
) -> dict[str, Any]:
    evaluation_sources = [
        _mapping(_mapping(case, "evaluation case").get("source"), "evaluation source")
        for case in cases
    ]
    pretraining_sources = [
        _mapping(case.get("source"), "pretraining source") for case in pretraining_cases
    ]
    organizations = [_required_string(source, "organization") for source in evaluation_sources]
    ciks = [_required_string(source, "cik") for source in evaluation_sources]
    accessions = [_required_string(source, "accession") for source in evaluation_sources]
    if len(set(organizations)) != len(organizations):
        raise ValueError("evaluation organizations must be unique")
    if len(set(ciks)) != len(ciks) or len(set(accessions)) != len(accessions):
        raise ValueError("evaluation CIKs and accessions must be unique")
    pretraining_organizations = {
        _required_string(source, "organization") for source in pretraining_sources
    }
    pretraining_accessions = {
        _required_string(source, "accession") for source in pretraining_sources
    }
    pretraining_ciks = {_sec_cik(source) for source in pretraining_sources}
    pretraining_periods = [
        date.fromisoformat(_required_string(source, "period_end"))
        for source in pretraining_sources
    ]
    evaluation_periods = [
        date.fromisoformat(_required_string(source, "period_end"))
        for source in evaluation_sources
    ]
    maximum_pretraining_period = max(pretraining_periods)
    if min(evaluation_periods) <= maximum_pretraining_period:
        raise ValueError("evaluation periods must be later than every pretraining period")
    organization_overlap = sorted(set(organizations) & pretraining_organizations)
    accession_overlap = sorted(set(accessions) & pretraining_accessions)
    cik_overlap = sorted(set(ciks) & pretraining_ciks)
    if organization_overlap or accession_overlap or cik_overlap:
        raise ValueError("evaluation sources overlap the pretraining corpus")
    return {
        "pretraining_max_period_end": maximum_pretraining_period.isoformat(),
        "evaluation_min_period_end": min(evaluation_periods).isoformat(),
        "rule": "evaluation period_end > maximum C0 period_end",
        "organization_overlap_count": len(organization_overlap),
        "accession_overlap_count": len(accession_overlap),
        "cik_overlap_count": len(cik_overlap),
    }


def _matched_brief(canonical: Mapping[str, Any]) -> dict[str, Any]:
    nodes = canonical["nodes"]
    edges = canonical["edges"]
    challenge = _mapping(canonical.get("challenge"), "canonical challenge")
    return {
        "case_id": canonical["case_id"],
        "partition": canonical["partition"],
        "instruction": (
            "Propose one business mechanism by changing at most three elements or relations. "
            "Address every objective and constraint. Return at most 120 words."
        ),
        "node_legend": [
            {"id": node["id"], "type": node["type"], "label": node["label"]}
            for node in nodes
        ],
        "relations": [
            [edge["source"], edge["relation"], edge["target"]] for edge in edges
        ],
        "objective_nodes": challenge["objective_nodes"],
        "constraint_nodes": challenge["constraint_nodes"],
    }


def _sec_cik(source: Mapping[str, Any]) -> str:
    registered = source.get("cik")
    if isinstance(registered, str) and registered.strip():
        return registered.zfill(10)
    match = re.search(r"/Archives/edgar/data/(\d+)/", _required_string(source, "url"))
    if match is None:
        raise ValueError("SEC source URL does not expose a CIK")
    return match.group(1).zfill(10)


def _load_yaml_mapping(path: Path, name: str) -> Mapping[str, Any]:
    return _mapping(yaml.safe_load(path.read_text(encoding="utf-8")), name)


def _read_jsonl(path: Path) -> list[Mapping[str, Any]]:
    return [
        _mapping(json.loads(line), path.name)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line
    ]


def _write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for record in records:
            handle.write(json.dumps(record, sort_keys=True, ensure_ascii=False) + "\n")


def _file_metadata(path: Path) -> dict[str, int | str]:
    return {"bytes": path.stat().st_size, "sha256": _sha256(path)}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _mapping(value: object, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{name} must be a mapping")
    return value


def _mapping_list(value: object, name: str) -> list[Mapping[str, Any]]:
    if not isinstance(value, list) or not all(isinstance(item, Mapping) for item in value):
        raise TypeError(f"{name} must be a list of mappings")
    return list(value)


def _required_string(values: Mapping[str, Any], name: str) -> str:
    value = values.get(name)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")
    return value


def _string_list(value: object, name: str) -> list[str]:
    if not isinstance(value, list) or not value or not all(isinstance(item, str) for item in value):
        raise ValueError(f"{name} must be a non-empty string list")
    return list(value)
