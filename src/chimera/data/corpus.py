"""Build, load and validate the model-ready Chimera Venture corpus."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
from numpy.typing import NDArray

from chimera.config import ModelConfig
from chimera.data.contracts import EditBatch, GraphBatch, TrainingBatch
from chimera.data.semantics import (
    FEATURE_NAMES,
    compute_proxy_scores,
    validate_annotated_features,
    with_value_proximity,
)
from chimera.generation.mutate import apply_edit_program
from chimera.schema import EdgeType, EditOperation, NodeType

CORPUS_SCHEMA_VERSION = 1
SPLITS = ("train", "validation", "test")

_ARRAY_NAMES = (
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

_INPUT_ARRAY_NAMES = (
    "graph_node_types",
    "graph_node_features",
    "graph_edge_types",
    "graph_node_mask",
)

_TARGET_GRAPH_ARRAY_NAMES = (
    "next_node_types",
    "next_node_features",
    "next_edge_types",
    "next_node_mask",
)


class CorpusSplit:
    """A safe numeric-only shard; provenance is stored in JSONL sidecars."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        with np.load(self.path, allow_pickle=False) as archive:
            missing = sorted(set(_ARRAY_NAMES) - set(archive.files))
            if missing:
                raise ValueError(f"corpus shard is missing arrays: {', '.join(missing)}")
            self._arrays = {name: archive[name].copy() for name in _ARRAY_NAMES}
        self._validate_shapes()

    def __len__(self) -> int:
        return int(self._arrays["graph_node_types"].shape[0])

    def batch(self, indices: Sequence[int] | NDArray[Any]) -> TrainingBatch:
        selected = np.asarray(indices, dtype=np.int64)
        if selected.ndim != 1 or selected.size == 0:
            raise ValueError("indices must be a non-empty one-dimensional sequence")
        if np.any(selected < 0) or np.any(selected >= len(self)):
            raise IndexError("corpus batch index is out of range")

        def tensor(name: str) -> torch.Tensor:
            return torch.from_numpy(self._arrays[name][selected].copy())

        return TrainingBatch(
            graph=GraphBatch(
                node_types=tensor("graph_node_types").long(),
                node_features=tensor("graph_node_features").float(),
                edge_types=tensor("graph_edge_types").long(),
                node_mask=tensor("graph_node_mask").bool(),
            ),
            edits=EditBatch(
                operations=tensor("edit_operations").long(),
                source_nodes=tensor("edit_source_nodes").long(),
                target_nodes=tensor("edit_target_nodes").long(),
                node_types=tensor("edit_node_types").long(),
                edge_types=tensor("edit_edge_types").long(),
                step_mask=tensor("edit_step_mask").bool(),
            ),
            next_graph=GraphBatch(
                node_types=tensor("next_node_types").long(),
                node_features=tensor("next_node_features").float(),
                edge_types=tensor("next_edge_types").long(),
                node_mask=tensor("next_node_mask").bool(),
            ),
            scores=tensor("scores").float(),
        )

    def all(self) -> TrainingBatch:
        return self.batch(np.arange(len(self), dtype=np.int64))

    def numeric_signatures(self) -> tuple[str, ...]:
        return self._signatures(_ARRAY_NAMES)

    def input_signatures(self) -> tuple[str, ...]:
        return self._signatures(_INPUT_ARRAY_NAMES)

    def target_graph_signatures(self) -> tuple[str, ...]:
        return self._signatures(_TARGET_GRAPH_ARRAY_NAMES)

    def _signatures(self, names: Sequence[str]) -> tuple[str, ...]:
        signatures: list[str] = []
        for index in range(len(self)):
            digest = hashlib.sha256()
            for name in names:
                digest.update(name.encode("ascii"))
                digest.update(np.ascontiguousarray(self._arrays[name][index]).tobytes())
            signatures.append(digest.hexdigest())
        return tuple(signatures)

    def _validate_shapes(self) -> None:
        count, nodes = self._arrays["graph_node_types"].shape
        edits = self._arrays["edit_operations"].shape[1]
        expected = {
            "graph_node_features": (count, nodes, len(FEATURE_NAMES)),
            "graph_edge_types": (count, nodes, nodes),
            "graph_node_mask": (count, nodes),
            "edit_operations": (count, edits),
            "edit_source_nodes": (count, edits),
            "edit_target_nodes": (count, edits),
            "edit_node_types": (count, edits),
            "edit_edge_types": (count, edits),
            "edit_step_mask": (count, edits),
            "next_node_types": (count, nodes),
            "next_node_features": (count, nodes, len(FEATURE_NAMES)),
            "next_edge_types": (count, nodes, nodes),
            "next_node_mask": (count, nodes),
            "scores": (count, 3),
        }
        invalid = [name for name, shape in expected.items() if self._arrays[name].shape != shape]
        if invalid:
            raise ValueError(f"corpus shard has invalid shapes: {', '.join(invalid)}")


def build_corpus(
    source_path: str | Path,
    output_dir: str | Path,
    *,
    model_config: ModelConfig | None = None,
    examples_per_case: int = 64,
    seed: int = 1701,
) -> dict[str, Any]:
    """Build deterministic denoising transitions from source-grounded graphs."""

    if examples_per_case <= 0:
        raise ValueError("examples_per_case must be positive")
    config = model_config or ModelConfig()
    if config.node_numeric_features != len(FEATURE_NAMES):
        raise ValueError("model feature count does not match the registered corpus semantics")
    source = Path(source_path)
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    document = _load_source_document(source)
    cases = document["cases"]
    split_records: dict[str, list[dict[str, NDArray[Any] | str]]] = {split: [] for split in SPLITS}
    canonical_records: list[dict[str, Any]] = []
    split_case_ids: dict[str, list[str]] = {split: [] for split in SPLITS}

    for case_index, case_value in enumerate(cases):
        case = _mapping(case_value, "case")
        case_id = _required_string(case, "case_id")
        split = _required_string(case, "split")
        if split not in split_records:
            raise ValueError(f"case {case_id} uses unknown split {split}")
        target, canonical = graph_from_annotated_case(case, config)
        canonical_records.append(canonical)
        split_case_ids[split].append(case_id)
        rng = np.random.default_rng(seed + case_index * 1009)
        scores = compute_proxy_scores(target)[0]
        seen_signatures: set[str] = set()
        example_index = 0
        attempts = 0
        while example_index < examples_per_case:
            depth = 1 + example_index % min(3, config.max_edits)
            graph, edits, corruption_names = _corrupt_graph(target, config, rng, depth)
            restored = apply_edit_program(graph, edits)
            _assert_same_graph(restored, target, case_id=case_id)
            record_id = f"CVC0-{split[:2].upper()}-{case_index:02d}-{example_index:03d}"
            record = _record_arrays(
                record_id,
                case_id,
                graph,
                edits,
                target,
                scores,
                corruption_names,
            )
            signature = _record_signature(record)
            if signature in seen_signatures:
                attempts += 1
                if attempts > examples_per_case * 100:
                    raise ValueError(f"unable to generate unique records for {case_id}")
                continue
            seen_signatures.add(signature)
            split_records[split].append(record)
            example_index += 1

    canonical_path = destination / "canonical_graphs.jsonl"
    with canonical_path.open("w", encoding="utf-8", newline="\n") as handle:
        for record in canonical_records:
            handle.write(json.dumps(record, sort_keys=True, ensure_ascii=False) + "\n")

    records_path = destination / "records.jsonl"
    with records_path.open("w", encoding="utf-8", newline="\n") as handle:
        for split in SPLITS:
            for record in split_records[split]:
                handle.write(
                    json.dumps(
                        {
                            "record_id": str(record["record_ids"]),
                            "case_id": str(record["case_ids"]),
                            "split": split,
                            "corruptions": str(record["corruption_names"]).split("+"),
                        },
                        sort_keys=True,
                    )
                    + "\n"
                )

    shard_paths: dict[str, Path] = {}
    for split, records in split_records.items():
        shard_path = destination / f"{split}.npz"
        _write_shard(shard_path, records)
        shard_paths[split] = shard_path

    files = {
        "canonical_graphs.jsonl": _file_metadata(canonical_path),
        "records.jsonl": _file_metadata(records_path),
    }
    for split, path in shard_paths.items():
        files[f"{split}.npz"] = _file_metadata(path)
    manifest = {
        "schema_version": CORPUS_SCHEMA_VERSION,
        "corpus_id": _required_string(document, "corpus_id"),
        "model_family": "Chimera Venture",
        "feature_schema": _required_string(document, "feature_schema"),
        "feature_names": list(FEATURE_NAMES),
        "score_names": ["utility_proxy", "feasibility_proxy", "coherence_proxy"],
        "seed": seed,
        "examples_per_case": examples_per_case,
        "max_nodes": config.max_nodes,
        "max_edits": config.max_edits,
        "source_accessed_at": _required_string(document, "accessed_at"),
        "source_document": {
            "path": source.name,
            "sha256": _sha256(source),
        },
        "counts": {
            "canonical_graphs": len(canonical_records),
            **{split: len(records) for split, records in split_records.items()},
            "total_transitions": sum(len(records) for records in split_records.values()),
        },
        "split_case_ids": split_case_ids,
        "files": files,
        "claim_boundary": "source-grounded structural denoising; no outcome or creativity labels",
    }
    manifest_path = destination / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    validate_corpus(manifest_path)
    return manifest


def validate_corpus(manifest_path: str | Path) -> dict[str, int]:
    """Validate hashes, split isolation, tensors and edit reconstruction."""

    path = Path(manifest_path)
    manifest_value = json.loads(path.read_text(encoding="utf-8"))
    manifest = _mapping(manifest_value, "manifest")
    if manifest.get("schema_version") != CORPUS_SCHEMA_VERSION:
        raise ValueError("unsupported corpus schema version")
    base = path.parent
    files = _mapping(manifest.get("files"), "manifest files")
    for name, metadata_value in files.items():
        metadata = _mapping(metadata_value, f"file metadata for {name}")
        file_path = base / name
        if _sha256(file_path) != _required_string(metadata, "sha256"):
            raise ValueError(f"corpus file hash mismatch: {name}")
        if file_path.stat().st_size != int(metadata["bytes"]):
            raise ValueError(f"corpus file size mismatch: {name}")

    split_case_ids = _mapping(manifest.get("split_case_ids"), "split_case_ids")
    records_by_split: dict[str, list[Mapping[str, Any]]] = {split: [] for split in SPLITS}
    for line in (base / "records.jsonl").read_text(encoding="utf-8").splitlines():
        record = _mapping(json.loads(line), "record sidecar entry")
        split = _required_string(record, "split")
        if split not in records_by_split:
            raise ValueError(f"record sidecar uses unknown split {split}")
        records_by_split[split].append(record)
    seen_cases: set[str] = set()
    seen_records: set[str] = set()
    seen_numeric_signatures: set[str] = set()
    seen_input_signatures: set[str] = set()
    input_targets: dict[str, str] = {}
    total = 0
    for split in SPLITS:
        registered_cases = {str(value) for value in split_case_ids[split]}
        if seen_cases & registered_cases:
            raise ValueError("case IDs cross corpus split boundaries")
        seen_cases |= registered_cases
        shard = CorpusSplit(base / f"{split}.npz")
        sidecar_cases = {_required_string(record, "case_id") for record in records_by_split[split]}
        if sidecar_cases != registered_cases:
            raise ValueError(f"shard case IDs do not match manifest: {split}")
        record_ids = [_required_string(record, "record_id") for record in records_by_split[split]]
        if len(record_ids) != len(shard) or seen_records & set(record_ids):
            raise ValueError(f"invalid or duplicate record IDs in {split} shard")
        seen_records |= set(record_ids)
        signatures = set(shard.numeric_signatures())
        if len(signatures) != len(shard) or seen_numeric_signatures & signatures:
            raise ValueError(f"duplicate numeric records in {split} shard")
        seen_numeric_signatures |= signatures
        input_signatures = shard.input_signatures()
        target_signatures = shard.target_graph_signatures()
        current_inputs = set(input_signatures)
        if seen_input_signatures & current_inputs:
            raise ValueError("input graph signatures cross corpus split boundaries")
        seen_input_signatures |= current_inputs
        for input_signature, target_signature in zip(
            input_signatures, target_signatures, strict=True
        ):
            registered_target = input_targets.setdefault(input_signature, target_signature)
            if registered_target != target_signature:
                raise ValueError("identical input graphs map to conflicting target graphs")
        batch = shard.all()
        batch.validate(feature_dim=len(FEATURE_NAMES), score_dimensions=3)
        if not torch.isfinite(batch.graph.node_features).all():
            raise ValueError(f"non-finite graph features in {split} shard")
        if not torch.isfinite(batch.scores).all():
            raise ValueError(f"non-finite scores in {split} shard")
        if torch.any((batch.graph.node_features < 0) | (batch.graph.node_features > 1)):
            raise ValueError(f"graph features outside [0, 1] in {split} shard")
        restored = apply_edit_program(batch.graph, batch.edits)
        _assert_same_graph(restored, batch.next_graph, case_id=split)
        total += len(shard)
    expected_total = int(_mapping(manifest.get("counts"), "counts")["total_transitions"])
    if total != expected_total:
        raise ValueError("manifest transition count does not match shards")
    return {"canonical_graphs": len(seen_cases), "transitions": total}


def _load_source_document(path: Path) -> Mapping[str, Any]:
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    document = _mapping(value, "source document")
    if document.get("schema_version") != CORPUS_SCHEMA_VERSION:
        raise ValueError("unsupported source graph schema version")
    cases = document.get("cases")
    if not isinstance(cases, list) or not cases:
        raise ValueError("source document must contain cases")
    return document


def graph_from_annotated_case(
    case: Mapping[str, Any], config: ModelConfig
) -> tuple[GraphBatch, dict[str, Any]]:
    """Convert one evidence-annotated case into the numeric graph contract."""
    case_id = _required_string(case, "case_id")
    nodes_value = case.get("nodes")
    edges_value = case.get("edges")
    if not isinstance(nodes_value, list) or not nodes_value:
        raise ValueError(f"case {case_id} must contain nodes")
    if not isinstance(edges_value, list) or not edges_value:
        raise ValueError(f"case {case_id} must contain edges")
    if len(nodes_value) > config.max_nodes:
        raise ValueError(f"case {case_id} exceeds max_nodes")

    node_types = torch.zeros((1, config.max_nodes), dtype=torch.long)
    node_features = torch.zeros(
        (1, config.max_nodes, config.node_numeric_features), dtype=torch.float32
    )
    node_mask = torch.zeros((1, config.max_nodes), dtype=torch.bool)
    node_index: dict[str, int] = {}
    canonical_nodes: list[dict[str, Any]] = []
    for index, node_value in enumerate(nodes_value):
        node = _mapping(node_value, f"node in {case_id}")
        node_id = _required_string(node, "id")
        if node_id in node_index:
            raise ValueError(f"duplicate node ID {node_id} in {case_id}")
        try:
            node_type = NodeType[_required_string(node, "type")]
        except KeyError as error:
            raise ValueError(f"unknown node type in {case_id}") from error
        if node_type is NodeType.PAD:
            raise ValueError("source graphs cannot declare PAD nodes")
        ratings_value = node.get("ratings")
        if not isinstance(ratings_value, list):
            raise TypeError(f"node {node_id} ratings must be a list")
        ratings = validate_annotated_features(ratings_value)
        node_index[node_id] = index
        node_types[0, index] = int(node_type)
        node_features[0, index, :6] = torch.tensor(ratings[:6])
        node_features[0, index, 7] = ratings[6]
        node_mask[0, index] = True
        canonical_nodes.append(
            {
                "id": node_id,
                "label": _required_string(node, "label"),
                "type": node_type.name,
                "annotated_features": dict(
                    zip(
                        (
                            "salience",
                            "evidence",
                            "control",
                            "immediacy",
                            "recurrence",
                            "scalability",
                            "risk",
                        ),
                        ratings,
                        strict=True,
                    )
                ),
            }
        )

    edge_types = torch.zeros((1, config.max_nodes, config.max_nodes), dtype=torch.long)
    canonical_edges: list[dict[str, str]] = []
    for edge_value in edges_value:
        if not isinstance(edge_value, list) or len(edge_value) != 3:
            raise TypeError(f"edges in {case_id} must be [source, relation, target]")
        source_id, relation_name, target_id = (str(value) for value in edge_value)
        if source_id not in node_index or target_id not in node_index:
            raise ValueError(f"edge in {case_id} references an unknown node")
        try:
            relation = EdgeType[relation_name]
        except KeyError as error:
            raise ValueError(f"unknown relation {relation_name} in {case_id}") from error
        if relation is EdgeType.NONE:
            raise ValueError("source graphs cannot declare NONE edges")
        source_index = node_index[source_id]
        target_index = node_index[target_id]
        if edge_types[0, source_index, target_index] != 0:
            raise ValueError(f"duplicate directed edge in {case_id}")
        edge_types[0, source_index, target_index] = int(relation)
        canonical_edges.append(
            {"source": source_id, "relation": relation.name, "target": target_id}
        )

    graph = with_value_proximity(
        GraphBatch(
            node_types=node_types,
            node_features=node_features,
            edge_types=edge_types,
            node_mask=node_mask,
        )
    )
    graph.validate(feature_dim=config.node_numeric_features)
    for index, node in enumerate(canonical_nodes):
        node["features"] = [float(value) for value in graph.node_features[0, index].tolist()]
    source = _mapping(case.get("source"), f"source for {case_id}")
    for required in ("organization", "form", "accession", "period_end", "url"):
        _required_string(source, required)
    evidence = case.get("evidence")
    if not isinstance(evidence, list) or not evidence:
        raise ValueError(f"case {case_id} must contain evidence notes")
    canonical = {
        "case_id": case_id,
        "split": _required_string(case, "split"),
        "source": dict(source),
        "evidence": [str(value) for value in evidence],
        "nodes": canonical_nodes,
        "edges": canonical_edges,
    }
    return graph, canonical


def _corrupt_graph(
    target: GraphBatch,
    config: ModelConfig,
    rng: np.random.Generator,
    depth: int,
) -> tuple[GraphBatch, EditBatch, tuple[str, ...]]:
    graph = target.clone()
    inverses: list[tuple[int, int, int, int, int]] = []
    names: list[str] = []
    used_nodes: set[int] = set()
    for corruption_index in range(depth):
        kinds = ["drop_edge", "substitute_node", "invert_edge"]
        offset = int(rng.integers(0, len(kinds)))
        ordered = kinds[offset:] + kinds[:offset]
        applied = False
        for kind in ordered:
            if kind == "drop_edge":
                candidates = torch.nonzero(graph.edge_types[0] > 0, as_tuple=False).tolist()
                if candidates:
                    source, target_node = candidates[int(rng.integers(0, len(candidates)))]
                    relation = int(graph.edge_types[0, source, target_node])
                    graph.edge_types[0, source, target_node] = 0
                    inverses.append((int(EditOperation.CONNECT), source, target_node, 0, relation))
                    applied = True
            elif kind == "substitute_node":
                candidates = [
                    int(index)
                    for index in torch.where(graph.node_mask[0])[0].tolist()
                    if int(index) not in used_nodes
                ]
                if candidates:
                    node = candidates[int(rng.integers(0, len(candidates)))]
                    original = int(graph.node_types[0, node])
                    replacement = 1 + int(rng.integers(0, len(NodeType) - 1))
                    if replacement == original:
                        replacement = 1 + replacement % (len(NodeType) - 1)
                    graph.node_types[0, node] = replacement
                    used_nodes.add(node)
                    inverses.append((int(EditOperation.SUBSTITUTE), node, node, original, 0))
                    applied = True
            else:
                candidates = [
                    (int(source), int(target_node))
                    for source, target_node in torch.nonzero(
                        graph.edge_types[0] > 0, as_tuple=False
                    ).tolist()
                    if source != target_node and graph.edge_types[0, target_node, source] == 0
                ]
                if candidates:
                    source, target_node = candidates[int(rng.integers(0, len(candidates)))]
                    relation = int(graph.edge_types[0, source, target_node])
                    graph.edge_types[0, source, target_node] = 0
                    graph.edge_types[0, target_node, source] = relation
                    inverses.append(
                        (
                            int(EditOperation.INVERT_RELATION),
                            target_node,
                            source,
                            0,
                            relation,
                        )
                    )
                    applied = True
            if applied:
                names.append(kind)
                break
        if not applied:
            raise ValueError(f"unable to construct corruption {corruption_index}")

    graph = with_value_proximity(graph)
    inverse_program = list(reversed(inverses))
    operations = torch.zeros((1, config.max_edits), dtype=torch.long)
    sources = torch.zeros_like(operations)
    targets = torch.zeros_like(operations)
    node_types = torch.zeros_like(operations)
    edge_types = torch.zeros_like(operations)
    step_mask = torch.zeros((1, config.max_edits), dtype=torch.bool)
    for index, (operation, source, target_node, node_type, edge_type) in enumerate(inverse_program):
        operations[0, index] = operation
        sources[0, index] = source
        targets[0, index] = target_node
        node_types[0, index] = node_type
        edge_types[0, index] = edge_type
        step_mask[0, index] = True
    return (
        graph,
        EditBatch(
            operations=operations,
            source_nodes=sources,
            target_nodes=targets,
            node_types=node_types,
            edge_types=edge_types,
            step_mask=step_mask,
        ),
        tuple(names),
    )


def _record_arrays(
    record_id: str,
    case_id: str,
    graph: GraphBatch,
    edits: EditBatch,
    target: GraphBatch,
    scores: torch.Tensor,
    corruption_names: tuple[str, ...],
) -> dict[str, NDArray[Any] | str]:
    return {
        "graph_node_types": graph.node_types[0].numpy().astype(np.int16),
        "graph_node_features": graph.node_features[0].numpy().astype(np.float32),
        "graph_edge_types": graph.edge_types[0].numpy().astype(np.int16),
        "graph_node_mask": graph.node_mask[0].numpy(),
        "edit_operations": edits.operations[0].numpy().astype(np.int16),
        "edit_source_nodes": edits.source_nodes[0].numpy().astype(np.int16),
        "edit_target_nodes": edits.target_nodes[0].numpy().astype(np.int16),
        "edit_node_types": edits.node_types[0].numpy().astype(np.int16),
        "edit_edge_types": edits.edge_types[0].numpy().astype(np.int16),
        "edit_step_mask": edits.step_mask[0].numpy(),
        "next_node_types": target.node_types[0].numpy().astype(np.int16),
        "next_node_features": target.node_features[0].numpy().astype(np.float32),
        "next_edge_types": target.edge_types[0].numpy().astype(np.int16),
        "next_node_mask": target.node_mask[0].numpy(),
        "scores": scores.numpy().astype(np.float32),
        "record_ids": record_id,
        "case_ids": case_id,
        "corruption_names": "+".join(corruption_names),
    }


def _write_shard(path: Path, records: list[dict[str, NDArray[Any] | str]]) -> None:
    if not records:
        raise ValueError(f"cannot write empty corpus shard: {path.name}")
    arrays: dict[str, NDArray[Any]] = {}
    for name in _ARRAY_NAMES:
        arrays[name] = np.stack([np.asarray(record[name]) for record in records])
    np.savez_compressed(path, **arrays)  # type: ignore[arg-type]


def _record_signature(record: Mapping[str, NDArray[Any] | str]) -> str:
    digest = hashlib.sha256()
    for name in _ARRAY_NAMES:
        digest.update(name.encode("ascii"))
        digest.update(np.ascontiguousarray(record[name]).tobytes())
    return digest.hexdigest()


def _assert_same_graph(left: GraphBatch, right: GraphBatch, *, case_id: str) -> None:
    if not torch.equal(left.node_types, right.node_types):
        raise ValueError(f"edit reconstruction changed node types for {case_id}")
    if not torch.equal(left.edge_types, right.edge_types):
        raise ValueError(f"edit reconstruction changed edges for {case_id}")
    if not torch.equal(left.node_mask, right.node_mask):
        raise ValueError(f"edit reconstruction changed node mask for {case_id}")
    if not torch.allclose(left.node_features, right.node_features, atol=1e-6, rtol=0):
        raise ValueError(f"edit reconstruction changed features for {case_id}")


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


def _required_string(values: Mapping[str, Any], name: str) -> str:
    value = values.get(name)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")
    return value
