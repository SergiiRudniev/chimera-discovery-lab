"""Training, evaluation and structured generation for Venture Trial T0."""

from __future__ import annotations

import hashlib
import json
import math
import platform
import subprocess
from collections import Counter, defaultdict
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import Tensor

from chimera.config import VentureTrialConfig
from chimera.data.contracts import EditBatch, GraphBatch, TrainingBatch
from chimera.data.corpus import SPLITS, CorpusSplit, validate_corpus
from chimera.data.semantics import compute_proxy_scores
from chimera.generation.archive import ArchiveEntry, MapElitesArchive
from chimera.generation.mutate import apply_edit_program, validate_edit_program
from chimera.generation.sampler import sample_edit_program
from chimera.models.venture import ChimeraVenture
from chimera.schema import SCORE_NAMES, EdgeType, EditOperation, NodeType
from chimera.training.objectives import chimera_loss
from chimera.training.trainer import ChimeraTrainer


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_commit() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _batches(
    shard: CorpusSplit, batch_size: int, indices: Sequence[int] | None = None
) -> Iterator[TrainingBatch]:
    selected: Sequence[int] = range(len(shard)) if indices is None else indices
    for start in range(0, len(selected), batch_size):
        yield shard.batch(selected[start : start + batch_size])


def _same_graph_per_row(left: GraphBatch, right: GraphBatch) -> Tensor:
    categorical = (
        (left.node_types == right.node_types).all(dim=1)
        & (left.edge_types == right.edge_types).all(dim=(1, 2))
        & (left.node_mask == right.node_mask).all(dim=1)
    )
    numeric = torch.isclose(left.node_features, right.node_features, atol=1e-6, rtol=0.0).all(
        dim=(1, 2)
    )
    return categorical & numeric


@torch.no_grad()
def evaluate_split(
    trainer: ChimeraTrainer,
    shard: CorpusSplit,
    *,
    batch_size: int,
    indices: Sequence[int] | None = None,
) -> dict[str, float]:
    """Evaluate teacher-forced symbols and exact executed-graph reconstruction."""

    trainer.model.eval()
    loss_totals: defaultdict[str, float] = defaultdict(float)
    examples = 0
    active_tokens = 0
    operation_correct = 0
    exact_programs = 0
    exact_graphs = 0
    score_absolute_error = 0.0
    score_values = 0
    for raw_batch in _batches(shard, batch_size, indices):
        batch = raw_batch.with_terminal_stop().to(trainer.device)
        output = trainer.model(batch.graph, batch.edits)
        target_state = trainer.target_encoder(batch.next_graph).graph_state
        losses = chimera_loss(
            output,
            batch.edits,
            batch.scores,
            target_state,
            weights=trainer.loss_weights,
        )
        count = batch.graph.batch_size
        examples += count
        for name, value in losses.items():
            loss_totals[name] += float(value.detach().cpu()) * count

        predicted = EditBatch(
            operations=output.operation_logits.argmax(dim=-1),
            source_nodes=output.source_logits.argmax(dim=-1),
            target_nodes=output.target_logits.argmax(dim=-1),
            node_types=output.node_type_logits.argmax(dim=-1),
            edge_types=output.edge_type_logits.argmax(dim=-1),
            step_mask=batch.edits.step_mask,
        )
        mask = batch.edits.step_mask
        active_tokens += int(mask.sum())
        operation_correct += int(((predicted.operations == batch.edits.operations) & mask).sum())
        fields_equal = (
            (predicted.operations == batch.edits.operations)
            & (predicted.source_nodes == batch.edits.source_nodes)
            & (predicted.target_nodes == batch.edits.target_nodes)
            & (predicted.node_types == batch.edits.node_types)
            & (predicted.edge_types == batch.edits.edge_types)
        )
        exact_programs += int((fields_equal | ~mask).all(dim=1).sum())
        executed = apply_edit_program(batch.graph, predicted)
        exact_graphs += int(_same_graph_per_row(executed, batch.next_graph).sum())
        score_absolute_error += float(
            (torch.sigmoid(output.score_logits) - batch.scores).abs().sum().cpu()
        )
        score_values += int(batch.scores.numel())

    if examples == 0:
        raise ValueError("evaluation selection is empty")
    return {
        **{name: total / examples for name, total in sorted(loss_totals.items())},
        "operation_accuracy": operation_correct / max(active_tokens, 1),
        "exact_program_rate": exact_programs / examples,
        "exact_graph_rate": exact_graphs / examples,
        "score_mae": score_absolute_error / max(score_values, 1),
        "examples": float(examples),
    }


def _repeat_graph(graph: GraphBatch, count: int) -> GraphBatch:
    return GraphBatch(
        node_types=graph.node_types.repeat(count, 1),
        node_features=graph.node_features.repeat(count, 1, 1),
        edge_types=graph.edge_types.repeat(count, 1, 1),
        node_mask=graph.node_mask.repeat(count, 1),
    )


def _graph_signature(graph: GraphBatch, index: int) -> str:
    digest = hashlib.sha256()
    for tensor in (
        graph.node_types[index],
        graph.node_features[index],
        graph.edge_types[index],
        graph.node_mask[index],
    ):
        digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def _program_signature(edits: EditBatch, index: int) -> str:
    digest = hashlib.sha256()
    for tensor in (
        edits.operations[index],
        edits.source_nodes[index],
        edits.target_nodes[index],
        edits.node_types[index],
        edits.edge_types[index],
        edits.step_mask[index],
    ):
        digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def _canonical_cases(manifest_path: Path) -> list[tuple[str, GraphBatch]]:
    records: dict[str, list[Mapping[str, Any]]] = {split: [] for split in SPLITS}
    for line in (manifest_path.parent / "records.jsonl").read_text(encoding="utf-8").splitlines():
        record = json.loads(line)
        if not isinstance(record, Mapping):
            raise TypeError("corpus record sidecar entries must be mappings")
        records[str(record["split"])].append(record)
    cases: list[tuple[str, GraphBatch]] = []
    seen: set[str] = set()
    for split in SPLITS:
        shard = CorpusSplit(manifest_path.parent / f"{split}.npz")
        for index, record in enumerate(records[split]):
            case_id = str(record["case_id"])
            if case_id in seen:
                continue
            seen.add(case_id)
            cases.append((case_id, shard.batch([index]).next_graph))
    return cases


def _symbolic_edits(edits: EditBatch, index: int) -> list[dict[str, int | str]]:
    records: list[dict[str, int | str]] = []
    for step in range(edits.steps):
        if not bool(edits.step_mask[index, step]):
            continue
        operation = EditOperation(int(edits.operations[index, step]))
        node_type_id = int(edits.node_types[index, step])
        edge_type_id = int(edits.edge_types[index, step])
        record: dict[str, int | str] = {
            "step": step,
            "operation_id": int(operation),
            "operation": operation.name,
            "source_node": int(edits.source_nodes[index, step]),
            "target_node": int(edits.target_nodes[index, step]),
            "node_type_id": node_type_id,
            "edge_type_id": edge_type_id,
        }
        record["node_type"] = NodeType(node_type_id).name
        record["edge_type"] = EdgeType(edge_type_id).name
        records.append(record)
        if operation is EditOperation.STOP:
            break
    return records


@torch.no_grad()
def generate_candidates(
    model: ChimeraVenture,
    cases: Sequence[tuple[str, GraphBatch]],
    config: VentureTrialConfig,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    model.eval()
    evaluation = config.evaluation
    device = next(model.parameters()).device
    generator = torch.Generator(device=device.type).manual_seed(evaluation.generation_seed)
    archive = MapElitesArchive(
        bins=evaluation.archive_bins,
        bounds=((0.0, 1.0), (0.0, 1.0)),
    )
    candidates: list[dict[str, Any]] = []
    signatures: set[str] = set()
    operation_counts: Counter[str] = Counter()
    invalid = 0
    changed = 0

    for case_id, canonical_cpu in cases:
        canonical = _repeat_graph(canonical_cpu.to(device), evaluation.candidates_per_case)
        programs = sample_edit_program(
            model,
            canonical,
            max_edits=evaluation.max_edits,
            min_edits=evaluation.min_edits,
            temperature=evaluation.generation_temperature,
            generator=generator,
        )
        failures = validate_edit_program(canonical, programs)
        generated = apply_edit_program(canonical, programs)
        output = model(canonical, programs)
        predicted_scores = torch.sigmoid(output.score_logits)
        proxy_scores = compute_proxy_scores(generated)
        latents = model.encoder(generated).graph_state.detach().cpu().numpy().astype(np.float32)
        source_signature = _graph_signature(canonical, 0)

        for index in range(evaluation.candidates_per_case):
            candidate_id = f"{config.trial_id}-{len(candidates):04d}"
            signature = _graph_signature(generated, index)
            program_signature = _program_signature(programs, index)
            signatures.add(signature)
            is_changed = signature != source_signature
            changed += int(is_changed)
            invalid += int(bool(failures[index]))
            symbolic = _symbolic_edits(programs, index)
            for edit in symbolic:
                if edit["operation"] != EditOperation.STOP.name:
                    operation_counts[str(edit["operation"])] += 1
            active_edits = sum(edit["operation"] != EditOperation.STOP.name for edit in symbolic)
            structural_delta = active_edits / evaluation.max_edits
            feasibility = float(proxy_scores[index, 1].cpu())
            utility = float(proxy_scores[index, 0].cpu())
            novelty = archive.novelty(latents[index])
            retained = archive.add(
                ArchiveEntry(
                    candidate_id=candidate_id,
                    descriptors=(structural_delta, feasibility),
                    quality=utility,
                    latent=latents[index],
                )
            )
            candidates.append(
                {
                    "candidate_id": candidate_id,
                    "source_case_id": case_id,
                    "program_signature": program_signature,
                    "graph_signature": signature,
                    "valid": not failures[index],
                    "validation_errors": list(failures[index]),
                    "changed": is_changed,
                    "edits": symbolic,
                    "predicted_scores": {
                        name: float(predicted_scores[index, score_index].cpu())
                        for score_index, name in enumerate(SCORE_NAMES)
                    },
                    "proxy_scores": {
                        name: float(proxy_scores[index, score_index].cpu())
                        for score_index, name in enumerate(SCORE_NAMES)
                    },
                    "descriptors": {
                        "structural_delta": structural_delta,
                        "feasibility": feasibility,
                    },
                    "latent_novelty": novelty,
                    "archive_retained": retained,
                }
            )

    total = len(candidates)
    if total == 0:
        raise ValueError("candidate generation produced no records")
    replay_case = _repeat_graph(cases[0][1].to(device), evaluation.candidates_per_case)
    first_generator = torch.Generator(device=device.type).manual_seed(evaluation.generation_seed)
    second_generator = torch.Generator(device=device.type).manual_seed(evaluation.generation_seed)
    first = sample_edit_program(
        model,
        replay_case,
        max_edits=evaluation.max_edits,
        min_edits=evaluation.min_edits,
        temperature=evaluation.generation_temperature,
        generator=first_generator,
    )
    second = sample_edit_program(
        model,
        replay_case,
        max_edits=evaluation.max_edits,
        min_edits=evaluation.min_edits,
        temperature=evaluation.generation_temperature,
        generator=second_generator,
    )
    reproducible = all(
        torch.equal(left, right)
        for left, right in (
            (first.operations, second.operations),
            (first.source_nodes, second.source_nodes),
            (first.target_nodes, second.target_nodes),
            (first.node_types, second.node_types),
            (first.edge_types, second.edge_types),
            (first.step_mask, second.step_mask),
        )
    )
    summary = {
        "candidates": total,
        "source_cases": len(cases),
        "invalid_candidate_rate": invalid / total,
        "changed_candidate_rate": changed / total,
        "unique_graph_rate": len(signatures) / total,
        "unique_graphs": len(signatures),
        "operation_counts": dict(sorted(operation_counts.items())),
        "operation_coverage": len(operation_counts),
        "archive_entries": len(archive),
        "archive_coverage": archive.coverage,
        "reproducible": reproducible,
    }
    return candidates, summary


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, values: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for value in values:
            handle.write(json.dumps(value, sort_keys=True) + "\n")


def _save_inference_checkpoint(
    trainer: ChimeraTrainer,
    path: Path,
    *,
    config: VentureTrialConfig,
    code_commit: str,
    corpus_manifest_sha256: str,
    best_step: int,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "format_version": 1,
        "trial_id": config.trial_id,
        "hypothesis_id": config.hypothesis_id,
        "model_name": "Chimera Venture M0",
        "model_config": asdict(config.model),
        "model_state": trainer.model.state_dict(),
        "training_step": best_step,
        "code_commit": code_commit,
        "corpus_manifest_sha256": corpus_manifest_sha256,
        "claim_boundary": "engineering structural pretraining; no creativity claim",
    }
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def run_venture_trial(
    config_path: str | Path,
    output_dir: str | Path,
    *,
    checkpoint_dir: str | Path,
) -> dict[str, Any]:
    """Run a frozen structural-pretraining qualification and persist its artifacts."""

    config_source = Path(config_path)
    config = VentureTrialConfig.from_yaml(config_source)
    output = Path(output_dir)
    checkpoints = Path(checkpoint_dir)
    manifest_path = Path(config.evaluation.corpus_manifest)
    validate_corpus(manifest_path)
    code_commit = _git_commit()
    corpus_hash = _sha256(manifest_path)
    train = CorpusSplit(manifest_path.parent / "train.npz")
    validation = CorpusSplit(manifest_path.parent / "validation.npz")
    test = CorpusSplit(manifest_path.parent / "test.npz")
    trainer = ChimeraTrainer(ChimeraVenture(config.model), config.training)
    rng = np.random.default_rng(config.training.seed)
    order = rng.permutation(len(train))
    cursor = 0
    history: list[dict[str, Any]] = []
    first_loss: float | None = None
    best_validation_loss = math.inf
    best_step = 0
    best_state: dict[str, Tensor] | None = None
    last_training_loss = math.inf

    started_at = datetime.now(timezone.utc)
    for step in range(1, config.training.steps + 1):
        if cursor + config.training.batch_size > len(order):
            order = rng.permutation(len(train))
            cursor = 0
        indices = order[cursor : cursor + config.training.batch_size]
        cursor += config.training.batch_size
        train_metrics = trainer.train_step(train.batch(indices))
        last_training_loss = train_metrics["loss"]
        if first_loss is None:
            first_loss = train_metrics["loss"]
        history.append({"kind": "train", "step": step, **train_metrics})
        if step == 1 or step % config.evaluation.eval_interval == 0:
            validation_metrics = evaluate_split(
                trainer,
                validation,
                batch_size=config.evaluation.evaluation_batch_size,
            )
            history.append({"kind": "validation", "step": step, **validation_metrics})
            if validation_metrics["loss"] < best_validation_loss:
                best_validation_loss = validation_metrics["loss"]
                best_step = step
                best_state = {
                    name: value.detach().cpu().clone()
                    for name, value in trainer.model.state_dict().items()
                }
        print(
            json.dumps(
                {
                    "trial_id": config.trial_id,
                    "step": step,
                    "steps": config.training.steps,
                    "loss": train_metrics["loss"],
                },
                sort_keys=True,
            ),
            flush=True,
        )

    if best_state is None or first_loss is None:
        raise RuntimeError("training completed without a validation checkpoint")
    trainer.model.load_state_dict(best_state)
    train_metrics = evaluate_split(
        trainer, train, batch_size=config.evaluation.evaluation_batch_size
    )
    validation_metrics = evaluate_split(
        trainer, validation, batch_size=config.evaluation.evaluation_batch_size
    )
    test_metrics = evaluate_split(trainer, test, batch_size=config.evaluation.evaluation_batch_size)
    candidates, generation = generate_candidates(
        trainer.model, _canonical_cases(manifest_path), config
    )
    finished_at = datetime.now(timezone.utc)

    checkpoint_name = f"chimera-venture-m0-t0-step{best_step:06d}.pt"
    checkpoint_path = checkpoints / checkpoint_name
    _save_inference_checkpoint(
        trainer,
        checkpoint_path,
        config=config,
        code_commit=code_commit,
        corpus_manifest_sha256=corpus_hash,
        best_step=best_step,
    )
    checkpoint_manifest = {
        "file": checkpoint_name,
        "bytes": checkpoint_path.stat().st_size,
        "sha256": _sha256(checkpoint_path),
        "format_version": 1,
        "model": "Chimera Venture M0",
        "parameters": trainer.model.trainable_parameter_count(),
        "best_step": best_step,
        "release_tag": "venture-m0-t0",
        "claim_boundary": "engineering structural pretraining; no creativity claim",
    }
    checks = {
        "finite_training": all(math.isfinite(float(record["loss"])) for record in history),
        "loss_reduced": last_training_loss < first_loss,
        "memorization_exact_graph": (
            train_metrics["exact_graph_rate"] >= config.evaluation.memorization_exact_graph_min
        ),
        "candidate_validity": (
            generation["invalid_candidate_rate"] <= config.evaluation.invalid_candidate_rate_max
        ),
        "generation_reproducible": bool(generation["reproducible"]),
    }
    result = {
        "schema_version": 1,
        "trial_id": config.trial_id,
        "hypothesis_id": config.hypothesis_id,
        "status": "passed" if all(checks.values()) else "completed_with_gaps",
        "started_at": started_at.isoformat(),
        "finished_at": finished_at.isoformat(),
        "duration_seconds": (finished_at - started_at).total_seconds(),
        "code_commit": code_commit,
        "config_sha256": _sha256(config_source),
        "corpus_manifest_sha256": corpus_hash,
        "device": str(trainer.device),
        "best_step": best_step,
        "first_training_loss": first_loss,
        "last_training_loss": last_training_loss,
        "best_validation_loss": best_validation_loss,
        "metrics": {
            "train": train_metrics,
            "validation": validation_metrics,
            "test": test_metrics,
            "generation": generation,
        },
        "checks": checks,
        "checkpoint": checkpoint_manifest,
        "claim_boundary": (
            "T0 qualifies engineering behavior only; it does not evaluate novelty, "
            "commercial utility or the CHM-V-H001 language hypothesis."
        ),
    }
    environment = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "torch": torch.__version__,
        "numpy": np.__version__,
        "device": str(trainer.device),
        "cuda_available": torch.cuda.is_available(),
        "code_commit": code_commit,
    }
    _write_jsonl(output / "metrics.jsonl", history)
    _write_jsonl(output / "candidates.jsonl", candidates)
    _write_json(output / "checkpoint_manifest.json", checkpoint_manifest)
    _write_json(output / "environment.json", environment)
    _write_json(output / "result.json", result)
    return result
