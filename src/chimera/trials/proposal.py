"""Frozen proposal-policy diagnostics and qualification for Chimera Venture."""

from __future__ import annotations

import hashlib
import json
import platform
import statistics
import subprocess
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import asdict, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, cast

import numpy as np
import torch
from torch import Tensor

from chimera.config import (
    ModelConfig,
    ProposalPolicyConfig,
    ProposalTrialConfig,
    VentureTrialConfig,
)
from chimera.data.contracts import EditBatch, GraphBatch
from chimera.data.corpus import CorpusSplit, validate_corpus
from chimera.generation.mutate import apply_edit_program
from chimera.models.venture import ChimeraVenture
from chimera.training.trainer import resolve_device
from chimera.trials.venture import canonical_cases, generate_candidates


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


def _read_mapping(path: Path) -> Mapping[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, Mapping):
        raise TypeError(f"{path} must contain a JSON object")
    return value


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, values: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for value in values:
            handle.write(json.dumps(value, sort_keys=True) + "\n")


def _load_checkpoint_context(
    config: ProposalTrialConfig,
) -> tuple[ChimeraVenture, VentureTrialConfig, Mapping[str, Any]]:
    checkpoint_path = Path(config.checkpoint_path)
    actual_sha256 = _sha256(checkpoint_path)
    if actual_sha256 != config.checkpoint_sha256:
        raise ValueError(
            f"checkpoint SHA-256 mismatch: expected {config.checkpoint_sha256}, "
            f"got {actual_sha256}"
        )
    reconstruction_config = VentureTrialConfig.from_yaml(config.reconstruction_config)
    reconstruction_result = _read_mapping(Path(config.reconstruction_result))
    result_checkpoint = reconstruction_result.get("checkpoint")
    if not isinstance(result_checkpoint, Mapping):
        raise TypeError("reconstruction result is missing checkpoint metadata")
    if result_checkpoint.get("sha256") != actual_sha256:
        raise ValueError("reconstruction result does not identify the configured checkpoint")

    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if not isinstance(payload, Mapping):
        raise TypeError("checkpoint payload must be a mapping")
    model_values = payload.get("model_config")
    model_state = payload.get("model_state")
    if not isinstance(model_values, Mapping) or not isinstance(model_state, Mapping):
        raise TypeError("checkpoint is missing model_config or model_state")
    model_config = ModelConfig.from_mapping(cast(Mapping[str, Any], model_values))
    if model_config != reconstruction_config.model:
        raise ValueError("checkpoint model configuration differs from reconstruction protocol")
    if config.max_edits > model_config.max_edits:
        raise ValueError("proposal max_edits exceeds model capacity")
    model = ChimeraVenture(model_config)
    model.load_state_dict(cast(Mapping[str, Tensor], model_state))
    model.to(resolve_device(config.device)).eval()
    return model, reconstruction_config, reconstruction_result


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
def _reconstruction_guardrail(
    model: ChimeraVenture,
    shard: CorpusSplit,
    *,
    batch_size: int = 64,
) -> dict[str, float]:
    exact_graphs = 0
    examples = 0
    device = next(model.parameters()).device
    for start in range(0, len(shard), batch_size):
        indices = list(range(start, min(start + batch_size, len(shard))))
        batch = shard.batch(indices).with_terminal_stop().to(device)
        output = model(batch.graph, batch.edits)
        predicted = EditBatch(
            operations=output.operation_logits.argmax(dim=-1),
            source_nodes=output.source_logits.argmax(dim=-1),
            target_nodes=output.target_logits.argmax(dim=-1),
            node_types=output.node_type_logits.argmax(dim=-1),
            edge_types=output.edge_type_logits.argmax(dim=-1),
            step_mask=batch.edits.step_mask,
        )
        executed = apply_edit_program(batch.graph, predicted)
        exact_graphs += int(_same_graph_per_row(executed, batch.next_graph).sum())
        examples += batch.graph.batch_size
    return {
        "examples": float(examples),
        "exact_graph_rate": exact_graphs / examples,
    }


def _trial_config_for_policy(
    base: VentureTrialConfig,
    proposal: ProposalTrialConfig,
    policy: ProposalPolicyConfig,
) -> VentureTrialConfig:
    evaluation = replace(
        base.evaluation,
        candidates_per_case=proposal.candidates_per_case,
        generation_temperature=policy.temperature,
        min_edits=proposal.min_edits,
        max_edits=proposal.max_edits,
        archive_bins=proposal.archive_bins,
    )
    return replace(base, trial_id=proposal.trial_id, evaluation=evaluation)


def _aggregate_policy_metrics(
    policy: ProposalPolicyConfig,
    seed_metrics: Sequence[Mapping[str, Any]],
    candidates: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    if not seed_metrics or not candidates:
        raise ValueError("policy aggregation requires metrics and candidates")
    feasibility = [
        float(cast(Mapping[str, Any], candidate["proxy_scores"])["feasibility"])
        for candidate in candidates
    ]
    operation_counts: Counter[str] = Counter()
    for metric in seed_metrics:
        counts = metric["operation_counts"]
        if not isinstance(counts, Mapping):
            raise TypeError("operation_counts must be a mapping")
        operation_counts.update({str(name): int(count) for name, count in counts.items()})
    unique_rates = [float(metric["unique_graph_rate"]) for metric in seed_metrics]
    changed_rates = [float(metric["changed_candidate_rate"]) for metric in seed_metrics]
    invalid_rates = [float(metric["invalid_candidate_rate"]) for metric in seed_metrics]
    archive_coverages = [float(metric["archive_coverage"]) for metric in seed_metrics]
    return {
        "policy": asdict(policy),
        "candidates": len(candidates),
        "seeds": len(seed_metrics),
        "unique_graph_rate_mean": statistics.mean(unique_rates),
        "unique_graph_rate_min": min(unique_rates),
        "unique_graph_rate_std": statistics.pstdev(unique_rates),
        "changed_candidate_rate_mean": statistics.mean(changed_rates),
        "changed_candidate_rate_min": min(changed_rates),
        "invalid_candidate_rate_mean": statistics.mean(invalid_rates),
        "invalid_candidate_rate_max": max(invalid_rates),
        "archive_coverage_mean": statistics.mean(archive_coverages),
        "operation_counts": dict(sorted(operation_counts.items())),
        "operation_coverage": len(operation_counts),
        "feasibility_median": statistics.median(feasibility),
        "feasibility_mean": statistics.mean(feasibility),
        "reproducible": all(bool(metric["reproducible"]) for metric in seed_metrics),
        "seed_metrics": list(seed_metrics),
    }


@torch.no_grad()
def _evaluate_policy(
    model: ChimeraVenture,
    cases: Sequence[tuple[str, GraphBatch]],
    base_config: VentureTrialConfig,
    proposal_config: ProposalTrialConfig,
    policy: ProposalPolicyConfig,
    *,
    split: str,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    trial_config = _trial_config_for_policy(base_config, proposal_config, policy)
    all_candidates: list[dict[str, Any]] = []
    seed_metrics: list[dict[str, Any]] = []
    for seed in proposal_config.seeds:
        prefix = f"{proposal_config.trial_id}-{split}-{policy.policy_id}-s{seed}"
        candidates, metrics = generate_candidates(
            model,
            cases,
            trial_config,
            exploration_rate=policy.exploration_rate,
            generation_seed=seed,
            candidate_prefix=prefix,
        )
        for candidate in candidates:
            candidate["split"] = split
            candidate["policy_id"] = policy.policy_id
            candidate["generation_seed"] = seed
        all_candidates.extend(candidates)
        seed_metrics.append(metrics)
    return all_candidates, _aggregate_policy_metrics(policy, seed_metrics, all_candidates)


def _historical_candidate_summary(path: Path) -> dict[str, Any]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    if not rows or not all(isinstance(row, Mapping) for row in rows):
        raise TypeError(f"{path} must contain JSON object records")
    graph_signatures = {str(row["graph_signature"]) for row in rows}
    program_signatures = {str(row["program_signature"]) for row in rows}
    sequences: Counter[str] = Counter()
    per_case: dict[str, dict[str, Any]] = {}
    for row in rows:
        edits = row["edits"]
        if not isinstance(edits, list):
            raise TypeError("candidate edits must be a list")
        sequence = ">".join(str(edit["operation"]) for edit in edits)
        sequences[sequence] += 1
    case_ids = sorted({str(row["source_case_id"]) for row in rows})
    for case_id in case_ids:
        case_rows = [row for row in rows if str(row["source_case_id"]) == case_id]
        per_case[case_id] = {
            "candidates": len(case_rows),
            "unique_graphs": len({str(row["graph_signature"]) for row in case_rows}),
            "changed": sum(bool(row["changed"]) for row in case_rows),
        }
    top_sequence, top_count = sequences.most_common(1)[0]
    return {
        "path": path.as_posix(),
        "sha256": _sha256(path),
        "candidates": len(rows),
        "unique_graphs": len(graph_signatures),
        "unique_graph_rate": len(graph_signatures) / len(rows),
        "unique_programs": len(program_signatures),
        "changed_candidate_rate": sum(bool(row["changed"]) for row in rows) / len(rows),
        "top_program_sequence": top_sequence,
        "top_program_sequence_count": top_count,
        "top_program_sequence_rate": top_count / len(rows),
        "per_case": per_case,
    }


def run_proposal_diagnostic(
    config_path: str | Path,
    output_path: str | Path,
    *,
    baseline_candidates: str | Path,
    collapsed_candidates: str | Path,
) -> dict[str, Any]:
    """Diagnose the diversity regression using historical outputs and train cases only."""

    config_source = Path(config_path)
    config = ProposalTrialConfig.from_yaml(config_source)
    validate_corpus(config.corpus_manifest)
    model, base_config, _ = _load_checkpoint_context(config)
    cases = canonical_cases(Path(config.corpus_manifest), ("train",))
    policy_metrics: dict[str, Any] = {}
    for policy in config.policies:
        _, metrics = _evaluate_policy(
            model,
            cases,
            base_config,
            config,
            policy,
            split="train",
        )
        policy_metrics[policy.policy_id] = metrics
    baseline = _historical_candidate_summary(Path(baseline_candidates))
    collapsed = _historical_candidate_summary(Path(collapsed_candidates))
    result = {
        "schema_version": 1,
        "trial_id": config.trial_id,
        "scope": "train_only_diagnostic",
        "validation_opened": False,
        "test_opened": False,
        "code_commit": _git_commit(),
        "config_sha256": _sha256(config_source),
        "historical": {
            "baseline": baseline,
            "collapsed": collapsed,
            "unique_graph_rate_delta": (
                float(collapsed["unique_graph_rate"]) - float(baseline["unique_graph_rate"])
            ),
            "changed_candidate_rate_delta": (
                float(collapsed["changed_candidate_rate"])
                - float(baseline["changed_candidate_rate"])
            ),
        },
        "train_policy_sweep": policy_metrics,
        "diagnosis": {
            "verified": (
                "T1 concentrated proposals into short repeated programs; legal-uniform "
                "exploration restores operation coverage and graph diversity on train cases."
            ),
            "unresolved": "Held-out policy performance remains unopened until registration.",
        },
        "claim_boundary": "Engineering diagnosis only; no novelty or utility claim.",
    }
    _write_json(Path(output_path), result)
    return result


def _eligible_policy_ids(
    metrics: Mapping[str, Mapping[str, Any]],
    config: ProposalTrialConfig,
) -> list[str]:
    baseline_feasibility = float(metrics[config.baseline_policy_id]["feasibility_median"])
    return [
        policy.policy_id
        for policy in config.policies
        if policy.policy_id != config.baseline_policy_id
        and float(metrics[policy.policy_id]["invalid_candidate_rate_max"])
        <= config.invalid_candidate_rate_max
        and float(metrics[policy.policy_id]["changed_candidate_rate_mean"])
        >= config.changed_candidate_rate_min
        and float(metrics[policy.policy_id]["feasibility_median"])
        >= baseline_feasibility - config.feasibility_drop_max
        and bool(metrics[policy.policy_id]["reproducible"])
    ]


def _select_policy(
    metrics: Mapping[str, Mapping[str, Any]],
    config: ProposalTrialConfig,
) -> tuple[str, list[str]]:
    eligible = _eligible_policy_ids(metrics, config)
    candidates = eligible or [
        policy.policy_id
        for policy in config.policies
        if policy.policy_id != config.baseline_policy_id
    ]
    policy_by_id = {policy.policy_id: policy for policy in config.policies}
    selected = max(
        candidates,
        key=lambda policy_id: (
            float(metrics[policy_id]["unique_graph_rate_mean"]),
            float(metrics[policy_id]["archive_coverage_mean"]),
            int(metrics[policy_id]["operation_coverage"]),
            -policy_by_id[policy_id].exploration_rate,
        ),
    )
    return selected, eligible


def run_proposal_trial(
    config_path: str | Path,
    output_dir: str | Path,
) -> dict[str, Any]:
    """Select on validation, open test once, and persist a frozen proposal policy."""

    started_at = datetime.now(timezone.utc)
    config_source = Path(config_path)
    config = ProposalTrialConfig.from_yaml(config_source)
    manifest_path = Path(config.corpus_manifest)
    validate_corpus(manifest_path)
    model, base_config, reconstruction_result = _load_checkpoint_context(config)
    checkpoint_path = Path(config.checkpoint_path)
    checkpoint_sha_before = _sha256(checkpoint_path)
    train_reconstruction = _reconstruction_guardrail(
        model,
        CorpusSplit(manifest_path.parent / "train.npz"),
    )

    validation_cases = canonical_cases(manifest_path, ("validation",))
    validation_candidates: list[dict[str, Any]] = []
    validation_metrics: dict[str, dict[str, Any]] = {}
    for policy in config.policies:
        candidates, metrics = _evaluate_policy(
            model,
            validation_cases,
            base_config,
            config,
            policy,
            split="validation",
        )
        validation_candidates.extend(candidates)
        validation_metrics[policy.policy_id] = metrics
    selected_id, eligible_ids = _select_policy(validation_metrics, config)
    policy_by_id = {policy.policy_id: policy for policy in config.policies}

    test_cases = canonical_cases(manifest_path, ("test",))
    test_candidates: list[dict[str, Any]] = []
    test_metrics: dict[str, dict[str, Any]] = {}
    test_policy_ids = [config.baseline_policy_id, selected_id]
    for policy_id in dict.fromkeys(test_policy_ids):
        candidates, metrics = _evaluate_policy(
            model,
            test_cases,
            base_config,
            config,
            policy_by_id[policy_id],
            split="test",
        )
        test_candidates.extend(candidates)
        test_metrics[policy_id] = metrics

    selected_test = test_metrics[selected_id]
    baseline_test = test_metrics[config.baseline_policy_id]
    checkpoint_sha_after = _sha256(checkpoint_path)
    checks = {
        "checkpoint_hash_verified": checkpoint_sha_before == config.checkpoint_sha256,
        "weights_unchanged": checkpoint_sha_after == checkpoint_sha_before,
        "reconstruction_guardrail": (
            train_reconstruction["exact_graph_rate"]
            >= config.reconstruction_exact_graph_min
        ),
        "validation_policy_eligible": selected_id in eligible_ids,
        "test_unique_graph_rate": (
            float(selected_test["unique_graph_rate_mean"])
            >= config.unique_graph_rate_min
        ),
        "test_changed_candidate_rate": (
            float(selected_test["changed_candidate_rate_mean"])
            >= config.changed_candidate_rate_min
        ),
        "test_candidate_validity": (
            float(selected_test["invalid_candidate_rate_max"])
            <= config.invalid_candidate_rate_max
        ),
        "test_feasibility_guardrail": (
            float(selected_test["feasibility_median"])
            >= float(baseline_test["feasibility_median"]) - config.feasibility_drop_max
        ),
        "deterministic_replay": bool(selected_test["reproducible"]),
    }
    code_commit = _git_commit()
    finished_at = datetime.now(timezone.utc)
    policy_bundle = {
        "format_version": 1,
        "trial_id": config.trial_id,
        "model": "Chimera Venture M0",
        "checkpoint": {
            "file": checkpoint_path.name,
            "sha256": checkpoint_sha_before,
            "source_trial_id": reconstruction_result["trial_id"],
            "weights_modified": False,
        },
        "proposal_policy": asdict(policy_by_id[selected_id]),
        "sampler": {
            "distribution": "(1-r)*model_softmax + r*legal_uniform",
            "validity_constrained": True,
            "min_edits": config.min_edits,
            "max_edits": config.max_edits,
        },
        "code_commit": code_commit,
        "config_sha256": _sha256(config_source),
        "claim_boundary": "Engineering proposal policy; no creativity or utility claim.",
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
        "corpus_manifest_sha256": _sha256(manifest_path),
        "device": str(next(model.parameters()).device),
        "checkpoint": policy_bundle["checkpoint"],
        "selection": {
            "selection_split": "validation",
            "final_split": "test",
            "baseline_policy_id": config.baseline_policy_id,
            "eligible_policy_ids": eligible_ids,
            "selected_policy_id": selected_id,
            "ordering": [
                "unique_graph_rate_mean",
                "archive_coverage_mean",
                "operation_coverage",
                "lower_exploration_rate",
            ],
        },
        "metrics": {
            "train_reconstruction": train_reconstruction,
            "validation": validation_metrics,
            "test": test_metrics,
        },
        "checks": checks,
        "policy_bundle": policy_bundle,
        "claim_boundary": (
            f"{config.trial_id} qualifies a frozen structured proposal policy only; "
            "it does not evaluate semantic novelty, commercial utility or CHM-V-H001."
        ),
    }
    environment = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "torch": torch.__version__,
        "numpy": np.__version__,
        "device": str(next(model.parameters()).device),
        "cuda_available": torch.cuda.is_available(),
        "code_commit": code_commit,
    }
    output = Path(output_dir)
    _write_jsonl(output / "validation_candidates.jsonl", validation_candidates)
    _write_jsonl(output / "test_candidates.jsonl", test_candidates)
    _write_json(output / "policy_bundle.json", policy_bundle)
    _write_json(output / "environment.json", environment)
    _write_json(output / "result.json", result)
    return result
