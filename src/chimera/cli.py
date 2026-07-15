"""Command-line entrypoint for model, corpus and research workflows."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

import torch

from chimera.config import ExperimentConfig
from chimera.data.corpus import CorpusSplit, build_corpus, validate_corpus
from chimera.data.evaluation import build_evaluation_corpus, validate_evaluation_corpus
from chimera.data.review import (
    build_review_packet,
    build_review_template,
    evaluate_review_gate,
)
from chimera.data.synthetic import make_synthetic_batch
from chimera.models.venture import ChimeraVenture
from chimera.research import load_research_registry
from chimera.training.trainer import ChimeraTrainer
from chimera.trials.proposal import run_proposal_diagnostic, run_proposal_trial
from chimera.trials.venture import run_venture_trial


def _inspect(config: ExperimentConfig) -> int:
    model = ChimeraVenture(config.model)
    payload = {
        "experiment_id": config.experiment_id,
        "trainable_parameters": model.trainable_parameter_count(),
        "max_nodes": config.model.max_nodes,
        "max_edits": config.model.max_edits,
        "hidden_dim": config.model.hidden_dim,
        "encoder_layers": config.model.encoder_layers,
        "decoder_layers": config.model.decoder_layers,
        "transition_layers": config.model.transition_layers,
    }
    print(json.dumps(payload, indent=2))
    return 0


def _smoke(config: ExperimentConfig, steps: int) -> int:
    if steps <= 0:
        raise ValueError("steps must be positive")
    trainer = ChimeraTrainer(ChimeraVenture(config.model), config.training)
    initial_loss: float | None = None
    final: dict[str, float] = {}
    for step in range(steps):
        batch = make_synthetic_batch(
            config.model,
            batch_size=config.training.batch_size,
            seed=config.training.seed,
            device=trainer.device,
        )
        final = trainer.train_step(batch)
        if initial_loss is None:
            initial_loss = final["loss"]
        print(json.dumps({"step": step + 1, **final}, sort_keys=True))
    summary = {
        "experiment_id": config.experiment_id,
        "steps": steps,
        "initial_loss": initial_loss,
        "final_loss": final["loss"],
        "finite": bool(torch.isfinite(torch.tensor(final["loss"]))),
    }
    print(json.dumps({"summary": summary}, sort_keys=True))
    return 0


def _build_corpus(arguments: argparse.Namespace) -> int:
    config = ExperimentConfig.from_yaml(arguments.config)
    manifest = build_corpus(
        arguments.source,
        arguments.output,
        model_config=config.model,
        examples_per_case=arguments.examples_per_case,
        seed=arguments.seed,
    )
    print(
        json.dumps(
            {
                "corpus_id": manifest["corpus_id"],
                "canonical_graphs": manifest["counts"]["canonical_graphs"],
                "transitions": manifest["counts"]["total_transitions"],
            },
            sort_keys=True,
        )
    )
    return 0


def _validate_corpus(arguments: argparse.Namespace) -> int:
    print(json.dumps(validate_corpus(arguments.manifest), sort_keys=True))
    return 0


def _build_evaluation_corpus(arguments: argparse.Namespace) -> int:
    config = ExperimentConfig.from_yaml(arguments.config)
    manifest = build_evaluation_corpus(
        arguments.source,
        arguments.output,
        pretraining_manifest_path=arguments.pretraining_manifest,
        model_config=config.model,
    )
    print(json.dumps({"corpus_id": manifest["corpus_id"], **manifest["counts"]}, sort_keys=True))
    return 0


def _validate_evaluation_corpus(arguments: argparse.Namespace) -> int:
    print(json.dumps(validate_evaluation_corpus(arguments.manifest), sort_keys=True))
    return 0


def _build_review_packet(arguments: argparse.Namespace) -> int:
    packet = build_review_packet(
        arguments.manifest,
        arguments.source,
        arguments.audit,
        arguments.protocol,
        arguments.output,
    )
    build_review_template(arguments.output, arguments.template)
    print(json.dumps({"packet_id": packet["packet_id"], "cases": packet["case_count"]}))
    return 0


def _validate_review_gate(arguments: argparse.Namespace) -> int:
    result = evaluate_review_gate(
        arguments.manifest,
        arguments.packet,
        arguments.protocol,
        arguments.reviews_dir,
        status_path=arguments.status_output,
    )
    print(json.dumps(result, sort_keys=True))
    return 0 if result["generation_allowed"] else 2


def _corpus_smoke(
    config: ExperimentConfig, manifest_path: Path, steps: int, batch_size: int
) -> int:
    if steps <= 0 or batch_size <= 0:
        raise ValueError("steps and batch_size must be positive")
    shard = CorpusSplit(manifest_path.parent / "train.npz")
    if batch_size > len(shard):
        raise ValueError("batch_size exceeds the training shard")
    batch = shard.batch(list(range(batch_size)))
    trainer = ChimeraTrainer(ChimeraVenture(config.model), config.training)
    initial_loss: float | None = None
    final: dict[str, float] = {}
    for step in range(steps):
        final = trainer.train_step(batch)
        if initial_loss is None:
            initial_loss = final["loss"]
        print(json.dumps({"step": step + 1, **final}, sort_keys=True))
    summary = {
        "experiment_id": config.experiment_id,
        "steps": steps,
        "batch_size": batch_size,
        "initial_loss": initial_loss,
        "final_loss": final["loss"],
        "finite": bool(torch.isfinite(torch.tensor(final["loss"]))),
    }
    print(json.dumps({"summary": summary}, sort_keys=True))
    return 0


def _venture_trial(arguments: argparse.Namespace) -> int:
    result = run_venture_trial(
        arguments.config,
        arguments.output,
        checkpoint_dir=arguments.checkpoint_dir,
    )
    print(
        json.dumps(
            {
                "trial_id": result["trial_id"],
                "status": result["status"],
                "best_step": result["best_step"],
                "checkpoint": result["checkpoint"]["file"],
            },
            sort_keys=True,
        )
    )
    return 0


def _proposal_diagnostic(arguments: argparse.Namespace) -> int:
    result = run_proposal_diagnostic(
        arguments.config,
        arguments.output,
        baseline_candidates=arguments.baseline_candidates,
        collapsed_candidates=arguments.collapsed_candidates,
    )
    print(
        json.dumps(
            {
                "trial_id": result["trial_id"],
                "scope": result["scope"],
                "output": str(arguments.output),
            },
            sort_keys=True,
        )
    )
    return 0


def _proposal_trial(arguments: argparse.Namespace) -> int:
    result = run_proposal_trial(arguments.config, arguments.output)
    print(
        json.dumps(
            {
                "trial_id": result["trial_id"],
                "status": result["status"],
                "selected_policy_id": result["selection"]["selected_policy_id"],
            },
            sort_keys=True,
        )
    )
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="chimera")
    subparsers = parser.add_subparsers(dest="command", required=True)
    inspect_parser = subparsers.add_parser("inspect")
    inspect_parser.add_argument("--config", type=Path, required=True)
    smoke_parser = subparsers.add_parser("smoke")
    smoke_parser.add_argument("--config", type=Path, required=True)
    smoke_parser.add_argument("--steps", type=int, default=10)
    research_parser = subparsers.add_parser("validate-research")
    research_parser.add_argument("--registry", type=Path, default=Path("research/registry.yaml"))
    corpus_parser = subparsers.add_parser("build-corpus")
    corpus_parser.add_argument(
        "--source",
        type=Path,
        default=Path("datasets/venture_corpus_c0/source_graphs.yaml"),
    )
    corpus_parser.add_argument("--output", type=Path, default=Path("datasets/venture_corpus_c0"))
    corpus_parser.add_argument(
        "--config", type=Path, default=Path("configs/venture/venture_m0_20m.yaml")
    )
    corpus_parser.add_argument("--examples-per-case", type=int, default=64)
    corpus_parser.add_argument("--seed", type=int, default=1701)
    corpus_validation_parser = subparsers.add_parser("validate-corpus")
    corpus_validation_parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("datasets/venture_corpus_c0/manifest.json"),
    )
    evaluation_parser = subparsers.add_parser("build-evaluation-corpus")
    evaluation_parser.add_argument(
        "--source", type=Path, default=Path("datasets/venture_corpus_c1/source_cases.yaml")
    )
    evaluation_parser.add_argument(
        "--output", type=Path, default=Path("datasets/venture_corpus_c1")
    )
    evaluation_parser.add_argument(
        "--pretraining-manifest",
        type=Path,
        default=Path("datasets/venture_corpus_c0/manifest.json"),
    )
    evaluation_parser.add_argument(
        "--config", type=Path, default=Path("configs/venture/venture_m0_20m.yaml")
    )
    evaluation_validation_parser = subparsers.add_parser("validate-evaluation-corpus")
    evaluation_validation_parser.add_argument(
        "--manifest", type=Path, default=Path("datasets/venture_corpus_c1/manifest.json")
    )
    review_packet_parser = subparsers.add_parser("build-review-packet")
    review_packet_parser.add_argument(
        "--manifest", type=Path, default=Path("datasets/venture_corpus_c1/manifest.json")
    )
    review_packet_parser.add_argument(
        "--source", type=Path, default=Path("datasets/venture_corpus_c1/source_cases.yaml")
    )
    review_packet_parser.add_argument(
        "--audit",
        type=Path,
        default=Path("datasets/venture_corpus_c1/internal_source_audit.yaml"),
    )
    review_packet_parser.add_argument(
        "--protocol",
        type=Path,
        default=Path("datasets/venture_corpus_c1/review_protocol.yaml"),
    )
    review_packet_parser.add_argument(
        "--output",
        type=Path,
        default=Path("datasets/venture_corpus_c1/reviewer_packet.json"),
    )
    review_packet_parser.add_argument(
        "--template",
        type=Path,
        default=Path("datasets/venture_corpus_c1/review_template.json"),
    )
    review_gate_parser = subparsers.add_parser("validate-review-gate")
    review_gate_parser.add_argument(
        "--manifest", type=Path, default=Path("datasets/venture_corpus_c1/manifest.json")
    )
    review_gate_parser.add_argument(
        "--packet",
        type=Path,
        default=Path("datasets/venture_corpus_c1/reviewer_packet.json"),
    )
    review_gate_parser.add_argument(
        "--protocol",
        type=Path,
        default=Path("datasets/venture_corpus_c1/review_protocol.yaml"),
    )
    review_gate_parser.add_argument(
        "--reviews-dir", type=Path, default=Path("datasets/venture_corpus_c1/reviews")
    )
    review_gate_parser.add_argument(
        "--status-output",
        type=Path,
        default=Path("datasets/venture_corpus_c1/review_status.json"),
    )
    corpus_smoke_parser = subparsers.add_parser("corpus-smoke")
    corpus_smoke_parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("datasets/venture_corpus_c0/manifest.json"),
    )
    corpus_smoke_parser.add_argument(
        "--config", type=Path, default=Path("configs/venture/venture_m0_20m.yaml")
    )
    corpus_smoke_parser.add_argument("--steps", type=int, default=5)
    corpus_smoke_parser.add_argument("--batch-size", type=int, default=2)
    trial_parser = subparsers.add_parser("venture-trial")
    trial_parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/venture/venture_trial_t0.yaml"),
    )
    trial_parser.add_argument(
        "--output",
        type=Path,
        default=Path("research/trials/CHM-V-T000"),
    )
    trial_parser.add_argument(
        "--checkpoint-dir",
        type=Path,
        default=Path("checkpoints/venture_m0_t0"),
    )
    proposal_diagnostic_parser = subparsers.add_parser("proposal-diagnostic")
    proposal_diagnostic_parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/venture/venture_trial_t2.yaml"),
    )
    proposal_diagnostic_parser.add_argument(
        "--output",
        type=Path,
        default=Path("research/trials/CHM-V-T002/diagnostic.json"),
    )
    proposal_diagnostic_parser.add_argument(
        "--baseline-candidates",
        type=Path,
        default=Path("research/trials/CHM-V-T000/candidates.jsonl"),
    )
    proposal_diagnostic_parser.add_argument(
        "--collapsed-candidates",
        type=Path,
        default=Path("research/trials/CHM-V-T001/candidates.jsonl"),
    )
    proposal_trial_parser = subparsers.add_parser("proposal-trial")
    proposal_trial_parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/venture/venture_trial_t2.yaml"),
    )
    proposal_trial_parser.add_argument(
        "--output",
        type=Path,
        default=Path("research/trials/CHM-V-T002"),
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = build_parser().parse_args(argv)
    if arguments.command == "validate-research":
        records = load_research_registry(arguments.registry)
        print(json.dumps({"validated_hypotheses": len(records)}))
        return 0
    if arguments.command == "build-corpus":
        return _build_corpus(arguments)
    if arguments.command == "validate-corpus":
        return _validate_corpus(arguments)
    if arguments.command == "build-evaluation-corpus":
        return _build_evaluation_corpus(arguments)
    if arguments.command == "validate-evaluation-corpus":
        return _validate_evaluation_corpus(arguments)
    if arguments.command == "build-review-packet":
        return _build_review_packet(arguments)
    if arguments.command == "validate-review-gate":
        return _validate_review_gate(arguments)
    if arguments.command == "venture-trial":
        return _venture_trial(arguments)
    if arguments.command == "proposal-diagnostic":
        return _proposal_diagnostic(arguments)
    if arguments.command == "proposal-trial":
        return _proposal_trial(arguments)
    config = ExperimentConfig.from_yaml(arguments.config)
    if arguments.command == "inspect":
        return _inspect(config)
    if arguments.command == "smoke":
        return _smoke(config, arguments.steps)
    if arguments.command == "corpus-smoke":
        return _corpus_smoke(config, arguments.manifest, arguments.steps, arguments.batch_size)
    raise AssertionError(f"unhandled command: {arguments.command}")


if __name__ == "__main__":
    raise SystemExit(main())
