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
from chimera.data.synthetic import make_synthetic_batch
from chimera.meta_world.config import MetaWorldExperimentConfig
from chimera.meta_world.generators import (
    GeneratedWorldDatasetConfig,
    SplitName,
    WorldGenerationPipeline,
    build_generated_world_dataset,
    validate_generated_world_dataset,
)
from chimera.meta_world.h002 import run_h002_preflight
from chimera.meta_world.h003 import run_h003_preflight
from chimera.meta_world.h004 import (
    build_h004_probe_dataset,
    run_h004_preflight,
    validate_h004_probe_dataset,
)
from chimera.meta_world.h005 import run_h005_preflight, run_h005_validation
from chimera.meta_world.h006 import run_h006_preflight
from chimera.meta_world.h007 import run_h007_preflight
from chimera.meta_world.h008 import run_h008_development_suite, run_h008_preflight
from chimera.meta_world.h009 import run_h009_preflight
from chimera.meta_world.h010 import run_h010_preflight
from chimera.meta_world.h011 import run_h011_preflight
from chimera.meta_world.h012 import build_h012_smoke_dataset, run_h012_preflight
from chimera.meta_world.h013 import run_h013_development_suite, run_h013_preflight
from chimera.meta_world.h013.dataset import build_h013_smoke_dataset
from chimera.meta_world.model import ChimeraMetaWorld
from chimera.meta_world.trial import run_meta_world_trial
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


def _meta_world_inspect(config: MetaWorldExperimentConfig) -> int:
    model = ChimeraMetaWorld(config.model)
    payload = {
        "experiment_id": config.experiment_id,
        "trial_id": config.trial_id,
        "model": "Chimera Meta-World W0",
        "trainable_parameters": model.trainable_parameter_count(),
        "hidden_dim": config.model.hidden_dim,
        "max_slots": config.model.max_slots,
        "context_steps": config.model.context_steps,
        "spatial_layers": config.model.spatial_layers,
        "temporal_layers": config.model.temporal_layers,
        "transition_layers": config.model.transition_layers,
        "language_inputs": False,
    }
    print(json.dumps(payload, indent=2))
    return 0


def _meta_world_trial(arguments: argparse.Namespace) -> int:
    result = run_meta_world_trial(arguments.config, arguments.output, arguments.result)
    print(
        json.dumps(
            {
                "id": result["id"],
                "trial_id": result["trial_id"],
                "decision": result["decision"],
                "parameters": result["parameters"],
                "output": str(arguments.output),
            },
            sort_keys=True,
        )
    )
    return 0


def _build_generated_world_dataset(arguments: argparse.Namespace) -> int:
    manifest = build_generated_world_dataset(
        arguments.output,
        arguments.config,
        trajectories_per_split=arguments.trajectories_per_split,
    )
    counts = manifest["counts"]
    if not isinstance(counts, dict):
        raise TypeError("generated-world manifest counts must be a mapping")
    print(
        json.dumps(
            {
                "dataset_id": manifest["dataset_id"],
                "manifest": str(arguments.output / "manifest.json"),
                **counts,
            },
            sort_keys=True,
        )
    )
    return 0


def _validate_generated_world_dataset(arguments: argparse.Namespace) -> int:
    report = validate_generated_world_dataset(arguments.manifest)
    counts = report["counts"]
    if not isinstance(counts, dict):
        raise TypeError("generated-world report counts must be a mapping")
    print(
        json.dumps(
            {
                "dataset_id": report["dataset_id"],
                "status": report["status"],
                **counts,
            },
            sort_keys=True,
        )
    )
    return 0 if report["status"] == "passed" else 1


def _build_h004_probe_dataset(arguments: argparse.Namespace) -> int:
    manifest = build_h004_probe_dataset(
        arguments.output,
        arguments.config,
        trajectories_per_split=arguments.trajectories_per_split,
    )
    counts = manifest["counts"]
    if not isinstance(counts, dict):
        raise TypeError("WG1 manifest counts must be a mapping")
    print(
        json.dumps(
            {
                "dataset_id": manifest["dataset_id"],
                "manifest": str(arguments.output / "manifest.json"),
                **counts,
            },
            sort_keys=True,
        )
    )
    return 0


def _validate_h004_probe_dataset(arguments: argparse.Namespace) -> int:
    report = validate_h004_probe_dataset(arguments.manifest)
    counts = report["counts"]
    if not isinstance(counts, dict):
        raise TypeError("WG1 validation counts must be a mapping")
    print(
        json.dumps(
            {
                "dataset_id": report["dataset_id"],
                "status": report["status"],
                "probe_response_separation": report["probe_response_separation"],
                **counts,
            },
            sort_keys=True,
        )
    )
    return 0 if report["status"] == "passed" else 1


def _world_generator_smoke(arguments: argparse.Namespace) -> int:
    config = GeneratedWorldDatasetConfig.from_yaml(arguments.config)
    batch = WorldGenerationPipeline(config).online_batch(
        SplitName(arguments.split),
        arguments.batch_size,
        start_index=arguments.start_index,
    )
    payload = {
        "dataset_id": config.dataset_id,
        "split": arguments.split,
        "batch_size": batch.batch_size,
        "observations": list(batch.observations.shape),
        "relations": list(batch.relations.shape),
        "actions": list(batch.actions.shape),
        "outcomes": list(batch.outcomes.shape),
        "all_finite": bool(
            torch.isfinite(batch.observations).all()
            and torch.isfinite(batch.relations).all()
            and torch.isfinite(batch.outcomes).all()
        ),
        "language_inputs": False,
    }
    print(json.dumps(payload, sort_keys=True))
    return 0


def _h009_generator_smoke(arguments: argparse.Namespace) -> int:
    manifest = build_generated_world_dataset(
        arguments.output,
        arguments.config,
        trajectories_per_split=arguments.trajectories_per_split,
        claim_boundary=(
            "H009 generated-world engineering smoke only; no transfer, causal, "
            "business-utility or production claim."
        ),
    )
    report = validate_generated_world_dataset(arguments.output / "manifest.json")
    print(
        json.dumps(
            {
                "dataset_id": manifest["dataset_id"],
                "hypothesis_id": report["hypothesis_id"],
                "status": report["status"],
                "manifest": str(arguments.output / "manifest.json"),
                "counts": report["counts"],
                "checks": report["checks"],
                "scientific_result": False,
            },
            sort_keys=True,
        )
    )
    return 0 if report["status"] == "passed" else 1


def _h002_preflight(arguments: argparse.Namespace) -> int:
    result = run_h002_preflight(arguments.config, arguments.output)
    print(
        json.dumps(
            {
                "run_id": result["run_id"],
                "status": result["status"],
                "arm": result["arm"],
                "parameters": result["parameters"],
                "best_step": result["best_step"],
                "output": str(arguments.output),
                "test_metrics_opened": result["test_metrics_opened"],
            },
            sort_keys=True,
        )
    )
    return 0


def _h009_preflight(arguments: argparse.Namespace) -> int:
    result = run_h009_preflight(arguments.config, arguments.output)
    print(
        json.dumps(
            {
                "run_id": result["run_id"],
                "hypothesis_id": result["hypothesis_id"],
                "status": result["status"],
                "arm": result["arm"],
                "parameters": result["parameters"],
                "best_step": result["best_step"],
                "output": str(arguments.output),
                "test_metrics_opened": result["test_metrics_opened"],
            },
            sort_keys=True,
        )
    )
    return 0


def _h010_preflight(arguments: argparse.Namespace) -> int:
    result = run_h010_preflight(arguments.config, arguments.output)
    print(
        json.dumps(
            {
                "run_id": result["run_id"],
                "hypothesis_id": result["hypothesis_id"],
                "status": result["status"],
                "arm": result["arm"],
                "model_variant": result["model_variant"],
                "projection_prediction_delta": result[
                    "projection_prediction_delta"
                ],
                "parameters": result["parameters"],
                "best_step": result["best_step"],
                "output": str(arguments.output),
                "test_metrics_opened": result["test_metrics_opened"],
            },
            sort_keys=True,
        )
    )
    return 0


def _h011_preflight(arguments: argparse.Namespace) -> int:
    result = run_h011_preflight(arguments.config, arguments.output)
    print(
        json.dumps(
            {
                "run_id": result["run_id"],
                "hypothesis_id": result["hypothesis_id"],
                "status": result["status"],
                "arm": result["arm"],
                "response_consistency_weight": result[
                    "response_consistency_weight"
                ],
                "paired_effect_mean_disagreement": result[
                    "paired_effect_mean_disagreement"
                ],
                "parameters": result["parameters"],
                "best_step": result["best_step"],
                "output": str(arguments.output),
                "test_metrics_opened": result["test_metrics_opened"],
            },
            sort_keys=True,
        )
    )
    return 0


def _h012_generator_smoke(arguments: argparse.Namespace) -> int:
    report = build_h012_smoke_dataset(
        arguments.output,
        arguments.config,
        trajectories_per_split=arguments.trajectories_per_split,
    )
    print(
        json.dumps(
            {
                "dataset_id": report["dataset_id"],
                "hypothesis_id": report["hypothesis_id"],
                "status": report["status"],
                "manifest": str(arguments.output / "manifest.json"),
                "counts": report["counts"],
                "checks": report["checks"],
                "scientific_result": False,
            },
            sort_keys=True,
        )
    )
    return 0


def _h012_preflight(arguments: argparse.Namespace) -> int:
    result = run_h012_preflight(arguments.config, arguments.output)
    print(
        json.dumps(
            {
                "run_id": result["run_id"],
                "hypothesis_id": result["hypothesis_id"],
                "status": result["status"],
                "arm": result["arm"],
                "training_family_policy": result["training_family_policy"],
                "parameters": result["parameters"],
                "best_step": result["best_step"],
                "output": str(arguments.output),
                "test_metrics_opened": result["test_metrics_opened"],
                "scientific_result": result["scientific_result"],
            },
            sort_keys=True,
        )
    )
    return 0


def _h013_generator_smoke(arguments: argparse.Namespace) -> int:
    report = build_h013_smoke_dataset(
        arguments.config,
        arguments.output,
        arguments.report,
        trajectories_per_split=arguments.trajectories_per_split,
    )
    print(
        json.dumps(
            {
                "dataset_id": report["dataset_id"],
                "hypothesis_id": report["hypothesis_id"],
                "status": report["status"],
                "report": str(arguments.report),
                "counts": report["counts"],
                "checks": report["checks"],
                "test_metrics_opened": False,
            },
            sort_keys=True,
        )
    )
    return 0


def _h013_preflight(arguments: argparse.Namespace) -> int:
    result = run_h013_preflight(arguments.config, arguments.output)
    print(
        json.dumps(
            {
                "run_id": result["run_id"],
                "status": result["status"],
                "arm": result["arm"],
                "transition_semantics": result["transition_semantics"],
                "parameters": result["parameters"],
                "best_step": result["best_step"],
                "output": str(arguments.output),
                "test_metrics_opened": result["test_metrics_opened"],
            },
            sort_keys=True,
        )
    )
    return 0


def _h013_suite(arguments: argparse.Namespace) -> int:
    result = run_h013_development_suite(
        arguments.config,
        arguments.output,
        arguments.report,
    )
    print(
        json.dumps(
            {
                "preflight_id": result["preflight_id"],
                "status": result["status"],
                "decision": result["decision"],
                "passed": result["development_gate"]["passed"],
                "report": str(arguments.report),
                "test_metrics_opened": result["test_metrics_opened"],
            },
            sort_keys=True,
        )
    )
    return 0


def _h003_preflight(arguments: argparse.Namespace) -> int:
    result = run_h003_preflight(arguments.config, arguments.output)
    print(
        json.dumps(
            {
                "run_id": result["run_id"],
                "status": result["status"],
                "arm": result["arm"],
                "parameters": result["parameters"],
                "best_step": result["best_step"],
                "output": str(arguments.output),
                "test_metrics_opened": result["test_metrics_opened"],
            },
            sort_keys=True,
        )
    )
    return 0


def _h004_preflight(arguments: argparse.Namespace) -> int:
    result = run_h004_preflight(arguments.config, arguments.output)
    print(
        json.dumps(
            {
                "run_id": result["run_id"],
                "status": result["status"],
                "arm": result["arm"],
                "train_action_policy": result["train_action_policy"],
                "parameters": result["parameters"],
                "best_step": result["best_step"],
                "output": str(arguments.output),
                "test_metrics_opened": result["test_metrics_opened"],
            },
            sort_keys=True,
        )
    )
    return 0


def _h005_preflight(arguments: argparse.Namespace) -> int:
    result = run_h005_preflight(arguments.config, arguments.output)
    print(
        json.dumps(
            {
                "run_id": result["run_id"],
                "status": result["status"],
                "arm": result["arm"],
                "train_action_policy": result["train_action_policy"],
                "parameters": result["parameters"],
                "best_step": result["best_step"],
                "output": str(arguments.output),
                "test_metrics_opened": result["test_metrics_opened"],
            },
            sort_keys=True,
        )
    )
    return 0


def _h005_validation(arguments: argparse.Namespace) -> int:
    result = run_h005_validation(arguments.config, arguments.output)
    print(
        json.dumps(
            {
                "run_id": result["run_id"],
                "status": result["status"],
                "arm": result["arm"],
                "seed": result["seed"],
                "parameters": result["parameters"],
                "selected_step": result["best_step"],
                "output": str(arguments.output),
                "test_metrics_opened": result["test_metrics_opened"],
            },
            sort_keys=True,
        )
    )
    return 0


def _h006_preflight(arguments: argparse.Namespace) -> int:
    result = run_h006_preflight(arguments.config, arguments.output)
    print(
        json.dumps(
            {
                "run_id": result["run_id"],
                "status": result["status"],
                "arm": result["arm"],
                "objective_routing": result["objective_routing"],
                "parameters": result["parameters"],
                "best_step": result["best_step"],
                "output": str(arguments.output),
                "test_metrics_opened": result["test_metrics_opened"],
            },
            sort_keys=True,
        )
    )
    return 0


def _h007_preflight(arguments: argparse.Namespace) -> int:
    result = run_h007_preflight(arguments.config, arguments.output)
    print(
        json.dumps(
            {
                "run_id": result["run_id"],
                "status": result["status"],
                "arm": result["arm"],
                "gradient_intervention": result["gradient_intervention"],
                "parameters": result["parameters"],
                "best_step": result["best_step"],
                "output": str(arguments.output),
                "test_metrics_opened": result["test_metrics_opened"],
            },
            sort_keys=True,
        )
    )
    return 0


def _h008_preflight(arguments: argparse.Namespace) -> int:
    result = run_h008_preflight(arguments.config, arguments.output)
    print(
        json.dumps(
            {
                "run_id": result["run_id"],
                "status": result["status"],
                "arm": result["arm"],
                "outcome_head": result["outcome_head"],
                "parameters": result["parameters"],
                "best_step": result["best_step"],
                "counterfactual_audit": result["counterfactual_audit"],
                "output": str(arguments.output),
                "test_metrics_opened": result["test_metrics_opened"],
            },
            sort_keys=True,
        )
    )
    return 0


def _h008_suite(arguments: argparse.Namespace) -> int:
    result = run_h008_development_suite(
        arguments.config,
        arguments.output,
        arguments.report,
    )
    print(
        json.dumps(
            {
                "preflight_id": result["preflight_id"],
                "status": result["status"],
                "decision": result["decision"],
                "passed": result["development_gate"]["passed"],
                "report": str(arguments.report),
                "test_metrics_opened": result["test_metrics_opened"],
            },
            sort_keys=True,
        )
    )
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
    meta_world_inspect_parser = subparsers.add_parser("meta-world-inspect")
    meta_world_inspect_parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/meta_world/meta_world_w0.yaml"),
    )
    meta_world_trial_parser = subparsers.add_parser("meta-world-trial")
    meta_world_trial_parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/meta_world/meta_world_w0.yaml"),
    )
    meta_world_trial_parser.add_argument(
        "--output",
        type=Path,
        default=Path("research/trials/CHM-W-T000"),
    )
    meta_world_trial_parser.add_argument(
        "--result",
        type=Path,
        default=Path("research/results/CHM-W-H000.json"),
    )
    world_generator_build_parser = subparsers.add_parser(
        "build-world-generator-dataset"
    )
    world_generator_build_parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/meta_world/world_generators_h002.yaml"),
    )
    world_generator_build_parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/meta_world_generator_smoke"),
    )
    world_generator_build_parser.add_argument(
        "--trajectories-per-split",
        type=int,
        default=12,
    )
    world_generator_validation_parser = subparsers.add_parser(
        "validate-world-generator-dataset"
    )
    world_generator_validation_parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("artifacts/meta_world_generator_smoke/manifest.json"),
    )
    world_generator_smoke_parser = subparsers.add_parser("world-generator-smoke")
    world_generator_smoke_parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/meta_world/world_generators_h002.yaml"),
    )
    world_generator_smoke_parser.add_argument(
        "--split",
        choices=[item.value for item in SplitName],
        default=SplitName.TRAIN.value,
    )
    world_generator_smoke_parser.add_argument("--batch-size", type=int, default=4)
    world_generator_smoke_parser.add_argument("--start-index", type=int, default=0)
    h009_generator_smoke_parser = subparsers.add_parser(
        "meta-world-h009-smoke-dataset"
    )
    h009_generator_smoke_parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/meta_world/world_generators_h009.yaml"),
    )
    h009_generator_smoke_parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/meta_world_h009_smoke"),
    )
    h009_generator_smoke_parser.add_argument(
        "--trajectories-per-split",
        type=int,
        default=16,
    )
    h004_dataset_parser = subparsers.add_parser("build-world-probe-dataset")
    h004_dataset_parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/meta_world/world_generators_h004.yaml"),
    )
    h004_dataset_parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/meta_world_probe_smoke"),
    )
    h004_dataset_parser.add_argument(
        "--trajectories-per-split",
        type=int,
        default=16,
    )
    h004_validation_parser = subparsers.add_parser("validate-world-probe-dataset")
    h004_validation_parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("artifacts/meta_world_probe_smoke/manifest.json"),
    )
    h002_preflight_parser = subparsers.add_parser("meta-world-h002-preflight")
    h002_preflight_parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/meta_world/world_h002_preflight.yaml"),
    )
    h002_preflight_parser.add_argument(
        "--output",
        type=Path,
        default=Path("runs/h002_preflight_aligned_a"),
    )
    h003_preflight_parser = subparsers.add_parser("meta-world-h003-preflight")
    h003_preflight_parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/meta_world/world_h003_preflight_closed_loop.yaml"),
    )
    h003_preflight_parser.add_argument(
        "--output",
        type=Path,
        default=Path("runs/h003_preflight_closed_loop"),
    )
    h004_preflight_parser = subparsers.add_parser("meta-world-h004-preflight")
    h004_preflight_parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/meta_world/world_h004_preflight_probe.yaml"),
    )
    h004_preflight_parser.add_argument(
        "--output",
        type=Path,
        default=Path("runs/h004_preflight_probe"),
    )
    h005_preflight_parser = subparsers.add_parser("meta-world-h005-preflight")
    h005_preflight_parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/meta_world/world_h005_development_mixed.yaml"),
    )
    h005_preflight_parser.add_argument(
        "--output",
        type=Path,
        default=Path("runs/h005_development_mixed"),
    )
    h005_validation_parser = subparsers.add_parser("meta-world-h005-validate")
    h005_validation_parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/meta_world/world_h005_validation_mixed_s260911.yaml"),
    )
    h005_validation_parser.add_argument(
        "--output",
        type=Path,
        default=Path("runs/h005_validation_mixed_s260911"),
    )
    h006_preflight_parser = subparsers.add_parser("meta-world-h006-preflight")
    h006_preflight_parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/meta_world/world_h006_development_routed.yaml"),
    )
    h006_preflight_parser.add_argument(
        "--output",
        type=Path,
        default=Path("runs/h006_development_routed"),
    )
    h007_preflight_parser = subparsers.add_parser("meta-world-h007-preflight")
    h007_preflight_parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/meta_world/world_h007_development_pcgrad.yaml"),
    )
    h007_preflight_parser.add_argument(
        "--output",
        type=Path,
        default=Path("runs/h007_development_pcgrad"),
    )
    h008_preflight_parser = subparsers.add_parser("meta-world-h008-preflight")
    h008_preflight_parser.add_argument(
        "--config",
        type=Path,
        default=Path(
            "configs/meta_world/world_h008_development_counterfactual_mixed.yaml"
        ),
    )
    h008_preflight_parser.add_argument(
        "--output",
        type=Path,
        default=Path("runs/h008_development_counterfactual_mixed"),
    )
    h008_suite_parser = subparsers.add_parser("meta-world-h008-suite")
    h008_suite_parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/meta_world/world_h008_development_suite.yaml"),
    )
    h008_suite_parser.add_argument(
        "--output",
        type=Path,
        default=Path("runs/h008_development"),
    )
    h008_suite_parser.add_argument(
        "--report",
        type=Path,
        default=Path("research/preflights/CHM-W-H008-development.json"),
    )
    h009_preflight_parser = subparsers.add_parser("meta-world-h009-preflight")
    h009_preflight_parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/meta_world/world_h009_development_aligned.yaml"),
    )
    h009_preflight_parser.add_argument(
        "--output",
        type=Path,
        default=Path("runs/h009_development_aligned"),
    )
    h010_preflight_parser = subparsers.add_parser("meta-world-h010-preflight")
    h010_preflight_parser.add_argument(
        "--config",
        type=Path,
        default=Path(
            "configs/meta_world/world_h010_development_shared_aligned.yaml"
        ),
    )
    h010_preflight_parser.add_argument(
        "--output",
        type=Path,
        default=Path("runs/h010_development_shared_aligned"),
    )
    h011_preflight_parser = subparsers.add_parser("meta-world-h011-preflight")
    h011_preflight_parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/meta_world/world_h011_development_smoke.yaml"),
    )
    h011_preflight_parser.add_argument(
        "--output",
        type=Path,
        default=Path("runs/h011_development_smoke"),
    )
    h012_generator_smoke_parser = subparsers.add_parser(
        "meta-world-h012-smoke-dataset"
    )
    h012_generator_smoke_parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/meta_world/world_generators_h012.yaml"),
    )
    h012_generator_smoke_parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/meta_world_h012_smoke"),
    )
    h012_generator_smoke_parser.add_argument(
        "--trajectories-per-split",
        type=int,
        default=16,
    )
    h012_preflight_parser = subparsers.add_parser("meta-world-h012-preflight")
    h012_preflight_parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/meta_world/world_h012_development_smoke.yaml"),
    )
    h012_preflight_parser.add_argument(
        "--output",
        type=Path,
        default=Path("runs/h012_development_smoke"),
    )
    h013_generator_smoke_parser = subparsers.add_parser(
        "meta-world-h013-smoke-dataset"
    )
    h013_generator_smoke_parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/meta_world/world_generators_h013.yaml"),
    )
    h013_generator_smoke_parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/meta_world_h013_smoke"),
    )
    h013_generator_smoke_parser.add_argument(
        "--report",
        type=Path,
        default=Path("research/preflights/CHM-W-H013-WG4-integrity.json"),
    )
    h013_generator_smoke_parser.add_argument(
        "--trajectories-per-split",
        type=int,
        default=16,
    )
    h013_preflight_parser = subparsers.add_parser("meta-world-h013-preflight")
    h013_preflight_parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/meta_world/world_h013_development_factorized.yaml"),
    )
    h013_preflight_parser.add_argument(
        "--output",
        type=Path,
        default=Path("runs/h013_development_factorized"),
    )
    h013_suite_parser = subparsers.add_parser("meta-world-h013-suite")
    h013_suite_parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/meta_world/world_h013_suite.yaml"),
    )
    h013_suite_parser.add_argument(
        "--output",
        type=Path,
        default=Path("runs/h013_development"),
    )
    h013_suite_parser.add_argument(
        "--report",
        type=Path,
        default=Path("research/preflights/CHM-W-H013-development.json"),
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
    if arguments.command == "venture-trial":
        return _venture_trial(arguments)
    if arguments.command == "proposal-diagnostic":
        return _proposal_diagnostic(arguments)
    if arguments.command == "proposal-trial":
        return _proposal_trial(arguments)
    if arguments.command == "meta-world-inspect":
        return _meta_world_inspect(MetaWorldExperimentConfig.from_yaml(arguments.config))
    if arguments.command == "meta-world-trial":
        return _meta_world_trial(arguments)
    if arguments.command == "build-world-generator-dataset":
        return _build_generated_world_dataset(arguments)
    if arguments.command == "validate-world-generator-dataset":
        return _validate_generated_world_dataset(arguments)
    if arguments.command == "world-generator-smoke":
        return _world_generator_smoke(arguments)
    if arguments.command == "meta-world-h009-smoke-dataset":
        return _h009_generator_smoke(arguments)
    if arguments.command == "build-world-probe-dataset":
        return _build_h004_probe_dataset(arguments)
    if arguments.command == "validate-world-probe-dataset":
        return _validate_h004_probe_dataset(arguments)
    if arguments.command == "meta-world-h002-preflight":
        return _h002_preflight(arguments)
    if arguments.command == "meta-world-h003-preflight":
        return _h003_preflight(arguments)
    if arguments.command == "meta-world-h004-preflight":
        return _h004_preflight(arguments)
    if arguments.command == "meta-world-h005-preflight":
        return _h005_preflight(arguments)
    if arguments.command == "meta-world-h005-validate":
        return _h005_validation(arguments)
    if arguments.command == "meta-world-h006-preflight":
        return _h006_preflight(arguments)
    if arguments.command == "meta-world-h007-preflight":
        return _h007_preflight(arguments)
    if arguments.command == "meta-world-h008-preflight":
        return _h008_preflight(arguments)
    if arguments.command == "meta-world-h008-suite":
        return _h008_suite(arguments)
    if arguments.command == "meta-world-h009-preflight":
        return _h009_preflight(arguments)
    if arguments.command == "meta-world-h010-preflight":
        return _h010_preflight(arguments)
    if arguments.command == "meta-world-h011-preflight":
        return _h011_preflight(arguments)
    if arguments.command == "meta-world-h012-smoke-dataset":
        return _h012_generator_smoke(arguments)
    if arguments.command == "meta-world-h012-preflight":
        return _h012_preflight(arguments)
    if arguments.command == "meta-world-h013-smoke-dataset":
        return _h013_generator_smoke(arguments)
    if arguments.command == "meta-world-h013-preflight":
        return _h013_preflight(arguments)
    if arguments.command == "meta-world-h013-suite":
        return _h013_suite(arguments)
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
