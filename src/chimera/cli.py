"""Command-line entrypoint for inspection, validation and synthetic smoke runs."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

import torch

from chimera.config import ExperimentConfig
from chimera.data.synthetic import make_synthetic_batch
from chimera.models.venture import ChimeraVenture
from chimera.research import load_research_registry
from chimera.training.trainer import ChimeraTrainer


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
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = build_parser().parse_args(argv)
    if arguments.command == "validate-research":
        records = load_research_registry(arguments.registry)
        print(json.dumps({"validated_hypotheses": len(records)}))
        return 0
    config = ExperimentConfig.from_yaml(arguments.config)
    if arguments.command == "inspect":
        return _inspect(config)
    if arguments.command == "smoke":
        return _smoke(config, arguments.steps)
    raise AssertionError(f"unhandled command: {arguments.command}")


if __name__ == "__main__":
    raise SystemExit(main())
