"""Audit matched mixed/random checkpoint ensembles on opened validation data."""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import cast

import torch
import torch.nn.functional as F
from torch import Tensor

from chimera.meta_world.batch import MetaWorldBatch
from chimera.meta_world.config import MetaWorldModelConfig
from chimera.meta_world.generators import SplitName, WorldGenerationPipeline
from chimera.meta_world.h002 import (
    RelationalSequenceWorldModel,
    evaluate_h002_model,
    materialize_sequence_sample,
)
from chimera.meta_world.h004 import H004DatasetConfig
from chimera.meta_world.h007 import H007RunConfig
from chimera.meta_world.model import MetaWorldOutput


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _parse_checkpoint(value: str) -> tuple[int, Path]:
    seed_text, separator, path_text = value.partition("=")
    if not separator or not seed_text.isdigit() or not path_text:
        raise argparse.ArgumentTypeError("checkpoint must use SEED=PATH")
    return int(seed_text), Path(path_text)


def _mixture(mean_values: list[Tensor], log_variances: list[Tensor]) -> tuple[Tensor, Tensor]:
    means = torch.stack([value.float() for value in mean_values])
    variances = torch.stack(
        [torch.exp(value.float()) for value in log_variances]
    )
    mean = means.mean(dim=0)
    variance = (variances + means.square()).mean(dim=0) - mean.square()
    return mean, variance.clamp_min(1e-8).log()


@dataclass
class EnsemblePredictor:
    """Duck-typed H002 evaluator adapter for one fixed model subset."""

    models: list[RelationalSequenceWorldModel]
    device: torch.device

    @torch.no_grad()
    def predict(self, batch: MetaWorldBatch) -> MetaWorldOutput:
        outputs: list[MetaWorldOutput] = []
        device_batch = batch.to(self.device)
        for model in self.models:
            with torch.autocast(
                device_type=self.device.type,
                dtype=torch.bfloat16,
                enabled=self.device.type == "cuda",
            ):
                outputs.append(cast(MetaWorldOutput, model(device_batch)))
        state_mean, state_log_variance = _mixture(
            [output.next_state_mean for output in outputs],
            [output.next_state_log_variance for output in outputs],
        )
        effect_mean, effect_log_variance = _mixture(
            [output.effect_mean for output in outputs],
            [output.effect_log_variance for output in outputs],
        )
        return MetaWorldOutput(
            next_state_mean=state_mean,
            next_state_log_variance=state_log_variance,
            effect_mean=effect_mean,
            effect_log_variance=effect_log_variance,
            proposal_embedding=F.normalize(
                torch.stack(
                    [output.proposal_embedding.float() for output in outputs]
                ).mean(dim=0),
                dim=-1,
            ),
            final_slot_states=torch.stack(
                [output.final_slot_states.float() for output in outputs]
            ).mean(dim=0),
            transition_state=torch.stack(
                [output.transition_state.float() for output in outputs]
            ).mean(dim=0),
        )


def _load_models(
    checkpoints: dict[int, Path],
    model_config: MetaWorldModelConfig,
    device: torch.device,
) -> tuple[dict[int, RelationalSequenceWorldModel], dict[str, str]]:
    models: dict[int, RelationalSequenceWorldModel] = {}
    hashes: dict[str, str] = {}
    for seed, path in checkpoints.items():
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
        model = RelationalSequenceWorldModel(model_config).to(device)
        model.load_state_dict(checkpoint["model"], strict=True)
        model.eval()
        models[seed] = model
        hashes[str(seed)] = _sha256(path)
    return models, hashes


def diagnose(
    dataset_config_path: Path,
    run_config_path: Path,
    mixed_checkpoints: dict[int, Path],
    random_checkpoints: dict[int, Path],
) -> dict[str, object]:
    if set(mixed_checkpoints) != set(random_checkpoints):
        raise ValueError("mixed and random checkpoint seeds must match")
    required_seeds = {260910, 260911, 260912, 260913, 260914, 260918}
    if set(mixed_checkpoints) != required_seeds:
        raise ValueError("ensemble diagnostic checkpoint seeds differ from registration")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("ensemble diagnostic requires CUDA")
    model_config = H007RunConfig.from_yaml(run_config_path).runtime.model
    mixed_models, mixed_hashes = _load_models(
        mixed_checkpoints,
        model_config,
        device,
    )
    random_models, random_hashes = _load_models(
        random_checkpoints,
        model_config,
        device,
    )
    dataset = H004DatasetConfig.from_yaml(dataset_config_path)
    evaluation_pipeline = WorldGenerationPipeline(
        dataset.worlds,
        dataset.policies()[SplitName.VALIDATION],
    )
    sample = materialize_sequence_sample(
        evaluation_pipeline,
        SplitName.VALIDATION,
        start_index=0,
        batch_size=64,
    )
    groups = {
        "A_260910_260911_260912": [260910, 260911, 260912],
        "B_260911_260912_260913": [260911, 260912, 260913],
        "C_260912_260913_260914": [260912, 260913, 260914],
        "D_260913_260914_260918": [260913, 260914, 260918],
        "ALL_6": sorted(required_seeds),
    }
    results: dict[str, object] = {}
    for group_id, seeds in groups.items():
        mixed = evaluate_h002_model(
            cast(
                object,
                EnsemblePredictor([mixed_models[seed] for seed in seeds], device),
            ),
            sample,
            context_steps=model_config.context_steps,
            rollout_horizon=4,
        ).to_dict()
        random = evaluate_h002_model(
            cast(
                object,
                EnsemblePredictor([random_models[seed] for seed in seeds], device),
            ),
            sample,
            context_steps=model_config.context_steps,
            rollout_horizon=4,
        ).to_dict()
        results[group_id] = {
            "seeds": seeds,
            "mixed": mixed,
            "random": random,
            "ratios": {
                "intervention_effect_nrmse": mixed["intervention_effect_nrmse"]
                / random["intervention_effect_nrmse"],
                "four_step_rollout_nrmse": mixed["four_step_rollout_nrmse"]
                / random["four_step_rollout_nrmse"],
            },
        }
    return {
        "schema_version": 1,
        "diagnostic_id": "CHM-W-H008-ENSEMBLE-001",
        "scope": "already-opened validation ensemble diagnosis",
        "dataset_config": dataset_config_path.as_posix(),
        "dataset_config_sha256": _sha256(dataset_config_path),
        "run_config": run_config_path.as_posix(),
        "run_config_sha256": _sha256(run_config_path),
        "mixed_checkpoint_sha256": mixed_hashes,
        "random_checkpoint_sha256": random_hashes,
        "aggregation": {
            "mean": "arithmetic member mean",
            "variance": "law of total variance Gaussian mixture",
            "member_weights": "uniform",
        },
        "device": str(device),
        "device_name": torch.cuda.get_device_name(device),
        "precision": "bfloat16",
        "groups": results,
        "frozen_validation_seeds_opened": False,
        "test_metrics_opened": False,
        "claim_boundary": (
            "Post-hoc diagnosis on already-opened validation only; no new frozen "
            "validation, test-transfer, causal, business or production claim."
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-config", type=Path, required=True)
    parser.add_argument("--run-config", type=Path, required=True)
    parser.add_argument("--mixed", type=_parse_checkpoint, action="append", required=True)
    parser.add_argument("--random", type=_parse_checkpoint, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    result = diagnose(
        arguments.dataset_config,
        arguments.run_config,
        dict(arguments.mixed),
        dict(arguments.random),
    )
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
