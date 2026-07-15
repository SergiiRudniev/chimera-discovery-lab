"""Measure train-only state/effect gradient conflict on an H006 checkpoint."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
from dataclasses import replace
from pathlib import Path
from statistics import mean, median
from typing import cast

import torch
from torch import Tensor

from chimera.meta_world.config import MetaWorldTrainingConfig
from chimera.meta_world.generators import SplitName, WorldGenerationPipeline
from chimera.meta_world.h002 import (
    RelationalSequenceWorldModel,
    concatenate_sequence_samples,
    make_transition_window,
    materialize_sequence_sample,
)
from chimera.meta_world.h002.windows import GeneratedSequenceSample
from chimera.meta_world.h003 import MechanismMemoryQueue, h003_closed_loop_loss
from chimera.meta_world.h004 import SeededRandomPolicy
from chimera.meta_world.h006 import H006RunConfig
from chimera.meta_world.model import MetaWorldOutput


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _git_commit() -> str | None:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        check=False,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip() or None if result.returncode == 0 else None


def _percentile(values: list[float], quantile: float) -> float:
    if not values or not 0.0 <= quantile <= 1.0:
        raise ValueError("percentile requires values and a valid quantile")
    ordered = sorted(values)
    position = quantile * (len(ordered) - 1)
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def _gradient_geometry(
    state_loss: Tensor,
    effect_loss: Tensor,
    parameters: list[Tensor],
) -> tuple[float, float, float, int]:
    state_gradients = torch.autograd.grad(
        state_loss,
        parameters,
        retain_graph=True,
        allow_unused=True,
    )
    effect_gradients = torch.autograd.grad(
        effect_loss,
        parameters,
        allow_unused=True,
    )
    dot = torch.zeros((), device=state_loss.device, dtype=torch.float64)
    state_square = torch.zeros_like(dot)
    effect_square = torch.zeros_like(dot)
    shared_tensors = 0
    for state_gradient, effect_gradient in zip(
        state_gradients,
        effect_gradients,
        strict=True,
    ):
        if state_gradient is None or effect_gradient is None:
            continue
        state_float = state_gradient.detach().double()
        effect_float = effect_gradient.detach().double()
        dot += (state_float * effect_float).sum()
        state_square += state_float.square().sum()
        effect_square += effect_float.square().sum()
        shared_tensors += 1
    state_norm = state_square.sqrt()
    effect_norm = effect_square.sqrt()
    denominator = state_norm * effect_norm
    if shared_tensors == 0 or float(denominator.cpu()) == 0.0:
        raise RuntimeError("state and effect losses have no non-zero shared gradient")
    return (
        float((dot / denominator).cpu()),
        float(state_norm.cpu()),
        float(effect_norm.cpu()),
        shared_tensors,
    )


def _mixed_sample(
    probe_pipeline: WorldGenerationPipeline,
    random_pipeline: WorldGenerationPipeline,
    *,
    batch_size: int,
    batch_index: int,
) -> GeneratedSequenceSample:
    per_policy = batch_size // 2
    start_index = batch_index * per_policy
    return concatenate_sequence_samples(
        materialize_sequence_sample(
            probe_pipeline,
            SplitName.TRAIN,
            start_index=start_index,
            batch_size=per_policy,
        ),
        materialize_sequence_sample(
            random_pipeline,
            SplitName.TRAIN,
            start_index=start_index,
            batch_size=per_policy,
        ),
    )


def _losses(
    model: RelationalSequenceWorldModel,
    sample: GeneratedSequenceSample,
    *,
    prediction_step: int,
    rollout_horizon: int,
    context_steps: int,
    state_features: int,
    queue: MechanismMemoryQueue,
    training: MetaWorldTrainingConfig,
    device: torch.device,
) -> dict[str, Tensor]:
    generated = sample.batch.to(device)
    device_sample = replace(
        sample,
        batch=generated,
        mechanism_ids=sample.mechanism_ids.to(device),
        mechanism_keys=sample.mechanism_keys.to(device),
        world_family_ids=sample.world_family_ids.to(device),
        renderer_profile_ids=sample.renderer_profile_ids.to(device),
        trajectory_indices=sample.trajectory_indices.to(device),
    )
    frames = list(generated.observations.unbind(dim=1))
    outputs: list[MetaWorldOutput] = []
    windows = []
    for offset in range(rollout_horizon):
        step = prediction_step + offset
        rolling_sample = replace(
            device_sample,
            batch=replace(generated, observations=torch.stack(frames, dim=1)),
        )
        window = make_transition_window(
            rolling_sample,
            prediction_step=step,
            context_steps=context_steps,
        )
        output = cast(MetaWorldOutput, model(window))
        outputs.append(output)
        windows.append(window)
        rolled_state = torch.zeros_like(output.next_state_mean)
        rolled_state[:, :, :state_features] = output.next_state_mean[
            :, :, :state_features
        ]
        frames[step + 1] = torch.where(
            generated.object_mask[:, step + 1].unsqueeze(-1),
            rolled_state,
            frames[step + 1],
        )
    losses, _ = h003_closed_loop_loss(
        outputs,
        windows,
        device_sample.mechanism_keys,
        training,
        queue,
    )
    return losses


def diagnose(
    config_path: Path,
    checkpoint_path: Path,
    *,
    batches: int,
) -> dict[str, object]:
    if batches <= 0:
        raise ValueError("batches must be positive")
    config = H006RunConfig.from_yaml(config_path)
    runtime = config.runtime
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("registered H007 gradient diagnosis requires CUDA")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    model = RelationalSequenceWorldModel(runtime.model).to(device)
    model.load_state_dict(checkpoint["model"], strict=True)
    model.train()
    shared_parameters = [
        parameter
        for name, parameter in model.named_parameters()
        if not name.startswith(("state_head.", "effect_head.", "mechanism_projection."))
    ]
    worlds = runtime.dataset.worlds
    probe_pipeline = WorldGenerationPipeline(
        worlds,
        runtime.dataset.policies()[SplitName.TRAIN],
    )
    random_pipeline = WorldGenerationPipeline(
        worlds,
        SeededRandomPolicy(),
    )
    first_rollout_step = runtime.model.context_steps - 1
    rollout_start_count = (
        worlds.trajectory_steps
        - runtime.closed_loop.rollout_horizon
        - first_rollout_step
    )
    cosines: list[float] = []
    state_norms: list[float] = []
    effect_norms: list[float] = []
    shared_counts: list[int] = []
    queue = MechanismMemoryQueue(minimum_entries=256, maximum_entries=2048)
    torch.manual_seed(runtime.training.seed)
    torch.cuda.manual_seed_all(runtime.training.seed)
    for batch_index in range(batches):
        sample = _mixed_sample(
            probe_pipeline,
            random_pipeline,
            batch_size=runtime.training.batch_size,
            batch_index=batch_index,
        )
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            losses = _losses(
                model,
                sample,
                prediction_step=(
                    first_rollout_step + batch_index % rollout_start_count
                ),
                rollout_horizon=runtime.closed_loop.rollout_horizon,
                context_steps=runtime.model.context_steps,
                state_features=worlds.state_features,
                queue=queue,
                training=runtime.training,
                device=device,
            )
        cosine, state_norm, effect_norm, shared_count = _gradient_geometry(
            runtime.training.next_state_weight * losses["closed_loop_state_loss"],
            runtime.training.effect_weight * losses["closed_loop_effect_loss"],
            shared_parameters,
        )
        cosines.append(cosine)
        state_norms.append(state_norm)
        effect_norms.append(effect_norm)
        shared_counts.append(shared_count)
        model.zero_grad(set_to_none=True)
    return {
        "schema_version": 1,
        "diagnostic_id": "CHM-W-H007-GRADIENT-CONFLICT-001",
        "scope": "train-only post-H006 diagnosis",
        "source_hypothesis": "CHM-W-H006",
        "config": config_path.as_posix(),
        "config_sha256": _sha256(config_path),
        "checkpoint": checkpoint_path.as_posix(),
        "checkpoint_sha256": _sha256(checkpoint_path),
        "weights_kind": checkpoint["weights_kind"],
        "git_commit": _git_commit(),
        "device": str(device),
        "device_name": torch.cuda.get_device_name(device),
        "precision": "bfloat16",
        "batches": batches,
        "batch_size": runtime.training.batch_size,
        "gradient_cosine": {
            "mean": mean(cosines),
            "median": median(cosines),
            "p10": _percentile(cosines, 0.10),
            "p90": _percentile(cosines, 0.90),
            "minimum": min(cosines),
            "maximum": max(cosines),
            "negative_fraction": sum(value < 0.0 for value in cosines) / len(cosines),
        },
        "gradient_norm": {
            "state_median": median(state_norms),
            "effect_median": median(effect_norms),
            "effect_to_state_median_ratio": median(
                [effect / state for effect, state in zip(effect_norms, state_norms, strict=True)]
            ),
        },
        "shared_parameter_tensor_count": {
            "minimum": min(shared_counts),
            "maximum": max(shared_counts),
        },
        "frozen_validation_seeds_opened": False,
        "test_metrics_opened": False,
        "claim_boundary": (
            "Train-only gradient geometry; no validation, test-transfer, causal, "
            "business-utility or production claim."
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batches", type=int, default=32)
    arguments = parser.parse_args()
    result = diagnose(
        arguments.config,
        arguments.checkpoint,
        batches=arguments.batches,
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
