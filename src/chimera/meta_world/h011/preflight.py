"""Validation-only H011 response-consistency preflight."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch

from chimera.meta_world.generators import (
    GeneratedWorldDatasetConfig,
    SplitName,
    ViewCoupling,
    WorldGenerationPipeline,
)
from chimera.meta_world.h002.model import RelationalSequenceWorldModel
from chimera.meta_world.h002.preflight import run_generated_world_preflight
from chimera.meta_world.h002.windows import materialize_sequence_sample
from chimera.meta_world.h011.config import H011RunConfig
from chimera.meta_world.h011.evaluation import evaluate_paired_response_disagreement
from chimera.meta_world.h011.trainer import H011Trainer
from chimera.meta_world.h011.windows import make_paired_response_window


def run_h011_preflight(
    config_path: str | Path,
    output_dir: str | Path,
) -> dict[str, Any]:
    config = H011RunConfig.from_yaml(config_path)
    generator_config = GeneratedWorldDatasetConfig.from_yaml(
        config.common.generator_config
    )
    if generator_config.dataset_id != "CHM-W-WG2":
        raise ValueError("H011 requires the registered paired WG2 dataset")
    if generator_config.view_coupling is not ViewCoupling.PAIRED_WORLD_RENDERERS:
        raise ValueError("H011 requires paired renderer views")
    result = run_generated_world_preflight(
        config_path,
        output_dir,
        hypothesis_id="CHM-W-H011",
        run_config=config.common,
        model_factory=lambda common: RelationalSequenceWorldModel(common.model),
        trainer_factory=lambda model, training: H011Trainer(
            model,
            training,
            response_consistency_weight=config.response_consistency_weight,
            uncertainty_consistency_fraction=(
                config.uncertainty_consistency_fraction
            ),
        ),
        window_factory=make_paired_response_window,
    )
    checkpoint = torch.load(
        Path(output_dir) / "checkpoint.pt", map_location="cpu", weights_only=True
    )
    model = RelationalSequenceWorldModel(config.common.model)
    model.load_state_dict(checkpoint["model"])
    sample = materialize_sequence_sample(
        WorldGenerationPipeline(generator_config),
        SplitName.VALIDATION,
        start_index=0,
        batch_size=config.common.evaluation.validation_trajectories,
    )
    result["response_consistency_weight"] = config.response_consistency_weight
    result.update(
        evaluate_paired_response_disagreement(
            model,
            sample,
            context_steps=config.common.model.context_steps,
        )
    )
    Path(output_dir, "result.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return result
