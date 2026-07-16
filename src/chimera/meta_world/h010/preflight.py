"""Validation-only H010 execution with structural mechanism-path auditing."""

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
from chimera.meta_world.h002.config import H002RunConfig
from chimera.meta_world.h002.model import RelationalSequenceWorldModel
from chimera.meta_world.h002.preflight import run_generated_world_preflight
from chimera.meta_world.h002.windows import (
    make_transition_window,
    materialize_sequence_sample,
)
from chimera.meta_world.h010.config import H010ModelVariant, H010RunConfig
from chimera.meta_world.h010.evaluation import projection_prediction_delta
from chimera.meta_world.h010.model import SharedBottleneckRelationalWorldModel


def _model(config: H002RunConfig, variant: H010ModelVariant) -> RelationalSequenceWorldModel:
    if variant is H010ModelVariant.SHARED:
        return SharedBottleneckRelationalWorldModel(config.model)
    return RelationalSequenceWorldModel(config.model)


def run_h010_preflight(
    config_path: str | Path,
    output_dir: str | Path,
) -> dict[str, Any]:
    """Run H010 on train/validation only and persist the structural path audit."""

    config = H010RunConfig.from_yaml(config_path)
    generator_config = GeneratedWorldDatasetConfig.from_yaml(
        config.common.generator_config
    )
    if generator_config.dataset_id != "CHM-W-WG2":
        raise ValueError("H010 requires the registered paired WG2 dataset")
    if generator_config.view_coupling is not ViewCoupling.PAIRED_WORLD_RENDERERS:
        raise ValueError("H010 requires paired renderer views")
    result = run_generated_world_preflight(
        config_path,
        output_dir,
        hypothesis_id="CHM-W-H010",
        run_config=config.common,
        model_factory=lambda common: _model(common, config.model_variant),
    )
    checkpoint = torch.load(
        Path(output_dir) / "checkpoint.pt",
        map_location="cpu",
        weights_only=True,
    )
    model = _model(config.common, config.model_variant)
    model.load_state_dict(checkpoint["model"])
    pipeline = WorldGenerationPipeline(generator_config)
    sample = materialize_sequence_sample(
        pipeline,
        SplitName.VALIDATION,
        start_index=0,
        batch_size=generator_config.views_per_mechanism,
    )
    window = make_transition_window(
        sample,
        prediction_step=2,
        context_steps=config.common.model.context_steps,
    )
    result["model_variant"] = config.model_variant.value
    result["projection_prediction_delta"] = projection_prediction_delta(
        model,
        window,
    )
    Path(output_dir, "result.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return result
