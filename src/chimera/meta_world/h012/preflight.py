"""Validation-only H012 runner with target-family sampling semantics."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from chimera.meta_world.generators import (
    GeneratedWorldDatasetConfig,
    TrainingFamilyPolicy,
    ViewCoupling,
    WorldGenerationPipeline,
)
from chimera.meta_world.h002.config import H002Arm, H002RunConfig
from chimera.meta_world.h002.preflight import run_generated_world_preflight
from chimera.meta_world.h012.baselines import evaluate_legal_random_interventions


def run_h012_preflight(
    config_path: str | Path,
    output_dir: str | Path,
) -> dict[str, Any]:
    """Run one registered H012 arm while keeping every test split sealed."""

    run_config = H002RunConfig.from_yaml(config_path)
    generator = GeneratedWorldDatasetConfig.from_yaml(run_config.generator_config)
    if generator.hypothesis_id != "CHM-W-H012" or generator.dataset_id != "CHM-W-WG3":
        raise ValueError("H012 preflight requires the registered WG3 generator")
    if generator.view_coupling is not ViewCoupling.PAIRED_WORLD_RENDERERS:
        raise ValueError("H012 requires paired renderer views")
    training_policy = (
        TrainingFamilyPolicy.HELD_TARGET
        if run_config.arm is H002Arm.TARGET_FAMILY_ONLY
        else TrainingFamilyPolicy.CROSS_WORLD
    )
    result = run_generated_world_preflight(
        config_path,
        output_dir,
        hypothesis_id="CHM-W-H012",
        run_config=run_config,
        training_pipeline_factory=lambda config: WorldGenerationPipeline(
            config,
            training_family_policy=training_policy,
        ),
        allow_target_family_only=True,
    )
    result["training_family_policy"] = training_policy.value
    result["validation_random_intervention"] = evaluate_legal_random_interventions(
        generator
    ).to_dict()
    result["scientific_result"] = False
    Path(output_dir, "result.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return result
