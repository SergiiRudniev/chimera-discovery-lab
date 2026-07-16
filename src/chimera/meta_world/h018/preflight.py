"""Validation-only H018 training without compositional test access."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from chimera.meta_world.generators import (
    GeneratedWorldDatasetConfig,
    TrainingFamilyPolicy,
)
from chimera.meta_world.h002.config import H002Arm, H002RunConfig
from chimera.meta_world.h002.preflight import run_generated_world_preflight
from chimera.meta_world.h018.baselines import evaluate_h018_random_interventions
from chimera.meta_world.h018.dataset import make_h018_pipeline


def run_h018_preflight(
    config_path: str | Path,
    output_dir: str | Path,
) -> dict[str, Any]:
    """Train one frozen arm on train and select only on validation."""

    run = H002RunConfig.from_yaml(config_path)
    generator = GeneratedWorldDatasetConfig.from_yaml(run.generator_config)
    training_policy = (
        TrainingFamilyPolicy.HELD_TARGET
        if run.arm is H002Arm.TARGET_FAMILY_ONLY
        else TrainingFamilyPolicy.CROSS_WORLD
    )
    result = run_generated_world_preflight(
        config_path,
        output_dir,
        hypothesis_id="CHM-W-H018",
        run_config=run,
        training_pipeline_factory=lambda config: make_h018_pipeline(
            config, training_family_policy=training_policy
        ),
        validation_pipeline_factory=make_h018_pipeline,
        allow_target_family_only=True,
    )
    result["training_family_policy"] = training_policy.value
    result["validation_random_intervention"] = (
        evaluate_h018_random_interventions(generator).to_dict()
    )
    result["mechanism_program_metadata_passed_to_model"] = False
    result["scientific_result"] = False
    Path(output_dir, "result.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return result
