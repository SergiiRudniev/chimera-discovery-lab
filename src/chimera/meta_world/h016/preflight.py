"""Backbone runner for CHM-W-H016."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from chimera.meta_world.h013.preflight import execute_paired_transition_preflight
from chimera.meta_world.h014.model import (
    ResponseConditionedEffectWorldModel,
    ResponseSource,
)
from chimera.meta_world.h016.config import H016BackboneConfig


def run_h016_backbone_preflight(
    config_path: str | Path,
    output_dir: str | Path,
) -> dict[str, Any]:
    """Retrain the frozen H015 control architecture under the H016 seed."""

    config = H016BackboneConfig.from_yaml(config_path)
    return execute_paired_transition_preflight(
        config_path,
        output_dir,
        run_config=config.paired_runtime,
        hypothesis_id="CHM-W-H016",
        reported_arm="factual_residual_conditioned_ranking_backbone",
        model_factory=lambda runtime: ResponseConditionedEffectWorldModel(
            runtime.runtime.model,
            response_source=ResponseSource.FACTUAL_RESIDUAL,
        ),
        selection_metrics=("intervention_effect_nrmse",),
        result_metadata={
            "response_source": config.response_source,
            "ranking_targets_opened": False,
            "candidate_metrics_opened": False,
        },
    )
