"""Train the fixed H015 search backbone on WG4 development data."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from chimera.meta_world.h013.config import H013RunConfig
from chimera.meta_world.h013.preflight import execute_paired_transition_preflight
from chimera.meta_world.h014.model import (
    ResponseConditionedEffectWorldModel,
    ResponseSource,
)
from chimera.meta_world.h015.config import H015BackboneConfig


def run_h015_backbone_preflight(
    config_path: str | Path,
    output_dir: str | Path,
) -> dict[str, Any]:
    """Train the factual-residual control before any candidate evaluation."""

    config = H015BackboneConfig.from_yaml(config_path)

    def model_factory(runtime: H013RunConfig) -> ResponseConditionedEffectWorldModel:
        return ResponseConditionedEffectWorldModel(
            runtime.runtime.model,
            response_source=ResponseSource.FACTUAL_RESIDUAL,
        )

    return execute_paired_transition_preflight(
        config_path,
        output_dir,
        run_config=config.paired_runtime,
        hypothesis_id="CHM-W-H015",
        reported_arm="factual_residual_conditioned_search_backbone",
        model_factory=model_factory,
        selection_metrics=("intervention_effect_nrmse",),
        result_metadata={
            "response_source": config.response_source,
            "candidate_metrics_opened": False,
        },
    )
