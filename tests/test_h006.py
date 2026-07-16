from __future__ import annotations

from pathlib import Path

import yaml

from chimera.meta_world.config import MetaWorldModelConfig, MetaWorldTrainingConfig
from chimera.meta_world.h006 import H006Arm, H006RunConfig, run_h006_preflight


def _model_config() -> MetaWorldModelConfig:
    return MetaWorldModelConfig(
        observation_features=8,
        relation_features=4,
        intervention_types=1,
        intervention_parameters=3,
        effect_dimensions=4,
        domain_count=1,
        mechanism_count=2,
        hidden_dim=32,
        num_heads=4,
        spatial_layers=1,
        temporal_layers=1,
        transition_layers=1,
        feedforward_multiplier=2,
        max_slots=10,
        context_steps=4,
        dropout=0.0,
    )


def _training_config() -> MetaWorldTrainingConfig:
    return MetaWorldTrainingConfig(
        seed=260914,
        batch_size=4,
        active_slots=4,
        steps=2,
        learning_rate=1e-3,
        weight_decay=0.0,
        next_state_weight=1.0,
        effect_weight=0.5,
        alignment_weight=0.0,
        variance_weight=0.01,
        alignment_margin=0.2,
        primary_effect_weight=2.0,
        ema_decay=0.9,
        device="cpu",
        precision="float32",
    )


def _payload() -> dict[str, object]:
    return {
        "run_id": "H006-DEVELOPMENT-TEST",
        "mode": "preflight",
        "arm": "routed_mixed_closed_loop_without_discrimination",
        "dataset_config": "configs/meta_world/world_generators_h004.yaml",
        "model": _model_config().__dict__,
        "training": _training_config().__dict__,
        "curriculum": {"probe_fraction": 0.5},
        "closed_loop": {
            "rollout_horizon": 4,
            "queue_minimum_entries": 4,
            "queue_maximum_entries": 8,
        },
        "evaluation": {
            "evaluation_interval": 1,
            "validation_trajectories": 4,
            "rollout_horizon": 4,
        },
        "objective_routing": {
            "state_supervision": "all",
            "effect_supervision": "random_half",
            "route_passed_to_model": False,
        },
    }


def test_h006_config_maps_research_arm_to_mixed_sampler(tmp_path: Path) -> None:
    path = tmp_path / "h006.yaml"
    path.write_text(yaml.safe_dump(_payload(), sort_keys=False), encoding="utf-8")

    config = H006RunConfig.from_yaml(path)

    assert config.arm is H006Arm.ROUTED_MIXED
    assert config.runtime.arm.value == (
        "mixed_probe_random_closed_loop_without_discrimination"
    )
    assert config.routing.effect_supervision == "random_half"
    assert config.routing.route_passed_to_model is False


def test_h006_preflight_routes_effect_without_opening_frozen_data(
    tmp_path: Path,
) -> None:
    path = tmp_path / "h006.yaml"
    path.write_text(yaml.safe_dump(_payload(), sort_keys=False), encoding="utf-8")

    result = run_h006_preflight(path, tmp_path / "run")

    assert result["hypothesis_id"] == "CHM-W-H006"
    assert result["arm"] == "routed_mixed_closed_loop_without_discrimination"
    assert result["objective_routing"]["effect_supervision"] == "random_half"
    assert result["first_training"]["effect_supervision_fraction"] == 0.5
    assert result["frozen_validation_seeds_opened"] is False
    assert result["test_metrics_opened"] is False
