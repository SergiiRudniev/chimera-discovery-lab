from __future__ import annotations

import math
from pathlib import Path

import torch
import yaml

from chimera.meta_world.config import MetaWorldModelConfig, MetaWorldTrainingConfig
from chimera.meta_world.h007 import (
    H007Arm,
    H007RunConfig,
    project_task_gradients,
    run_h007_preflight,
)


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
        seed=260918,
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
        "run_id": "H007-DEVELOPMENT-TEST",
        "mode": "preflight",
        "arm": "pcgrad_mixed_closed_loop_without_discrimination",
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
        "gradient_intervention": "symmetric_global_pcgrad_v1",
    }


def test_symmetric_pcgrad_projects_conflicting_shared_gradients() -> None:
    combined, metrics = project_task_gradients(
        (torch.tensor([1.0, 0.0]),),
        (torch.tensor([-1.0, 1.0]),),
    )

    assert metrics.conflict_applied is True
    assert math.isclose(metrics.cosine, -(1.0 / math.sqrt(2.0)))
    assert combined[0] is not None
    torch.testing.assert_close(combined[0], torch.tensor([0.5, 1.5]))


def test_pcgrad_leaves_non_conflicting_sum_unchanged() -> None:
    combined, metrics = project_task_gradients(
        (torch.tensor([1.0, 0.0]),),
        (torch.tensor([1.0, 1.0]),),
    )

    assert metrics.conflict_applied is False
    assert combined[0] is not None
    torch.testing.assert_close(combined[0], torch.tensor([2.0, 1.0]))


def test_h007_config_maps_pcgrad_arm_to_mixed_sampler(tmp_path: Path) -> None:
    path = tmp_path / "h007.yaml"
    path.write_text(yaml.safe_dump(_payload(), sort_keys=False), encoding="utf-8")

    config = H007RunConfig.from_yaml(path)

    assert config.arm is H007Arm.PCGRAD_MIXED
    assert config.runtime.arm.value == (
        "mixed_probe_random_closed_loop_without_discrimination"
    )
    assert config.gradient_intervention == "symmetric_global_pcgrad_v1"


def test_h007_preflight_records_gradient_geometry_and_seals_data(
    tmp_path: Path,
) -> None:
    path = tmp_path / "h007.yaml"
    path.write_text(yaml.safe_dump(_payload(), sort_keys=False), encoding="utf-8")

    result = run_h007_preflight(path, tmp_path / "run")

    assert result["hypothesis_id"] == "CHM-W-H007"
    assert result["gradient_intervention"] == "symmetric_global_pcgrad_v1"
    assert result["gradient_task_id_passed_to_model"] is False
    assert 0.0 <= result["training_diagnostics"]["gradient_conflict_fraction"] <= 1.0
    assert result["frozen_validation_seeds_opened"] is False
    assert result["test_metrics_opened"] is False
