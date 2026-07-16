from __future__ import annotations

from pathlib import Path

import torch
import yaml

from chimera.meta_world.config import MetaWorldModelConfig, MetaWorldTrainingConfig
from chimera.meta_world.generators import SplitName, WorldGenerationPipeline
from chimera.meta_world.h002 import (
    concatenate_sequence_samples,
    materialize_sequence_sample,
)
from chimera.meta_world.h004 import H004DatasetConfig, SeededRandomPolicy
from chimera.meta_world.h005 import run_h005_preflight, run_h005_validation


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
        seed=260910,
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


def test_mixed_sample_pairs_same_mechanisms_under_different_policies() -> None:
    config = H004DatasetConfig.from_yaml(
        "configs/meta_world/world_generators_h004.yaml"
    )
    probe_pipeline = WorldGenerationPipeline(
        config.worlds,
        config.policies()[SplitName.TRAIN],
    )
    random_pipeline = WorldGenerationPipeline(
        config.worlds,
        config.random_training_policies()[SplitName.TRAIN],
    )
    probe = materialize_sequence_sample(
        probe_pipeline,
        SplitName.TRAIN,
        start_index=0,
        batch_size=4,
    )
    random = materialize_sequence_sample(
        random_pipeline,
        SplitName.TRAIN,
        start_index=0,
        batch_size=4,
    )

    mixed = concatenate_sequence_samples(probe, random)

    assert mixed.batch.batch_size == 8
    assert torch.equal(mixed.mechanism_keys[:4], mixed.mechanism_keys[4:])
    assert torch.equal(mixed.trajectory_indices[:4], mixed.trajectory_indices[4:])
    assert not torch.equal(mixed.batch.actions[:4], mixed.batch.actions[4:])
    assert not hasattr(mixed.batch, "action_policy_ids")


def test_paired_random_views_match_worlds_but_change_actions() -> None:
    config = H004DatasetConfig.from_yaml(
        "configs/meta_world/world_generators_h004.yaml"
    )
    first_pipeline = WorldGenerationPipeline(
        config.worlds,
        SeededRandomPolicy(),
    )
    second_pipeline = WorldGenerationPipeline(
        config.worlds,
        SeededRandomPolicy(draw_offset=1),
    )
    first = materialize_sequence_sample(
        first_pipeline,
        SplitName.TRAIN,
        start_index=0,
        batch_size=4,
    )
    second = materialize_sequence_sample(
        second_pipeline,
        SplitName.TRAIN,
        start_index=0,
        batch_size=4,
    )

    assert torch.equal(first.mechanism_keys, second.mechanism_keys)
    assert torch.equal(first.trajectory_indices, second.trajectory_indices)
    assert not torch.equal(first.batch.actions, second.batch.actions)


def test_h005_development_preflight_keeps_frozen_seeds_and_test_closed(
    tmp_path: Path,
) -> None:
    payload = {
        "run_id": "H005-DEVELOPMENT-TEST",
        "mode": "preflight",
        "arm": "mixed_probe_random_closed_loop_without_discrimination",
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
    }
    config_path = tmp_path / "h005.yaml"
    config_path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    output = tmp_path / "preflight"

    result = run_h005_preflight(config_path, output)

    assert result["status"] == "completed_development_preflight"
    assert result["train_action_policy"] == "mixed_probe_0.5_seeded_random_0.5"
    assert result["frozen_validation_seeds_opened"] is False
    assert result["test_metrics_opened"] is False
    assert result["opened_splits"] == ["train", "validation"]


def test_h005_frozen_validation_uses_registered_final_step(tmp_path: Path) -> None:
    payload = {
        "run_id": "H005-VALIDATION-MIXED-S260911",
        "mode": "frozen_validation",
        "arm": "mixed_probe_random_closed_loop_without_discrimination",
        "dataset_config": "configs/meta_world/world_generators_h004.yaml",
        "model": _model_config().__dict__,
        "training": {**_training_config().__dict__, "seed": 260911},
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
        "frozen_checkpoint_step": 2,
    }
    config_path = tmp_path / "h005-validation.yaml"
    config_path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")

    result = run_h005_validation(config_path, tmp_path / "validation")

    assert result["status"] == "completed_frozen_validation"
    assert result["seed"] == 260911
    assert result["best_step"] == 2
    assert result["frozen_validation_seeds_opened"] is True
    assert result["test_metrics_opened"] is False
