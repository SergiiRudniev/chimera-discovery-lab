from __future__ import annotations

from pathlib import Path

import torch
import yaml

from chimera.meta_world.config import MetaWorldModelConfig, MetaWorldTrainingConfig
from chimera.meta_world.generators import SplitName, WorldGenerationPipeline
from chimera.meta_world.h004 import (
    H004DatasetConfig,
    build_h004_probe_dataset,
    run_h004_preflight,
    validate_h004_probe_dataset,
)


def _config() -> H004DatasetConfig:
    return H004DatasetConfig.from_yaml(
        "configs/meta_world/world_generators_h004.yaml"
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
        seed=260906,
        batch_size=4,
        active_slots=4,
        steps=2,
        learning_rate=1e-3,
        weight_decay=0.0,
        next_state_weight=1.0,
        effect_weight=0.5,
        alignment_weight=0.2,
        variance_weight=0.01,
        alignment_margin=0.2,
        primary_effect_weight=2.0,
        ema_decay=0.9,
        device="cpu",
        precision="float32",
    )


def test_probe_policy_has_registered_excitation_schedule() -> None:
    config = _config()
    pipeline = WorldGenerationPipeline(
        config.worlds,
        config.policies()[SplitName.TRAIN],
    )

    batch = pipeline.online_batch(SplitName.TRAIN, batch_size=4)

    torch.testing.assert_close(
        batch.actions[0, :8, 0],
        torch.tensor([0.0, 0.25, 0.85, 0.85, 0.0, 0.25, 0.85, 0.85]),
    )
    torch.testing.assert_close(
        batch.actions[0, :8, 1],
        torch.tensor([0.0, 0.0, 1.0, -1.0, 0.0, 0.0, -1.0, 1.0]),
    )
    assert torch.all(batch.action_targets.sum(dim=-1) == 0.0)
    assert not hasattr(batch, "action_policy_ids")


def test_hybrid_evaluation_policy_is_deterministic_and_changes_after_prefix() -> None:
    config = _config()
    hybrid = WorldGenerationPipeline(
        config.worlds,
        config.policies()[SplitName.VALIDATION],
    )
    random = WorldGenerationPipeline(
        config.worlds,
        config.random_training_policies()[SplitName.TRAIN],
    )

    first = hybrid.online_batch(SplitName.VALIDATION, batch_size=4)
    replay = hybrid.online_batch(SplitName.VALIDATION, batch_size=4)
    random_batch = random.online_batch(SplitName.VALIDATION, batch_size=4)

    assert torch.equal(first.actions, replay.actions)
    assert torch.equal(first.observations, replay.observations)
    torch.testing.assert_close(
        first.actions[:, :4, 0],
        torch.tensor([0.0, 0.25, 0.85, 0.85]).expand(4, -1),
    )
    assert not torch.equal(first.actions[:, 4:], random_batch.actions[:, 4:])


def test_fixed_wg1_dataset_replays_and_preserves_split_isolation(tmp_path: Path) -> None:
    manifest = build_h004_probe_dataset(
        tmp_path,
        "configs/meta_world/world_generators_h004.yaml",
        trajectories_per_split=8,
    )
    report = validate_h004_probe_dataset(tmp_path / "manifest.json")

    assert manifest["dataset_id"] == "CHM-W-WG1"
    assert report["status"] == "passed"
    assert report["counts"]["total"] == 40
    assert report["probe_response_separation"] > 0.0
    assert all(report["checks"].values())
    assert report["action_policies"]["train"].startswith(
        "deterministic_system_identification_probe"
    )
    assert report["action_policies"]["validation"].startswith("probe_prefix_4")


def test_h004_preflight_matches_policy_boundary_and_seals_test(tmp_path: Path) -> None:
    payload = {
        "run_id": "H004-PREFLIGHT-TEST",
        "mode": "preflight",
        "arm": "probe_curriculum_closed_loop_with_mechanism_discrimination",
        "dataset_config": "configs/meta_world/world_generators_h004.yaml",
        "model": _model_config().__dict__,
        "training": _training_config().__dict__,
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
    config_path = tmp_path / "h004.yaml"
    config_path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    output = tmp_path / "preflight"

    result = run_h004_preflight(config_path, output)

    assert result["status"] == "completed_development_preflight"
    assert result["train_action_policy"].startswith(
        "deterministic_system_identification_probe"
    )
    assert result["evaluation_action_policy"].startswith("probe_prefix_4")
    assert result["opened_splits"] == ["train", "validation"]
    assert result["test_metrics_opened"] is False
