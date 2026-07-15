from __future__ import annotations

from pathlib import Path

import torch

from chimera.meta_world.generators import SplitName, WorldGenerationPipeline
from chimera.meta_world.h004 import (
    H004DatasetConfig,
    build_h004_probe_dataset,
    validate_h004_probe_dataset,
)


def _config() -> H004DatasetConfig:
    return H004DatasetConfig.from_yaml(
        "configs/meta_world/world_generators_h004.yaml"
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
