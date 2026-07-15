from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch

from chimera.meta_world.generators import (
    CompetitionWorld,
    FlowWorld,
    FunnelWorld,
    GeneratedWorldDatasetConfig,
    MechanismGenerator,
    SplitName,
    WorldAction,
    WorldFamily,
    WorldGenerationPipeline,
    WorldGenerator,
    build_generated_world_dataset,
    collate_trajectories,
    validate_generated_world_dataset,
)

CONFIG = Path("configs/meta_world/world_generators_h002.yaml")


def test_generated_world_config_round_trips() -> None:
    config = GeneratedWorldDatasetConfig.from_yaml(CONFIG)

    replay = GeneratedWorldDatasetConfig.from_mapping(config.to_dict())

    assert replay == config
    assert replay.hypothesis_id == "CHM-W-H002"
    assert replay.split(SplitName.TEST_MECHANISM).mechanism_template_ids == (4, 5)


def test_all_world_families_replay_and_emit_finite_transitions() -> None:
    mechanism = MechanismGenerator().generate(template_id=0, seed=17)
    generator = WorldGenerator(min_objects=4, max_objects=4)
    expected_types = {
        WorldFamily.FLOW: FlowWorld,
        WorldFamily.COMPETITION: CompetitionWorld,
        WorldFamily.FUNNEL: FunnelWorld,
    }
    for family, expected_type in expected_types.items():
        world = generator.generate(
            mechanism,
            family,
            world_seed=101 + int(family),
            renderer_seed=201 + int(family),
            renderer_profile=int(family),
        )
        assert isinstance(world, expected_type)
        with pytest.raises(RuntimeError, match="reset"):
            world.step(WorldAction(0, 1, 0.5, 0.0))
        first = world.reset(301)
        action_rng = np.random.default_rng(401)
        action = world.sample_action(action_rng)
        transition = world.step(action)
        second = world.reset(301)
        replay_action = world.sample_action(np.random.default_rng(401))
        replay_transition = world.step(replay_action)

        assert np.array_equal(first.values, second.values)
        assert action == replay_action
        assert np.array_equal(
            transition.observation.values, replay_transition.observation.values
        )
        assert np.array_equal(transition.outcome, replay_transition.outcome)
        assert transition.outcome.shape == (4,)
        assert np.isfinite(transition.outcome).all()
        with pytest.raises(ValueError, match="differ"):
            world.step(WorldAction(0, 0, 0.5, 0.0))


def test_pipeline_separates_metadata_and_aligns_two_views() -> None:
    config = GeneratedWorldDatasetConfig.from_yaml(CONFIG)
    pipeline = WorldGenerationPipeline(config)
    first = pipeline.materialize(SplitName.TRAIN, 0)
    second_view = pipeline.materialize(SplitName.TRAIN, 1)
    validation = pipeline.materialize(SplitName.VALIDATION, 0)
    transfer = pipeline.materialize(SplitName.TEST_WORLD_TRANSFER, 0)
    unseen = pipeline.materialize(SplitName.TEST_MECHANISM, 0)
    renderer = pipeline.materialize(SplitName.TEST_RENDERER, 0)

    assert first.metadata.mechanism_id == second_view.metadata.mechanism_id
    assert first.metadata.world_instance_id != second_view.metadata.world_instance_id
    assert first.metadata.mechanism_id != validation.metadata.mechanism_id
    assert transfer.metadata.world_family_id == config.held_family_by_template[
        transfer.metadata.mechanism_template_id
    ]
    assert unseen.metadata.mechanism_template_id in {4, 5}
    assert renderer.metadata.renderer_profile_id == 2
    replay = pipeline.materialize(SplitName.TRAIN, 0)
    for left, right in zip(first.transitions, replay.transitions, strict=True):
        assert left.action == right.action
        assert np.array_equal(left.observation.values, right.observation.values)
        assert np.array_equal(left.outcome, right.outcome)
    with pytest.raises(ValueError, match="trajectory index"):
        pipeline.materialize(SplitName.TRAIN, -1)


def test_online_batch_matches_language_free_tensor_contract() -> None:
    config = GeneratedWorldDatasetConfig.from_yaml(CONFIG)
    batch = WorldGenerationPipeline(config).online_batch(SplitName.TRAIN, 8)

    assert batch.observations.shape == (8, 8, 10, 8)
    assert batch.object_mask.shape == (8, 8, 10)
    assert batch.relations.shape == (8, 8, 10, 10, 4)
    assert batch.relation_mask.shape == (8, 8, 10, 10)
    assert batch.actions.shape == (8, 8, 2)
    assert batch.action_targets.shape == (8, 8, 10)
    assert batch.delta_time.shape == (8, 8)
    assert batch.outcomes.shape == (8, 8, 4)
    assert batch.sequence_mask.all()
    assert torch.all((batch.action_targets != 0).sum(dim=-1) == 2)
    assert not hasattr(batch, "mechanism_id")
    assert not hasattr(batch, "world_family_id")
    batch.validate()
    with pytest.raises(ValueError, match="at least one"):
        collate_trajectories([], max_objects=10)


def test_fixed_dataset_is_byte_reproducible_and_leakage_free(tmp_path: Path) -> None:
    first = tmp_path / "first"
    second = tmp_path / "second"
    first_manifest = build_generated_world_dataset(
        first,
        CONFIG,
        trajectories_per_split=8,
    )
    second_manifest = build_generated_world_dataset(
        second,
        CONFIG,
        trajectories_per_split=8,
    )

    assert first_manifest == second_manifest
    for split in SplitName:
        assert (first / f"{split.value}.npz").read_bytes() == (
            second / f"{split.value}.npz"
        ).read_bytes()
    report = validate_generated_world_dataset(first / "manifest.json")
    assert report["status"] == "passed"
    assert all(report["checks"].values())
    assert report["counts"]["total"] == 40
    with np.load(first / "train.npz", allow_pickle=False) as arrays:
        assert arrays["observations"].dtype == np.float32
        assert arrays["object_mask"].dtype == np.bool_
        assert arrays["mechanism_ids"].dtype.kind == "U"
        assert arrays["mechanism_ids"][0] == arrays["mechanism_ids"][1]


def test_fixed_dataset_detects_manifest_tampering(tmp_path: Path) -> None:
    build_generated_world_dataset(tmp_path, CONFIG, trajectories_per_split=8)
    manifest_path = tmp_path / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["shards"]["train"]["sha256"] = "0" * 64
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    report = validate_generated_world_dataset(manifest_path)

    assert report["status"] == "failed"
    assert report["checks"]["file_integrity"] is False


def test_fixed_dataset_rejects_incomplete_alignment_groups(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="cover every template"):
        build_generated_world_dataset(tmp_path, CONFIG, trajectories_per_split=2)
