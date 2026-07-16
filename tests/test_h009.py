from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from chimera.meta_world.generators import (
    GeneratedWorldDatasetConfig,
    SplitName,
    ViewCoupling,
    WorldGenerationPipeline,
    build_generated_world_dataset,
    validate_generated_world_dataset,
)
from chimera.meta_world.h009 import run_h009_preflight

GENERATOR_CONFIG = Path("configs/meta_world/world_generators_h009.yaml")
SMOKE_CONFIG = Path("configs/meta_world/world_h009_development_smoke.yaml")


def test_h009_config_registers_paired_renderer_views() -> None:
    config = GeneratedWorldDatasetConfig.from_yaml(GENERATOR_CONFIG)

    assert config.hypothesis_id == "CHM-W-H009"
    assert config.dataset_id == "CHM-W-WG2"
    assert config.schema_version == 2
    assert config.view_coupling is ViewCoupling.PAIRED_WORLD_RENDERERS
    assert config.views_per_mechanism == 4
    assert config.render_views_per_world == 2
    assert GeneratedWorldDatasetConfig.from_mapping(config.to_dict()) == config


def test_h009_renderer_pairs_share_hidden_trajectory_only() -> None:
    config = GeneratedWorldDatasetConfig.from_yaml(GENERATOR_CONFIG)
    pipeline = WorldGenerationPipeline(config)
    views = [pipeline.materialize(SplitName.TRAIN, index) for index in range(4)]

    assert len({item.metadata.mechanism_id for item in views}) == 1
    assert views[0].metadata.world_instance_id == views[1].metadata.world_instance_id
    assert views[2].metadata.world_instance_id == views[3].metadata.world_instance_id
    assert views[0].metadata.world_instance_id != views[2].metadata.world_instance_id
    assert views[0].metadata.world_family_id != views[2].metadata.world_family_id
    assert views[0].metadata.generation_seed == views[1].metadata.generation_seed
    assert views[0].metadata.renderer_id != views[1].metadata.renderer_id
    assert not np.array_equal(
        views[0].initial_observation.values,
        views[1].initial_observation.values,
    )
    for left, right in ((views[0], views[1]), (views[2], views[3])):
        for left_step, right_step in zip(
            left.transitions,
            right.transitions,
            strict=True,
        ):
            assert left_step.action.magnitude == right_step.action.magnitude
            assert left_step.action.control == right_step.action.control
            assert np.array_equal(left_step.outcome, right_step.outcome)


def test_h009_fixed_dataset_is_reproducible_and_leakage_free(tmp_path: Path) -> None:
    first = tmp_path / "first"
    second = tmp_path / "second"
    first_manifest = build_generated_world_dataset(
        first,
        GENERATOR_CONFIG,
        trajectories_per_split=16,
    )
    second_manifest = build_generated_world_dataset(
        second,
        GENERATOR_CONFIG,
        trajectories_per_split=16,
    )

    assert first_manifest == second_manifest
    for split in SplitName:
        assert (first / f"{split.value}.npz").read_bytes() == (
            second / f"{split.value}.npz"
        ).read_bytes()
    report = validate_generated_world_dataset(first / "manifest.json")
    assert report["status"] == "passed"
    assert all(report["checks"].values())  # type: ignore[union-attr]
    assert report["checks"]["paired_renderer_trajectory_consistency"] is True  # type: ignore[index]
    with np.load(first / "train.npz", allow_pickle=False) as arrays:
        assert arrays["world_instance_ids"][0] == arrays["world_instance_ids"][1]
        assert arrays["renderer_ids"][0] != arrays["renderer_ids"][1]
        assert np.array_equal(arrays["outcomes"][0], arrays["outcomes"][1])


def test_h009_smoke_preflight_keeps_test_sealed(tmp_path: Path) -> None:
    result = run_h009_preflight(SMOKE_CONFIG, tmp_path)
    persisted = json.loads((tmp_path / "result.json").read_text(encoding="utf-8"))

    assert result["hypothesis_id"] == "CHM-W-H009"
    assert result["status"] == "completed_preflight"
    assert result["opened_splits"] == ["train", "validation"]
    assert result["test_metrics_opened"] is False
    assert result["best_step"] in {0, 1, 2}
    assert persisted == result
    assert not (tmp_path / "test_world_transfer.npz").exists()
