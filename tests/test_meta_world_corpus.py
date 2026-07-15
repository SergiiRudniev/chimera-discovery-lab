from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch

from chimera.meta_world.config import MetaWorldModelConfig
from chimera.meta_world.corpus import (
    MetaWorldCorpusSplit,
    build_meta_world_corpus,
    validate_meta_world_corpus,
)
from chimera.meta_world.model import ChimeraMetaWorld


def _build_small_corpus(tmp_path: Path) -> tuple[Path, dict[str, object]]:
    output = tmp_path / "meta_world_c0"
    manifest = build_meta_world_corpus(
        output,
        "configs/meta_world/meta_world_w0_t1.yaml",
        train_repeats=10,
        evaluation_repeats=1,
        transfer_repeats=1,
    )
    return output, manifest


def test_meta_world_corpus_builds_balanced_isolated_index(tmp_path: Path) -> None:
    output, manifest = _build_small_corpus(tmp_path)
    assert manifest["counts"] == {
        "total": 1_184,
        "train": 960,
        "validation": 96,
        "test": 96,
        "transfer": 32,
    }
    report = validate_meta_world_corpus(output / "manifest.json")
    assert report["status"] == "passed"
    assert all(report["checks"].values())
    assert report["profile"]["sampled_exact_duplicates"] == 0
    assert report["profile"]["persistence_rmse"] > 0


def test_meta_world_corpus_records_are_globally_unique(tmp_path: Path) -> None:
    output, _ = _build_small_corpus(tmp_path)
    record_ids: list[np.ndarray] = []
    record_seeds: list[np.ndarray] = []
    for name in ("train", "validation", "test", "transfer"):
        split = MetaWorldCorpusSplit(output / f"{name}.npz")
        record_ids.append(split.arrays["record_ids"])
        record_seeds.append(split.arrays["record_seeds"])
    combined_ids = np.concatenate(record_ids)
    combined_seeds = np.concatenate(record_seeds)
    assert np.unique(combined_ids).size == combined_ids.size
    assert np.unique(combined_seeds).size == combined_seeds.size


def test_meta_world_corpus_materialization_is_order_independent(tmp_path: Path) -> None:
    output, manifest = _build_small_corpus(tmp_path)
    config = MetaWorldModelConfig.from_mapping(manifest["model"]["config"])
    split = MetaWorldCorpusSplit(output / "train.npz")
    first = split.batch(
        [0, 1],
        config,
        active_slots=manifest["generation"]["active_slots"],
        transform_seed=manifest["generation"]["transform_seed"],
    )
    reversed_batch = split.batch(
        [1, 0],
        config,
        active_slots=manifest["generation"]["active_slots"],
        transform_seed=manifest["generation"]["transform_seed"],
    )
    torch.testing.assert_close(first.observations[0], reversed_batch.observations[1])
    torch.testing.assert_close(first.next_observations[1], reversed_batch.next_observations[0])


def test_meta_world_manifest_is_json_and_object_free(tmp_path: Path) -> None:
    output, built_manifest = _build_small_corpus(tmp_path)
    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["storage"] == "procedural_index"
    assert manifest["model"]["config"] == built_manifest["model"]["config"]
    config = MetaWorldModelConfig.from_mapping(manifest["model"]["config"])
    with torch.device("meta"):
        model = ChimeraMetaWorld(config)
    assert manifest["model"]["parameters"] == model.trainable_parameter_count()
    with np.load(output / "train.npz", allow_pickle=False) as shard:
        assert all(shard[name].dtype == np.int64 for name in shard.files)


def test_meta_world_corpus_rejects_partial_era_coverage(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="train repeats"):
        build_meta_world_corpus(
            tmp_path / "invalid",
            "configs/meta_world/meta_world_w0_t1.yaml",
            train_repeats=2,
            evaluation_repeats=1,
            transfer_repeats=1,
        )


def test_meta_world_corpus_build_is_byte_reproducible(tmp_path: Path) -> None:
    first, first_manifest = _build_small_corpus(tmp_path / "first")
    second, second_manifest = _build_small_corpus(tmp_path / "second")
    assert first_manifest["files"] == second_manifest["files"]
    for name in first_manifest["files"]:
        assert (first / name).read_bytes() == (second / name).read_bytes()


def test_meta_world_corpus_detects_manifest_count_tampering(tmp_path: Path) -> None:
    output, _ = _build_small_corpus(tmp_path)
    manifest_path = output / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["counts"]["train"] += 1
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    report = validate_meta_world_corpus(manifest_path)
    assert report["status"] == "failed"
    assert report["checks"]["count_consistency"] is False
