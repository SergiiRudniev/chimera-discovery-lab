from __future__ import annotations

import json
from pathlib import Path

from chimera.meta_world.generators import (
    GeneratedWorldDatasetConfig,
    SplitName,
    TrainingFamilyPolicy,
    WorldGenerationPipeline,
)
from chimera.meta_world.h012 import (
    H012SuiteConfig,
    build_h012_smoke_dataset,
    evaluate_legal_random_interventions,
    run_h012_preflight,
)

GENERATOR_CONFIG = Path("configs/meta_world/world_generators_h012.yaml")
SUITE_CONFIG = Path("configs/meta_world/world_h012_suite.yaml")
SMOKE_CONFIG = Path("configs/meta_world/world_h012_development_smoke.yaml")
TARGET_SMOKE_CONFIG = Path("configs/meta_world/world_h012_target_smoke.yaml")


def test_h012_suite_freezes_five_arms_and_test_boundary() -> None:
    suite = H012SuiteConfig.from_yaml(SUITE_CONFIG)

    assert suite.hypothesis_id == "CHM-W-H012"
    assert suite.trial_id == "CHM-W-T012"
    assert len(suite.arms) == 5
    assert suite.primary_split is SplitName.TEST_WORLD_TRANSFER
    assert suite.test_access == "sealed_until_all_validation_decisions_are_frozen"


def test_target_family_policy_changes_online_train_allocation_only() -> None:
    config = GeneratedWorldDatasetConfig.from_yaml(GENERATOR_CONFIG)
    cross_world = WorldGenerationPipeline(config)
    target_only = WorldGenerationPipeline(
        config,
        training_family_policy=TrainingFamilyPolicy.HELD_TARGET,
    )

    for index in range(config.views_per_mechanism):
        source = cross_world.materialize(SplitName.TRAIN, index)
        target = target_only.materialize(SplitName.TRAIN, index)
        held = config.held_family_by_template[target.metadata.mechanism_template_id]

        assert source.metadata.world_family_id != held
        assert target.metadata.world_family_id == held
        assert target.metadata.mechanism_id == source.metadata.mechanism_id
        assert target.metadata.generation_seed == source.metadata.generation_seed

    source_validation = cross_world.materialize(SplitName.VALIDATION, 0)
    target_validation = target_only.materialize(SplitName.VALIDATION, 0)
    assert target_validation.metadata == source_validation.metadata


def test_h012_fixed_dataset_is_reproducible_and_leakage_free(tmp_path: Path) -> None:
    first = tmp_path / "first"
    second = tmp_path / "second"

    first_report = build_h012_smoke_dataset(first, GENERATOR_CONFIG)
    second_report = build_h012_smoke_dataset(second, GENERATOR_CONFIG)

    assert first_report["status"] == "passed"
    assert first_report["checks"] == second_report["checks"]
    assert first_report["manifest_sha256"] == second_report["manifest_sha256"]
    assert first_report["counts"]["total"] == 80  # type: ignore[index]
    assert all(first_report["checks"].values())  # type: ignore[union-attr]
    for split in SplitName:
        assert (first / f"{split.value}.npz").read_bytes() == (
            second / f"{split.value}.npz"
        ).read_bytes()


def test_legal_random_baseline_is_deterministic_and_numeric() -> None:
    config = GeneratedWorldDatasetConfig.from_yaml(GENERATOR_CONFIG)

    first = evaluate_legal_random_interventions(config, samples=4, candidates_per_sample=6)
    second = evaluate_legal_random_interventions(
        config,
        samples=4,
        candidates_per_sample=6,
    )

    assert first == second
    assert first.legal_action_rate == 1.0
    assert first.mean_intervention_regret >= 0.0
    assert first.mean_best_candidate_effect >= first.mean_selected_effect


def test_h012_smoke_preflight_keeps_test_sealed(tmp_path: Path) -> None:
    result = run_h012_preflight(SMOKE_CONFIG, tmp_path)
    persisted = json.loads((tmp_path / "result.json").read_text(encoding="utf-8"))

    assert result == persisted
    assert result["hypothesis_id"] == "CHM-W-H012"
    assert result["status"] == "completed_preflight"
    assert result["opened_splits"] == ["train", "validation"]
    assert result["test_metrics_opened"] is False
    assert result["scientific_result"] is False
    assert result["training_family_policy"] == "cross_world"
    assert result["validation_random_intervention"]["legal_action_rate"] == 1.0
    assert not (tmp_path / "test_world_transfer.npz").exists()


def test_h012_target_smoke_uses_held_family_training(tmp_path: Path) -> None:
    result = run_h012_preflight(TARGET_SMOKE_CONFIG, tmp_path)

    assert result["arm"] == "target_family_only_training"
    assert result["training_family_policy"] == "held_target"
    assert result["opened_splits"] == ["train", "validation"]
    assert result["test_metrics_opened"] is False
