from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from chimera.meta_world.generators import (
    GeneratedWorldDatasetConfig,
    SplitName,
    TransferTemplatePolicy,
)
from chimera.meta_world.h018 import preflight as h018_preflight_module
from chimera.meta_world.h018 import suite as h018_suite_module
from chimera.meta_world.h018.config import H018SuiteConfig
from chimera.meta_world.h018.dataset import (
    build_h018_smoke_dataset,
    make_h018_pipeline,
    validate_h018_dataset,
)
from chimera.meta_world.h018.programs import (
    MECHANISM_TEST_PROGRAM_IDS,
    PROGRAM_SPECS,
    TRAIN_PROGRAM_IDS,
    TRANSFER_PROGRAM_IDS,
    MechanismProgramGenerator,
    operator_ids,
)

GENERATOR = Path("configs/meta_world/world_generators_h018.yaml")
SUITE = Path("configs/meta_world/world_h018_suite.yaml")


def test_h018_suite_and_program_split_are_frozen() -> None:
    suite = H018SuiteConfig.from_yaml(SUITE)
    generator = GeneratedWorldDatasetConfig.from_yaml(GENERATOR)
    assert suite.development_seed == 260962
    assert suite.frozen_validation_seeds == (260963, 260964, 260965)
    assert len(suite.arms) == 5
    assert generator.schema_version == 4
    assert (
        generator.transfer_template_policy
        is TransferTemplatePolicy.HELD_OUT_COMPOSITION
    )
    assert set(generator.split(SplitName.TRAIN).mechanism_template_ids) == set(
        TRAIN_PROGRAM_IDS
    )
    assert set(
        generator.split(SplitName.TEST_WORLD_TRANSFER).mechanism_template_ids
    ) == set(TRANSFER_PROGRAM_IDS)
    assert set(generator.split(SplitName.TEST_MECHANISM).mechanism_template_ids) == set(
        MECHANISM_TEST_PROGRAM_IDS
    )


def test_h018_mechanism_programs_are_deterministic_and_numerically_valid() -> None:
    generator = MechanismProgramGenerator()
    for program_id in PROGRAM_SPECS:
        first = generator.generate(program_id, 991)
        replay = generator.generate(program_id, 991)
        changed = generator.generate(program_id, 992)
        assert first.mechanism_id == replay.mechanism_id
        assert first.mechanism_id != changed.mechanism_id
        assert np.array_equal(first.latent_weights, replay.latent_weights)
        assert np.isfinite(first.latent_weights).all()
        assert 0.0 < first.retention <= 1.0
        assert 0.0 <= first.event_rate <= 1.0
    with pytest.raises(ValueError, match="unknown"):
        generator.generate(99, 1)


def test_h018_transfer_compositions_are_unseen_but_primitives_are_known() -> None:
    assert TRAIN_PROGRAM_IDS.isdisjoint(TRANSFER_PROGRAM_IDS)
    assert operator_ids(TRANSFER_PROGRAM_IDS).issubset(
        operator_ids(TRAIN_PROGRAM_IDS)
    )
    assert all(
        len(PROGRAM_SPECS[item].operators) == 2
        for item in TRAIN_PROGRAM_IDS | TRANSFER_PROGRAM_IDS
    )
    assert all(
        len(PROGRAM_SPECS[item].operators) == 3
        for item in MECHANISM_TEST_PROGRAM_IDS
    )


def test_h018_schema_rejects_train_transfer_program_overlap() -> None:
    config = GeneratedWorldDatasetConfig.from_yaml(GENERATOR)
    transfer = replace(
        config.split(SplitName.TEST_WORLD_TRANSFER),
        mechanism_template_ids=(0, 6, 7, 8),
    )
    splits = tuple(
        transfer if item.name is SplitName.TEST_WORLD_TRANSFER else item
        for item in config.split_configs
    )
    with pytest.raises(ValueError, match="absent from train"):
        replace(config, split_configs=splits)


def test_h018_pipeline_replays_paired_renderers_without_metadata_input() -> None:
    config = GeneratedWorldDatasetConfig.from_yaml(GENERATOR)
    pipeline = make_h018_pipeline(config)
    first = pipeline.materialize(SplitName.TEST_WORLD_TRANSFER, 0)
    second = pipeline.materialize(SplitName.TEST_WORLD_TRANSFER, 1)
    replay = make_h018_pipeline(config).materialize(
        SplitName.TEST_WORLD_TRANSFER, 0
    )
    assert first.metadata.mechanism_template_id in TRANSFER_PROGRAM_IDS
    assert first.metadata.mechanism_id == second.metadata.mechanism_id
    assert first.metadata.world_instance_id == second.metadata.world_instance_id
    assert first.metadata.renderer_id != second.metadata.renderer_id
    assert first.metadata == replay.metadata
    assert all(
        np.array_equal(left.outcome, right.outcome)
        for left, right in zip(first.transitions, second.transitions, strict=True)
    )
    batch = pipeline.online_batch(SplitName.TRAIN, 4)
    batch.validate()
    assert tuple(batch.observations.shape) == (4, 8, 10, 8)
    for service_field in (
        "world_family_id",
        "world_instance_id",
        "mechanism_id",
        "mechanism_program_id",
        "renderer_id",
        "generation_seed",
    ):
        assert not hasattr(batch, service_field)


def test_h018_fixed_dataset_is_sha_exact_and_leakage_free(tmp_path: Path) -> None:
    first = tmp_path / "first"
    second = tmp_path / "second"
    first_report = build_h018_smoke_dataset(first, GENERATOR)
    second_report = build_h018_smoke_dataset(second, GENERATOR)
    assert first_report["status"] == "passed"
    assert first_report["manifest_sha256"] == second_report["manifest_sha256"]
    assert first_report["programs"]["exact_train_transfer_overlap"] == 0
    assert first_report["checks"]["all_transfer_primitives_seen_in_train"] is True
    assert first_report["checks"]["deterministic_replay"] is True
    assert first_report["checks"]["exact_configuration_isolation"] is True
    first_manifest = json.loads((first / "manifest.json").read_text(encoding="utf-8"))
    second_manifest = json.loads((second / "manifest.json").read_text(encoding="utf-8"))
    assert first_manifest == second_manifest
    assert validate_h018_dataset(first / "manifest.json")["status"] == "passed"


def test_h018_preflight_injects_program_generator_and_keeps_test_sealed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    observed: dict[str, object] = {}

    def fake_run(
        config_path: object,
        output_dir: object,
        **kwargs: object,
    ) -> dict[str, object]:
        del config_path, output_dir
        config = GeneratedWorldDatasetConfig.from_yaml(GENERATOR)
        training_factory = kwargs["training_pipeline_factory"]
        validation_factory = kwargs["validation_pipeline_factory"]
        assert callable(training_factory)
        assert callable(validation_factory)
        train_pipeline = training_factory(config)
        validation_pipeline = validation_factory(config)
        observed["train_family"] = train_pipeline.materialize(
            SplitName.TRAIN, 0
        ).metadata.world_family_id
        observed["validation_program"] = validation_pipeline.materialize(
            SplitName.VALIDATION, 0
        ).metadata.mechanism_template_id
        return {
            "run_id": "h018-test",
            "hypothesis_id": "CHM-W-H018",
            "status": "completed_preflight",
            "arm": "target_family_only_training",
            "parameters": 1,
            "best_step": 0,
            "test_metrics_opened": False,
        }

    monkeypatch.setattr(
        h018_preflight_module,
        "run_generated_world_preflight",
        fake_run,
    )
    monkeypatch.setattr(
        h018_preflight_module,
        "evaluate_h018_random_interventions",
        lambda config: SimpleNamespace(to_dict=lambda: {"legal_action_rate": 1.0}),
    )
    output = tmp_path / "preflight"
    output.mkdir()
    result = h018_preflight_module.run_h018_preflight(
        "configs/meta_world/world_h018_development_target_family.yaml",
        output,
    )
    assert result["training_family_policy"] == "held_target"
    assert result["test_metrics_opened"] is False
    assert result["mechanism_program_metadata_passed_to_model"] is False
    assert observed["validation_program"] == 0


def test_h018_suite_runs_all_controls_without_opening_transfer(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        h018_suite_module,
        "build_h018_smoke_dataset",
        lambda *args, **kwargs: {
            "status": "passed",
            "checks": {
                "deterministic_replay": True,
                "mechanism_id_isolation": True,
                "world_instance_isolation": True,
                "seed_isolation": True,
                "exact_configuration_isolation": True,
                "exact_train_transfer_program_overlap_zero": True,
                "all_transfer_primitives_seen_in_train": True,
            },
            "programs": {"exact_train_transfer_overlap": 0},
        },
    )

    def fake_preflight(config: Path, output: Path) -> dict[str, object]:
        del output
        name = config.stem
        effect = 0.8 if "aligned" in name and "no_alignment" not in name else 1.0
        return {
            "status": "completed_preflight",
            "test_metrics_opened": False,
            "best_validation": {
                "intervention_effect_nrmse": effect,
                "four_step_rollout_nrmse": effect,
            },
        }

    monkeypatch.setattr(h018_suite_module, "run_h018_preflight", fake_preflight)
    monkeypatch.setattr(
        h018_suite_module,
        "evaluate_h018_random_interventions",
        lambda *args, **kwargs: SimpleNamespace(
            to_dict=lambda: {"legal_action_rate": 1.0}
        ),
    )
    report_path = tmp_path / "reports" / "h018.json"
    report = h018_suite_module.run_h018_development_suite(
        SUITE,
        tmp_path / "runs",
        report_path,
    )
    assert len(report["arms"]) == 5
    assert report["test_metrics_opened"] is False
    assert report["checkpoint_promoted"] is False
    assert report["decision"] == "engineering_gate_passed_test_remains_sealed"
    assert report["validation_diagnostic"]["primary_transfer_gate_evaluated"] is False
    assert json.loads(report_path.read_text(encoding="utf-8"))["scientific_result"] is False
