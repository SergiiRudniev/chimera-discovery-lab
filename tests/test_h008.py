from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import pytest
import torch
import yaml

from chimera.meta_world.config import MetaWorldModelConfig, MetaWorldTrainingConfig
from chimera.meta_world.generators import SplitName, WorldGenerationPipeline
from chimera.meta_world.h002 import (
    make_transition_window,
    materialize_sequence_sample,
)
from chimera.meta_world.h004 import H004DatasetConfig
from chimera.meta_world.h005.preflight import execute_policy_curriculum_run
from chimera.meta_world.h008 import (
    CounterfactualRelationalWorldModel,
    DirectOutcomeRelationalWorldModel,
    H008Arm,
    H008RunConfig,
    H008SuiteConfig,
    evaluate_legal_random_interventions,
    run_h008_development_suite,
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
        seed=260922,
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


def _payload(arm: H008Arm) -> dict[str, object]:
    counterfactual = arm in {
        H008Arm.COUNTERFACTUAL_MIXED,
        H008Arm.COUNTERFACTUAL_RANDOM,
    }
    return {
        "run_id": f"H008-TEST-{arm.value}",
        "mode": "preflight",
        "arm": arm.value,
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
        "outcome_head": (
            "counterfactual_difference_v1" if counterfactual else "direct_effect"
        ),
    }


def test_counterfactual_head_is_exact_and_parameter_matched() -> None:
    dataset = H004DatasetConfig.from_yaml(
        "configs/meta_world/world_generators_h004.yaml"
    )
    sample = materialize_sequence_sample(
        WorldGenerationPipeline(
            dataset.worlds,
            dataset.policies()[SplitName.TRAIN],
        ),
        SplitName.TRAIN,
        start_index=0,
        batch_size=4,
    )
    window = make_transition_window(sample, prediction_step=3, context_steps=4)
    direct = DirectOutcomeRelationalWorldModel(_model_config()).eval()
    counterfactual = CounterfactualRelationalWorldModel(_model_config()).eval()
    counterfactual.load_state_dict(direct.state_dict(), strict=True)

    with torch.no_grad():
        direct_output = direct(window)
        output = counterfactual(window)

    assert sum(parameter.numel() for parameter in direct.parameters()) == sum(
        parameter.numel() for parameter in counterfactual.parameters()
    )
    assert output.counterfactual_no_op_mean is not None
    assert output.counterfactual_no_op_log_variance is not None
    torch.testing.assert_close(output.effect_mean[:, :3], direct_output.effect_mean[:, :3])
    torch.testing.assert_close(
        output.counterfactual_no_op_mean,
        direct_output.effect_mean[:, 3:4],
    )
    torch.testing.assert_close(
        output.effect_mean[:, 0:1] - output.effect_mean[:, 3:4],
        output.counterfactual_no_op_mean,
    )
    torch.testing.assert_close(
        output.effect_log_variance[:, 3:4],
        torch.logaddexp(
            direct_output.effect_log_variance[:, 0:1],
            direct_output.effect_log_variance[:, 3:4],
        ).clamp(min=-6.0, max=2.0),
    )


def test_h008_config_maps_every_registered_arm(tmp_path: Path) -> None:
    expected = {
        H008Arm.COUNTERFACTUAL_MIXED: "mixed_probe_random_closed_loop_without_discrimination",
        H008Arm.DIRECT_MIXED: "mixed_probe_random_closed_loop_without_discrimination",
        H008Arm.COUNTERFACTUAL_RANDOM: "random_only_closed_loop_without_discrimination",
        H008Arm.DIRECT_RANDOM: "random_only_closed_loop_without_discrimination",
        H008Arm.ONE_STEP: "one_step_relational_without_discrimination",
        H008Arm.TEMPORAL: "temporal_predictor_without_relational_state",
    }
    for arm, sampler in expected.items():
        path = tmp_path / f"{arm.value}.yaml"
        path.write_text(yaml.safe_dump(_payload(arm), sort_keys=False), encoding="utf-8")
        config = H008RunConfig.from_yaml(path)
        assert config.arm is arm
        assert config.runtime.arm.value == sampler


def test_h008_rejects_head_semantics_that_disagree_with_arm(tmp_path: Path) -> None:
    payload = _payload(H008Arm.COUNTERFACTUAL_MIXED)
    payload["outcome_head"] = "direct_effect"
    path = tmp_path / "invalid.yaml"
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")

    with pytest.raises(ValueError, match="semantics disagree"):
        H008RunConfig.from_yaml(path)


def test_h008_random_intervention_baseline_is_deterministic() -> None:
    worlds = H004DatasetConfig.from_yaml(
        "configs/meta_world/world_generators_h004.yaml"
    ).worlds
    first = evaluate_legal_random_interventions(
        worlds,
        samples=4,
        candidates_per_sample=4,
    )
    second = evaluate_legal_random_interventions(
        worlds,
        samples=4,
        candidates_per_sample=4,
    )

    assert first == second
    assert first.legal_action_rate == 1.0
    assert first.mean_intervention_regret >= 0.0


def test_h008_suite_runs_all_arms_and_keeps_test_sealed(tmp_path: Path) -> None:
    arm_paths: dict[str, str] = {}
    for arm in H008Arm:
        path = tmp_path / f"{arm.value}.yaml"
        path.write_text(yaml.safe_dump(_payload(arm), sort_keys=False), encoding="utf-8")
        arm_paths[arm.value] = path.as_posix()
    generator = Path("configs/meta_world/world_generators_h004.yaml")
    integrity = tmp_path / "integrity.json"
    integrity.write_text(
        json.dumps(
            {
                "dataset_config_sha256": hashlib.sha256(generator.read_bytes()).hexdigest(),
                "development_gate": {
                    "deterministic_dataset_replay_rate": 1.0,
                    "split_leakage_findings": 0,
                },
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    suite_path = tmp_path / "suite.yaml"
    suite_path.write_text(
        yaml.safe_dump(
            {
                "hypothesis_id": "CHM-W-H008",
                "trial_id": "CHM-W-T008",
                "seed": 260922,
                "generator_config": generator.as_posix(),
                "dataset_integrity_report": integrity.as_posix(),
                "arms": arm_paths,
                "legal_random_intervention": {
                    "samples": 4,
                    "candidates_per_sample": 4,
                },
                "development_gate": {
                    "intervention_effect_nrmse_ratio_maximum": 0.90,
                    "four_step_rollout_nrmse_ratio_maximum": 1.00,
                    "intervention_effect_90_coverage_minimum": 0.85,
                    "counterfactual_identity_maximum_absolute_residual": 0.000001,
                },
                "test_access": "sealed_until_development_gate_passes",
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )

    assert len(H008SuiteConfig.from_yaml(suite_path).arms) == 6
    result = run_h008_development_suite(
        suite_path,
        tmp_path / "runs",
        tmp_path / "report.json",
    )

    assert len(result["arms"]) == 6
    assert result["development_gate"]["parameter_count_matched"] is True
    assert result["dataset_integrity"]["revalidated"] is False
    assert result["legal_random_intervention"]["legal_action_rate"] == 1.0
    assert result["test_metrics_opened"] is False
    assert result["frozen_validation_seeds_opened"] is False
    assert math.isfinite(
        result["comparisons"]["counterfactual_vs_direct_mixed"][
            "intervention_effect_nrmse_ratio"
        ]
    )
    assert (
        result["arms"][H008Arm.COUNTERFACTUAL_MIXED.value][
            "counterfactual_audit"
        ]["counterfactual_identity_maximum_absolute_residual"]
        <= 0.000001
    )


def test_policy_runner_reseeds_model_initialization(tmp_path: Path) -> None:
    arm = H008Arm.DIRECT_MIXED
    config_path = tmp_path / "direct.yaml"
    config_path.write_text(
        yaml.safe_dump(_payload(arm), sort_keys=False),
        encoding="utf-8",
    )
    config = H008RunConfig.from_yaml(config_path)

    first = execute_policy_curriculum_run(
        config_path,
        config.runtime,
        tmp_path / "first",
        expected_mode="preflight",
        hypothesis_id="CHM-W-H008",
        reported_arm=arm.value,
        effect_supervision="all",
    )
    torch.manual_seed(999_999)
    second = execute_policy_curriculum_run(
        config_path,
        config.runtime,
        tmp_path / "second",
        expected_mode="preflight",
        hypothesis_id="CHM-W-H008",
        reported_arm=arm.value,
        effect_supervision="all",
    )

    assert first["initial_validation"] == second["initial_validation"]
    assert first["best_validation"] == second["best_validation"]
