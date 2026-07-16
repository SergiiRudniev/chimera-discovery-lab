from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import torch

from chimera.meta_world.config import MetaWorldModelConfig
from chimera.meta_world.generators import (
    GeneratedWorldDatasetConfig,
    SplitName,
    WorldGenerationPipeline,
)
from chimera.meta_world.h002.windows import (
    make_transition_window,
    materialize_sequence_sample,
)
from chimera.meta_world.h014 import preflight as h014_preflight_module
from chimera.meta_world.h014 import suite as h014_suite_module
from chimera.meta_world.h014.config import H014Arm, H014RunConfig
from chimera.meta_world.h014.model import (
    ResponseConditionedEffectWorldModel,
    ResponseSource,
)
from chimera.meta_world.h014.suite import H014SuiteConfig, run_h014_development_suite

GENERATOR = Path("configs/meta_world/world_generators_h013.yaml")


def _model_config() -> MetaWorldModelConfig:
    return MetaWorldModelConfig(
        observation_features=8,
        relation_features=4,
        intervention_types=1,
        intervention_parameters=3,
        effect_dimensions=4,
        domain_count=1,
        mechanism_count=8,
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


def _window() -> object:
    config = GeneratedWorldDatasetConfig.from_yaml(GENERATOR)
    sample = materialize_sequence_sample(
        WorldGenerationPipeline(config),
        SplitName.VALIDATION,
        start_index=0,
        batch_size=4,
    )
    return make_transition_window(sample, prediction_step=3, context_steps=4)


def test_h014_configs_freeze_matched_sources() -> None:
    response = H014RunConfig.from_yaml(
        "configs/meta_world/world_h014_development_response.yaml"
    )
    control = H014RunConfig.from_yaml(
        "configs/meta_world/world_h014_development_control.yaml"
    )
    assert response.arm is H014Arm.RESPONSE
    assert control.arm is H014Arm.CONTROL
    assert response.paired_runtime.runtime.training.seed == 260946
    suite = H014SuiteConfig.from_yaml("configs/meta_world/world_h014_suite.yaml")
    assert set(suite.arms) == set(H014Arm)


def test_h014_models_are_parameter_matched_and_state_identical() -> None:
    window = _window()
    response = ResponseConditionedEffectWorldModel(
        _model_config(),
        response_source=ResponseSource.NO_OP_SUBTRACTED,
    ).eval()
    control = ResponseConditionedEffectWorldModel(
        _model_config(),
        response_source=ResponseSource.FACTUAL_RESIDUAL,
    ).eval()
    control.load_state_dict(response.state_dict(), strict=True)
    assert sum(parameter.numel() for parameter in response.parameters()) == sum(
        parameter.numel() for parameter in control.parameters()
    )
    with torch.no_grad():
        response_output = response(window)
        control_output = control(window)
    assert torch.equal(
        response_output.next_state_mean,
        control_output.next_state_mean,
    )
    assert torch.equal(
        response_output.counterfactual_no_op_state_mean,
        control_output.counterfactual_no_op_state_mean,
    )
    assert not torch.equal(response_output.effect_mean, control_output.effect_mean)


def test_h014_outcome_identity_is_exact() -> None:
    window = _window()
    model = ResponseConditionedEffectWorldModel(
        _model_config(),
        response_source=ResponseSource.NO_OP_SUBTRACTED,
    ).eval()
    with torch.no_grad():
        output = model(window)
    assert output.counterfactual_no_op_mean is not None
    residual = (
        output.effect_mean[:, :1]
        - output.effect_mean[:, 3:4]
        - output.counterfactual_no_op_mean
    )
    assert float(residual.abs().max()) <= 1e-6


def test_h014_preflight_routes_registered_response_source(
    tmp_path: Path,
    monkeypatch: object,
) -> None:
    captured: dict[str, object] = {}

    def fake_execute(
        config_path: object,
        output_dir: object,
        **kwargs: object,
    ) -> dict[str, object]:
        captured.update(kwargs)
        model_factory = kwargs["model_factory"]
        model = model_factory(kwargs["run_config"])  # type: ignore[operator]
        assert isinstance(model, ResponseConditionedEffectWorldModel)
        return {
            "run_id": "smoke",
            "status": "completed_preflight",
            "arm": kwargs["reported_arm"],
            "response_source": kwargs["result_metadata"]["response_source"],  # type: ignore[index]
        }

    monkeypatch.setattr(  # type: ignore[attr-defined]
        h014_preflight_module,
        "execute_paired_transition_preflight",
        fake_execute,
    )
    result = h014_preflight_module.run_h014_preflight(
        "configs/meta_world/world_h014_development_smoke.yaml",
        tmp_path / "run",
    )
    assert result["response_source"] == "predicted_factual_minus_predicted_no_op"
    assert captured["selection_metrics"] == ("intervention_effect_nrmse",)


def test_h014_suite_applies_matched_gate_without_revalidating_wg4(
    tmp_path: Path,
    monkeypatch: object,
) -> None:
    def fake_preflight(config_path: object, output_dir: object) -> dict[str, object]:
        response = "response" in str(config_path)
        output = Path(str(output_dir))
        output.mkdir(parents=True)
        (output / "checkpoint_manifest.json").write_text(
            json.dumps(
                {
                    "checkpoint_sha256": "a" * 64 if response else "b" * 64,
                    "weights_kind": "ema",
                }
            ),
            encoding="utf-8",
        )
        return {
            "run_id": "response" if response else "control",
            "response_source": (
                "predicted_factual_minus_predicted_no_op"
                if response
                else "predicted_factual_minus_final_observation"
            ),
            "best_step": 600,
            "model_class": "matched",
            "parameters": 100,
            "best_validation": {
                "intervention_effect_nrmse": 0.8 if response else 1.0,
                "four_step_rollout_nrmse": 1.0,
                "intervention_state_delta_nrmse": 1.0,
                "no_op_state_nrmse": 1.0,
                "intervention_effect_90_coverage": 0.9,
                "outcome_counterfactual_identity_maximum_absolute_residual": 0.0,
            },
            "runtime_seconds": 1.0,
            "peak_memory_bytes": 1,
            "environment": {"device": "cuda"},
        }

    monkeypatch.setattr(  # type: ignore[attr-defined]
        h014_suite_module,
        "run_h014_preflight",
        fake_preflight,
    )
    monkeypatch.setattr(  # type: ignore[attr-defined]
        h014_suite_module,
        "evaluate_legal_random_interventions",
        lambda config: SimpleNamespace(to_dict=lambda: {"legal_action_rate": 1.0}),
    )
    report_path = tmp_path / "reports" / "development.json"
    report = run_h014_development_suite(
        "configs/meta_world/world_h014_suite.yaml",
        tmp_path / "runs",
        report_path,
    )
    assert report["development_gate"]["passed"]
    assert report["dataset_integrity"]["revalidated"] is False
    assert json.loads(report_path.read_text(encoding="utf-8"))["test_metrics_opened"] is False
