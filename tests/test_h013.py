from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from chimera.meta_world.config import MetaWorldModelConfig
from chimera.meta_world.generators import (
    GeneratedWorldDatasetConfig,
    MechanismGenerator,
    SplitName,
    WorldAction,
    WorldFamily,
    WorldGenerationPipeline,
    WorldGenerator,
)
from chimera.meta_world.h002.windows import (
    make_transition_window,
    materialize_sequence_sample,
)
from chimera.meta_world.h013.config import H013Arm, H013RunConfig
from chimera.meta_world.h013.dataset import build_h013_smoke_dataset
from chimera.meta_world.h013.model import (
    DirectDualTransitionWorldModel,
    FactorizedCounterfactualTransitionWorldModel,
)
from chimera.meta_world.h013.objectives import h013_loss
from chimera.meta_world.h013.suite import H013SuiteConfig

GENERATOR_CONFIG = Path("configs/meta_world/world_generators_h013.yaml")


def _small_model_config() -> MetaWorldModelConfig:
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


def _window() -> tuple[object, object]:
    generator = GeneratedWorldDatasetConfig.from_yaml(GENERATOR_CONFIG)
    sample = materialize_sequence_sample(
        WorldGenerationPipeline(generator),
        SplitName.VALIDATION,
        start_index=0,
        batch_size=4,
    )
    return sample, make_transition_window(sample, prediction_step=3, context_steps=4)


def test_h013_configs_freeze_registered_contract() -> None:
    generator = GeneratedWorldDatasetConfig.from_yaml(GENERATOR_CONFIG)
    assert generator.schema_version == 3
    assert generator.paired_counterfactual_transitions
    assert generator.shared_external_event
    assert generator.shared_renderer_noise
    assert H013RunConfig.from_yaml(
        "configs/meta_world/world_h013_development_factorized.yaml"
    ).arm is H013Arm.FACTORIZED
    suite = H013SuiteConfig.from_yaml("configs/meta_world/world_h013_suite.yaml")
    assert set(suite.arms) == set(H013Arm)
    assert suite.test_access == "sealed"


def test_zero_action_proves_shared_event_and_renderer_noise() -> None:
    config = GeneratedWorldDatasetConfig.from_yaml(GENERATOR_CONFIG)
    metadata = WorldGenerationPipeline(config).materialize(SplitName.VALIDATION, 0).metadata
    mechanism = MechanismGenerator().generate(
        metadata.mechanism_template_id,
        metadata.mechanism_seed,
    )
    world = WorldGenerator(
        min_objects=config.min_objects,
        max_objects=config.max_objects,
        observation_features=config.observation_features,
        relation_features=config.relation_features,
    ).generate(
        mechanism,
        WorldFamily(metadata.world_family_id),
        world_seed=metadata.world_seed,
        renderer_seed=metadata.renderer_seed,
        renderer_profile=metadata.renderer_profile_id,
        independent_renderer_rng=True,
    )
    world.reset(metadata.generation_seed)
    transition = world.step(WorldAction(source=0, target=1, magnitude=0.0, control=0.0))
    assert transition.counterfactual_no_op_observation is not None
    assert np.array_equal(
        transition.observation.values,
        transition.counterfactual_no_op_observation.values,
    )


def test_wg4_window_carries_target_but_models_do_not_read_it() -> None:
    _, window = _window()
    assert window.counterfactual_no_op_observations is not None
    altered = torch.full_like(window.counterfactual_no_op_observations, 1000.0)
    factorized = FactorizedCounterfactualTransitionWorldModel(
        _small_model_config()
    ).eval()
    with torch.no_grad():
        first = factorized(window)
        second = factorized(
            window.__class__(
                **{
                    **window.__dict__,
                    "counterfactual_no_op_observations": altered,
                }
            )
        )
    assert torch.equal(first.next_state_mean, second.next_state_mean)


def test_factorized_identity_and_parameter_matching() -> None:
    _, window = _window()
    direct = DirectDualTransitionWorldModel(_small_model_config()).eval()
    factorized = FactorizedCounterfactualTransitionWorldModel(
        _small_model_config()
    ).eval()
    factorized.load_state_dict(direct.state_dict(), strict=True)
    assert sum(parameter.numel() for parameter in direct.parameters()) == sum(
        parameter.numel() for parameter in factorized.parameters()
    )
    with torch.no_grad():
        output = factorized(window)
    assert output.counterfactual_no_op_state_mean is not None
    assert output.intervention_state_delta_mean is not None
    residual = (
        output.next_state_mean
        - output.counterfactual_no_op_state_mean
        - output.intervention_state_delta_mean
    )
    assert float(residual.abs().max()) <= 1e-6


def test_h013_auxiliary_loss_is_finite() -> None:
    _, window = _window()
    model = FactorizedCounterfactualTransitionWorldModel(_small_model_config())
    output = model(window)
    runtime = H013RunConfig.from_yaml(
        "configs/meta_world/world_h013_development_factorized.yaml"
    )
    losses = h013_loss(
        output,
        window,
        runtime.runtime.training,
        no_op_state_weight=1.0,
        intervention_delta_weight=1.0,
    )
    assert torch.isfinite(torch.stack(list(losses.values()))).all()
    assert losses["no_op_state_loss"].requires_grad
    assert losses["intervention_delta_loss"].requires_grad


def test_h013_smoke_dataset_is_replay_exact(tmp_path: Path) -> None:
    report_path = tmp_path / "integrity.json"
    report = build_h013_smoke_dataset(
        GENERATOR_CONFIG,
        tmp_path / "dataset",
        report_path,
        trajectories_per_split=16,
    )
    assert report["status"] == "passed"
    assert report["checks"]["deterministic_replay"]
    assert report["checks"]["paired_counterfactual_transition_present"]
    persisted = json.loads(report_path.read_text(encoding="utf-8"))
    assert persisted["human_or_llm_judging"] is False
