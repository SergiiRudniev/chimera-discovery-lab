from __future__ import annotations

import pytest

from chimera.config import ModelConfig, TrainingConfig
from chimera.meta_world.config import MetaWorldModelConfig, MetaWorldTrainingConfig


@pytest.fixture
def small_model_config() -> ModelConfig:
    return ModelConfig(
        hidden_dim=32,
        num_heads=4,
        encoder_layers=1,
        decoder_layers=1,
        transition_layers=1,
        feedforward_multiplier=2,
        max_nodes=8,
        max_edits=3,
        dropout=0.0,
    )


@pytest.fixture
def small_training_config() -> TrainingConfig:
    return TrainingConfig(
        seed=7,
        batch_size=2,
        steps=1,
        learning_rate=1e-3,
        weight_decay=0.0,
        target_ema_decay=0.9,
        device="cpu",
    )


@pytest.fixture
def small_meta_world_model_config() -> MetaWorldModelConfig:
    return MetaWorldModelConfig(
        observation_features=6,
        relation_features=4,
        intervention_types=8,
        intervention_parameters=4,
        effect_dimensions=4,
        domain_count=2,
        mechanism_count=2,
        hidden_dim=32,
        num_heads=4,
        spatial_layers=1,
        temporal_layers=1,
        transition_layers=1,
        feedforward_multiplier=2,
        max_slots=4,
        context_steps=3,
        dropout=0.0,
    )


@pytest.fixture
def small_meta_world_training_config() -> MetaWorldTrainingConfig:
    return MetaWorldTrainingConfig(
        seed=11,
        batch_size=4,
        active_slots=4,
        steps=1,
        learning_rate=1e-3,
        weight_decay=0.0,
        max_grad_norm=1.0,
        device="cpu",
        precision="float32",
    )
