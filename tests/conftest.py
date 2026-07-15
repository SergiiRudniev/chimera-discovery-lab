from __future__ import annotations

import pytest

from chimera.config import ModelConfig, TrainingConfig


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
