from __future__ import annotations

import math

from chimera.config import ModelConfig, TrainingConfig
from chimera.data.synthetic import make_synthetic_batch
from chimera.models.venture import ChimeraVenture
from chimera.training.trainer import ChimeraTrainer


def test_training_step_is_finite(
    small_model_config: ModelConfig, small_training_config: TrainingConfig
) -> None:
    model = ChimeraVenture(small_model_config)
    trainer = ChimeraTrainer(model, small_training_config)
    batch = make_synthetic_batch(small_model_config, batch_size=2, seed=7)
    metrics = trainer.train_step(batch)
    assert trainer.step == 1
    assert all(math.isfinite(value) for value in metrics.values())
    assert metrics["loss"] > 0
