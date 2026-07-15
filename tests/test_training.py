from __future__ import annotations

import math

import torch

from chimera.config import ModelConfig, TrainingConfig
from chimera.data.contracts import EditBatch
from chimera.data.synthetic import make_synthetic_batch
from chimera.models.venture import ChimeraVenture
from chimera.training.objectives import edit_argument_masks
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


def test_operation_conditioned_argument_masks() -> None:
    shape = (1, 4)
    zeros = torch.zeros(shape, dtype=torch.long)
    edits = EditBatch(
        operations=torch.tensor([[0, 2, 7, 6]], dtype=torch.long),
        source_nodes=zeros,
        target_nodes=zeros,
        node_types=zeros,
        edge_types=zeros,
        step_mask=torch.ones(shape, dtype=torch.bool),
    )
    masks = edit_argument_masks(edits)
    assert masks["source"].tolist() == [[False, True, True, True]]
    assert masks["target"].tolist() == [[False, True, False, True]]
    assert masks["node_type"].tolist() == [[False, False, True, False]]
    assert masks["edge_type"].tolist() == [[False, True, False, False]]


def test_cosine_schedule_reaches_registered_floor(
    small_model_config: ModelConfig,
) -> None:
    config = TrainingConfig(
        seed=7,
        batch_size=2,
        steps=4,
        learning_rate=1e-3,
        weight_decay=0.0,
        target_ema_decay=0.9,
        argument_loss_mode="operation_conditioned",
        learning_rate_schedule="cosine",
        warmup_steps=2,
        minimum_learning_rate=1e-4,
        device="cpu",
    )
    trainer = ChimeraTrainer(ChimeraVenture(small_model_config), config)
    batch = make_synthetic_batch(small_model_config, batch_size=2, seed=17)
    learning_rates = [trainer.train_step(batch)["learning_rate"] for _ in range(4)]
    assert learning_rates[0] == 5e-4
    assert learning_rates[1] == 1e-3
    assert learning_rates[-1] == 1e-4
