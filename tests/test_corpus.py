from __future__ import annotations

import math
from pathlib import Path

from chimera.config import ModelConfig, TrainingConfig
from chimera.data.corpus import CorpusSplit, build_corpus, validate_corpus
from chimera.models.venture import ChimeraVenture
from chimera.training.trainer import ChimeraTrainer


def test_committed_corpus_validates() -> None:
    result = validate_corpus("datasets/venture_corpus_c0/manifest.json")
    assert result == {"canonical_graphs": 10, "transitions": 640}


def test_corpus_build_is_deterministic(tmp_path: Path) -> None:
    first = tmp_path / "first"
    second = tmp_path / "second"
    first_manifest = build_corpus(
        "datasets/venture_corpus_c0/source_graphs.yaml",
        first,
        examples_per_case=2,
        seed=1701,
    )
    second_manifest = build_corpus(
        "datasets/venture_corpus_c0/source_graphs.yaml",
        second,
        examples_per_case=2,
        seed=1701,
    )
    assert first_manifest == second_manifest
    assert validate_corpus(first / "manifest.json") == {
        "canonical_graphs": 10,
        "transitions": 20,
    }


def test_corpus_batch_trains_without_language() -> None:
    shard = CorpusSplit("datasets/venture_corpus_c0/train.npz")
    batch = shard.batch([0, 1])
    model_config = ModelConfig(
        hidden_dim=32,
        num_heads=4,
        encoder_layers=1,
        decoder_layers=1,
        transition_layers=1,
        feedforward_multiplier=2,
        max_nodes=64,
        max_edits=8,
        dropout=0.0,
    )
    training_config = TrainingConfig(
        seed=11,
        batch_size=2,
        steps=1,
        learning_rate=1e-3,
        weight_decay=0.0,
        target_ema_decay=0.9,
        device="cpu",
    )
    trainer = ChimeraTrainer(ChimeraVenture(model_config), training_config)
    metrics = trainer.train_step(batch)
    assert all(math.isfinite(value) for value in metrics.values())
    assert batch.graph.node_features.shape == (2, 64, 8)
