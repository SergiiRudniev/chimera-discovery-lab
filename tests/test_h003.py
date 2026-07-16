from __future__ import annotations

import json
from pathlib import Path

import torch
import yaml

from chimera.meta_world.config import MetaWorldModelConfig, MetaWorldTrainingConfig
from chimera.meta_world.generators import (
    GeneratedWorldDatasetConfig,
    SplitName,
    WorldGenerationPipeline,
)
from chimera.meta_world.h002 import (
    RelationalSequenceWorldModel,
    make_transition_window,
    materialize_sequence_sample,
)
from chimera.meta_world.h003 import (
    H003Trainer,
    MechanismMemoryQueue,
    h003_closed_loop_loss,
    run_h003_preflight,
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
        seed=260903,
        batch_size=4,
        active_slots=4,
        steps=2,
        learning_rate=1e-3,
        weight_decay=0.0,
        next_state_weight=1.0,
        effect_weight=0.5,
        alignment_weight=0.2,
        variance_weight=0.01,
        alignment_margin=0.2,
        primary_effect_weight=2.0,
        ema_decay=0.9,
        device="cpu",
        precision="float32",
    )


def _pipeline() -> WorldGenerationPipeline:
    config = GeneratedWorldDatasetConfig.from_yaml(
        "configs/meta_world/world_generators_h002.yaml"
    )
    return WorldGenerationPipeline(config)


def test_mechanism_keys_are_stable_and_globally_distinct() -> None:
    pipeline = _pipeline()
    first = materialize_sequence_sample(
        pipeline,
        SplitName.TRAIN,
        start_index=0,
        batch_size=4,
    )
    second = materialize_sequence_sample(
        pipeline,
        SplitName.TRAIN,
        start_index=4,
        batch_size=4,
    )

    assert first.mechanism_ids.tolist() == second.mechanism_ids.tolist()
    assert first.mechanism_keys[0] == first.mechanism_keys[1]
    assert first.mechanism_keys[2] == first.mechanism_keys[3]
    assert set(first.mechanism_keys.tolist()).isdisjoint(second.mechanism_keys.tolist())


def test_mechanism_queue_is_detached_bounded_and_warmed() -> None:
    queue = MechanismMemoryQueue(minimum_entries=4, maximum_entries=6)
    embeddings = torch.randn(4, 8, requires_grad=True)
    keys = torch.arange(4)

    assert queue.candidates() == (None, None)
    queue.update(embeddings, keys)
    candidates, candidate_keys = queue.candidates()

    assert candidates is not None and candidate_keys is not None
    assert candidates.requires_grad is False
    assert queue.size == 4
    queue.update(torch.randn(4, 8), torch.arange(4, 8))
    _, candidate_keys = queue.candidates()
    assert queue.size == 6
    assert candidate_keys is not None
    assert candidate_keys.tolist() == [2, 3, 4, 5, 6, 7]


def test_h003_closed_loop_step_is_finite_and_populates_queue() -> None:
    pipeline = _pipeline()
    sample = materialize_sequence_sample(
        pipeline,
        SplitName.TRAIN,
        start_index=0,
        batch_size=4,
    )
    trainer = H003Trainer(
        RelationalSequenceWorldModel(_model_config()),
        _training_config(),
        rollout_horizon=4,
        state_features=4,
        queue_minimum_entries=4,
        queue_maximum_entries=8,
    )

    metrics = trainer.train_sequence_step(
        sample,
        prediction_step=3,
        context_steps=4,
    )

    assert all(torch.isfinite(torch.tensor(value)) for value in metrics.values())
    assert metrics["mechanism_queue_entries"] == 4.0
    assert trainer.queue.size == 4


def test_h003_closed_loop_accepts_trainer_only_effect_route() -> None:
    sample = materialize_sequence_sample(
        _pipeline(),
        SplitName.TRAIN,
        start_index=0,
        batch_size=4,
    )
    trainer = H003Trainer(
        RelationalSequenceWorldModel(_model_config()),
        _training_config(),
        rollout_horizon=4,
        state_features=4,
        queue_minimum_entries=4,
        queue_maximum_entries=8,
    )

    metrics = trainer.train_sequence_step(
        sample,
        prediction_step=3,
        context_steps=4,
        effect_supervision_mask=torch.tensor([False, False, True, True]),
    )

    assert metrics["effect_supervision_fraction"] == 0.5
    assert not hasattr(sample.batch, "effect_supervision_mask")


def test_effect_route_removes_probe_examples_from_effect_gradient() -> None:
    sample = materialize_sequence_sample(
        _pipeline(),
        SplitName.TRAIN,
        start_index=0,
        batch_size=4,
    )
    window = make_transition_window(sample, prediction_step=3, context_steps=4)
    output = RelationalSequenceWorldModel(_model_config())(window)
    output.effect_mean.retain_grad()

    losses, _ = h003_closed_loop_loss(
        [output],
        [window],
        sample.mechanism_keys,
        _training_config(),
        MechanismMemoryQueue(minimum_entries=4, maximum_entries=8),
        torch.tensor([False, False, True, True]),
    )
    losses["loss"].backward()

    assert output.effect_mean.grad is not None
    assert torch.count_nonzero(output.effect_mean.grad[:2]) == 0
    assert torch.count_nonzero(output.effect_mean.grad[2:]) > 0


def test_h003_preflight_keeps_test_splits_sealed(tmp_path: Path) -> None:
    payload = {
        "run_id": "H003-PREFLIGHT-TEST",
        "mode": "preflight",
        "arm": "closed_loop_with_cross_batch_mechanism_discrimination",
        "generator_config": "configs/meta_world/world_generators_h002.yaml",
        "model": _model_config().__dict__,
        "training": _training_config().__dict__,
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
    }
    config_path = tmp_path / "h003.yaml"
    config_path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    output = tmp_path / "preflight"

    result = run_h003_preflight(config_path, output)
    manifest = json.loads(
        (output / "checkpoint_manifest.json").read_text(encoding="utf-8")
    )

    assert result["hypothesis_id"] == "CHM-W-H003"
    assert result["status"] == "completed_validation_preflight"
    assert result["opened_splits"] == ["train", "validation"]
    assert result["test_metrics_opened"] is False
    assert manifest["promoted"] is False
    assert manifest["opened_splits"] == ["train", "validation"]
