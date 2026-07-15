from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest
import torch
import yaml

from chimera.meta_world.config import (
    MetaWorldExperimentConfig,
    MetaWorldModelConfig,
    MetaWorldTrainingConfig,
)
from chimera.meta_world.model import ChimeraMetaWorld
from chimera.meta_world.synthetic import make_mechanistic_batch
from chimera.meta_world.trainer import MetaWorldTrainer
from chimera.meta_world.trial import run_meta_world_trial


def test_registered_meta_world_config_loads() -> None:
    config = MetaWorldExperimentConfig.from_yaml("configs/meta_world/meta_world_w0.yaml")
    assert config.experiment_id == "CHM-W-H000"
    assert config.trial_id == "CHM-W-T000"
    assert config.model.hidden_dim == 512
    assert config.training.device == "cuda"


def test_mechanistic_batch_is_deterministic(
    small_meta_world_model_config: MetaWorldModelConfig,
) -> None:
    first = make_mechanistic_batch(
        small_meta_world_model_config,
        batch_size=4,
        active_slots=4,
        seed=19,
    )
    second = make_mechanistic_batch(
        small_meta_world_model_config,
        batch_size=4,
        active_slots=4,
        seed=19,
    )
    assert torch.equal(first.observations, second.observations)
    assert torch.equal(first.relations, second.relations)
    assert torch.equal(first.next_observations, second.next_observations)
    assert torch.equal(first.effect_targets, second.effect_targets)


def test_batch_rejects_reactivated_time_steps(
    small_meta_world_model_config: MetaWorldModelConfig,
) -> None:
    batch = make_mechanistic_batch(
        small_meta_world_model_config,
        batch_size=4,
        active_slots=4,
        seed=21,
    )
    invalid_time_mask = batch.time_mask.clone()
    invalid_time_mask[0] = torch.tensor([True, False, True])
    invalid = replace(batch, time_mask=invalid_time_mask)
    with pytest.raises(ValueError, match="contiguous"):
        invalid.validate()


def test_meta_world_output_shapes(
    small_meta_world_model_config: MetaWorldModelConfig,
) -> None:
    batch = make_mechanistic_batch(
        small_meta_world_model_config,
        batch_size=4,
        active_slots=4,
        seed=23,
    )
    output = ChimeraMetaWorld(small_meta_world_model_config).eval()(batch)
    assert output.next_state_mean.shape == (4, 4, 6)
    assert output.next_state_log_variance.shape == (4, 4, 6)
    assert output.effect_mean.shape == (4, 4)
    assert output.proposal_embedding.shape == (4, 32)
    assert torch.isfinite(output.next_state_mean).all()


def test_meta_world_forward_supports_bfloat16_autocast(
    small_meta_world_model_config: MetaWorldModelConfig,
) -> None:
    batch = make_mechanistic_batch(
        small_meta_world_model_config,
        batch_size=4,
        active_slots=4,
        seed=27,
    )
    model = ChimeraMetaWorld(small_meta_world_model_config).eval()
    with torch.no_grad(), torch.autocast(device_type="cpu", dtype=torch.bfloat16):
        output = model(batch)
    assert torch.isfinite(output.next_state_mean).all()
    assert torch.isfinite(output.effect_mean).all()


def test_meta_world_is_slot_permutation_equivariant(
    small_meta_world_model_config: MetaWorldModelConfig,
) -> None:
    batch = make_mechanistic_batch(
        small_meta_world_model_config,
        batch_size=4,
        active_slots=4,
        seed=29,
    )
    permutation = torch.tensor([2, 0, 3, 1])
    inverse = torch.argsort(permutation)
    permuted = replace(
        batch,
        observations=batch.observations[:, :, permutation],
        observation_mask=batch.observation_mask[:, :, permutation],
        slot_mask=batch.slot_mask[:, :, permutation],
        relations=batch.relations[:, :, permutation][:, :, :, permutation],
        source_slots=inverse[batch.source_slots],
        target_slots=inverse[batch.target_slots],
        next_observations=batch.next_observations[:, permutation],
        next_observation_mask=batch.next_observation_mask[:, permutation],
    )
    model = ChimeraMetaWorld(small_meta_world_model_config).eval()
    with torch.no_grad():
        original_output = model(batch)
        permuted_output = model(permuted)
    torch.testing.assert_close(
        permuted_output.next_state_mean,
        original_output.next_state_mean[:, permutation],
        rtol=1e-5,
        atol=1e-6,
    )
    torch.testing.assert_close(
        permuted_output.effect_mean,
        original_output.effect_mean,
        rtol=1e-5,
        atol=1e-6,
    )


def test_meta_world_train_step_is_finite(
    small_meta_world_model_config: MetaWorldModelConfig,
    small_meta_world_training_config: MetaWorldTrainingConfig,
) -> None:
    batch = make_mechanistic_batch(
        small_meta_world_model_config,
        batch_size=4,
        active_slots=4,
        seed=31,
    )
    trainer = MetaWorldTrainer(
        ChimeraMetaWorld(small_meta_world_model_config),
        small_meta_world_training_config,
    )
    metrics = trainer.train_step(batch)
    assert set(metrics) == {
        "loss",
        "next_state_loss",
        "effect_loss",
        "alignment_loss",
        "variance_loss",
        "gradient_norm",
    }
    assert all(torch.isfinite(torch.tensor(value)) for value in metrics.values())


def test_meta_world_trial_writes_auditable_artifacts(
    tmp_path: Path,
    small_meta_world_model_config: MetaWorldModelConfig,
    small_meta_world_training_config: MetaWorldTrainingConfig,
) -> None:
    config_path = tmp_path / "config.yaml"
    output = tmp_path / "trial"
    public_result = tmp_path / "result.json"
    payload = {
        "experiment_id": "CHM-W-H999",
        "trial_id": "CHM-W-T999",
        "model": small_meta_world_model_config.__dict__,
        "training": small_meta_world_training_config.__dict__,
        "qualification": {
            "minimum_parameters": 1,
            "maximum_parameters": 10_000_000,
            "minimum_loss_reduction_fraction": 0.0,
            "maximum_replay_delta": 0.0,
            "require_cuda": False,
            "require_all_finite": True,
        },
    }
    config_path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    result = run_meta_world_trial(config_path, output, public_result)
    assert result["status"] == "completed"
    assert result["decision"] == "accepted"
    assert (output / "metrics.jsonl").is_file()
    assert (output / "environment.json").is_file()
    assert (output / "result.json").is_file()
    assert public_result.is_file()
