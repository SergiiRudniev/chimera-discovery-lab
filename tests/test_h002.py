from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pytest
import torch
import yaml

from chimera.meta_world.config import MetaWorldModelConfig, MetaWorldTrainingConfig
from chimera.meta_world.generators import (
    GeneratedWorldDatasetConfig,
    SplitName,
    WorldGenerationPipeline,
)
from chimera.meta_world.h002 import (
    H002Trainer,
    RelationalSequenceWorldModel,
    TemporalWorldBaseline,
    evaluate_h002_model,
    make_transition_window,
    materialize_sequence_sample,
    run_h002_preflight,
)
from chimera.meta_world.h002.objectives import _mechanism_alignment


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
        seed=260902,
        batch_size=4,
        active_slots=4,
        steps=1,
        learning_rate=1e-3,
        weight_decay=0.0,
        next_state_weight=1.0,
        effect_weight=0.5,
        alignment_weight=0.1,
        variance_weight=0.01,
        alignment_margin=0.2,
        primary_effect_weight=2.0,
        ema_decay=0.9,
        device="cpu",
        precision="float32",
    )


def _sample() -> tuple[WorldGenerationPipeline, object]:
    generator_config = GeneratedWorldDatasetConfig.from_yaml(
        "configs/meta_world/world_generators_h002.yaml"
    )
    pipeline = WorldGenerationPipeline(generator_config)
    return pipeline, materialize_sequence_sample(
        pipeline,
        SplitName.TRAIN,
        start_index=0,
        batch_size=4,
    )


def test_generated_sequence_window_hides_service_metadata() -> None:
    _, sample = _sample()
    window = make_transition_window(sample, prediction_step=2, context_steps=4)  # type: ignore[arg-type]

    assert sample.mechanism_ids.tolist() == [0, 0, 1, 1]  # type: ignore[attr-defined]
    assert window.observations.shape == (4, 4, 10, 8)
    assert window.intervention_parameters.shape == (4, 3)
    assert window.effect_targets.shape == (4, 4)
    assert window.action_history is not None
    assert window.action_target_history is not None
    assert window.action_history.shape == (4, 4, 3)
    assert window.action_target_history.shape == (4, 4, 10)
    assert torch.count_nonzero(window.action_history[:, 0]) == 0
    assert torch.equal(window.domain_ids, torch.zeros(4, dtype=torch.long))
    assert not hasattr(window, "world_family_ids")
    assert torch.all(window.source_slots != window.target_slots)


def test_temporal_baseline_does_not_read_relations() -> None:
    _, sample = _sample()
    window = make_transition_window(sample, prediction_step=3, context_steps=4)  # type: ignore[arg-type]
    model = TemporalWorldBaseline(_model_config()).eval()
    changed = replace(window, relations=torch.randn_like(window.relations))

    with torch.no_grad():
        original_output = model(window)
        changed_output = model(changed)

    assert torch.equal(original_output.next_state_mean, changed_output.next_state_mean)
    assert torch.equal(original_output.effect_mean, changed_output.effect_mean)


def test_h002_relational_train_step_and_evaluation_are_finite() -> None:
    _, sample = _sample()
    model = RelationalSequenceWorldModel(_model_config())
    trainer = H002Trainer(model, _training_config())
    window = make_transition_window(sample, prediction_step=2, context_steps=4)  # type: ignore[arg-type]

    metrics = trainer.train_step(window)
    evaluation = evaluate_h002_model(
        trainer,
        sample,  # type: ignore[arg-type]
        context_steps=4,
        rollout_horizon=4,
    )

    assert all(torch.isfinite(torch.tensor(value)) for value in metrics.values())
    assert evaluation.one_step_prediction_rmse >= 0.0
    assert evaluation.intervention_effect_rmse >= 0.0
    assert evaluation.intervention_effect_nrmse >= 0.0
    assert 0.0 <= evaluation.four_step_rollout_nrmse < 100.0
    assert 0.0 <= evaluation.intervention_effect_90_coverage <= 1.0
    assert 0.0 <= evaluation.mechanism_retrieval_accuracy <= 1.0


def test_h002_relational_model_is_slot_permutation_equivariant() -> None:
    _, sample = _sample()
    window = make_transition_window(sample, prediction_step=3, context_steps=4)  # type: ignore[arg-type]
    permutation = torch.tensor([2, 0, 3, 1, 4, 5, 6, 7, 8, 9])
    inverse = torch.argsort(permutation)
    permuted = replace(
        window,
        observations=window.observations[:, :, permutation],
        observation_mask=window.observation_mask[:, :, permutation],
        slot_mask=window.slot_mask[:, :, permutation],
        relations=window.relations[:, :, permutation][:, :, :, permutation],
        source_slots=inverse[window.source_slots],
        target_slots=inverse[window.target_slots],
        next_observations=window.next_observations[:, permutation],
        next_observation_mask=window.next_observation_mask[:, permutation],
        action_target_history=(
            window.action_target_history[:, :, permutation]
            if window.action_target_history is not None
            else None
        ),
    )
    model = RelationalSequenceWorldModel(_model_config()).eval()

    with torch.no_grad():
        original_output = model(window)
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


def test_h002_mechanism_embedding_is_action_independent() -> None:
    _, sample = _sample()
    window = make_transition_window(sample, prediction_step=3, context_steps=4)  # type: ignore[arg-type]
    changed = replace(
        window,
        source_slots=window.target_slots,
        target_slots=window.source_slots,
        intervention_parameters=torch.randn_like(window.intervention_parameters),
    )
    model = RelationalSequenceWorldModel(_model_config()).eval()

    with torch.no_grad():
        original_output = model(window)
        changed_output = model(changed)

    torch.testing.assert_close(
        original_output.proposal_embedding,
        changed_output.proposal_embedding,
    )
    assert not torch.equal(
        original_output.next_state_mean,
        changed_output.next_state_mean,
    )


def test_h002_mechanism_embedding_reads_only_past_actions() -> None:
    _, sample = _sample()
    window = make_transition_window(sample, prediction_step=3, context_steps=4)  # type: ignore[arg-type]
    assert window.action_history is not None
    changed = replace(
        window,
        action_history=torch.randn_like(window.action_history),
    )
    model = RelationalSequenceWorldModel(_model_config()).eval()

    with torch.no_grad():
        original_output = model(window)
        changed_output = model(changed)

    assert not torch.equal(
        original_output.proposal_embedding,
        changed_output.proposal_embedding,
    )


def test_h002_alignment_penalizes_collapsed_embeddings() -> None:
    mechanism_ids = torch.tensor([0, 0, 1, 1])
    collapsed = torch.ones(4, 2)
    separated = torch.tensor(
        [[1.0, 0.0], [1.0, 0.0], [-1.0, 0.0], [-1.0, 0.0]]
    )

    collapsed_loss = _mechanism_alignment(collapsed, mechanism_ids, margin=0.2)
    separated_loss = _mechanism_alignment(separated, mechanism_ids, margin=0.2)

    assert separated_loss < collapsed_loss


def test_h002_trainer_uses_ema_evaluation_weights() -> None:
    _, sample = _sample()
    model = RelationalSequenceWorldModel(_model_config())
    trainer = H002Trainer(model, _training_config())
    window = make_transition_window(sample, prediction_step=2, context_steps=4)  # type: ignore[arg-type]
    before = {
        name: value.clone() for name, value in trainer.evaluation_state_dict().items()
    }

    trainer.train_step(window)
    after = trainer.evaluation_state_dict()

    assert trainer.evaluation_weights_kind == "ema"
    assert any(not torch.equal(before[name], after[name]) for name in before)


def test_sequence_sampler_rejects_partial_alignment_groups() -> None:
    pipeline, _ = _sample()

    try:
        materialize_sequence_sample(
            pipeline,
            SplitName.TRAIN,
            start_index=1,
            batch_size=3,
        )
    except ValueError as error:
        assert "mechanism-view" in str(error)
    else:
        raise AssertionError("partial mechanism-view group was accepted")


def test_validation_only_preflight_persists_unpromoted_checkpoint(tmp_path: Path) -> None:
    model = _model_config()
    training = _training_config()
    payload = {
        "run_id": "H002-PREFLIGHT-TEST",
        "mode": "preflight",
        "arm": "cross_world_pretraining_with_mechanism_alignment",
        "generator_config": "configs/meta_world/world_generators_h002.yaml",
        "model": model.__dict__,
        "training": {**training.__dict__, "steps": 2},
        "evaluation": {
            "evaluation_interval": 1,
            "validation_trajectories": 4,
            "rollout_horizon": 4,
        },
    }
    config_path = tmp_path / "preflight.yaml"
    config_path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    output = tmp_path / "output"

    result = run_h002_preflight(config_path, output)

    assert result["status"] == "completed_preflight"
    assert result["opened_splits"] == ["train", "validation"]
    assert result["test_metrics_opened"] is False
    assert (output / "checkpoint.pt").is_file()
    assert (output / "checkpoint_manifest.json").is_file()
    assert (output / "metrics.jsonl").is_file()


def test_preflight_persists_uncaught_execution_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = _model_config()
    training = _training_config()
    payload = {
        "run_id": "H002-PREFLIGHT-FAILURE-TEST",
        "mode": "preflight",
        "arm": "cross_world_pretraining_with_mechanism_alignment",
        "generator_config": "configs/meta_world/world_generators_h002.yaml",
        "model": model.__dict__,
        "training": {**training.__dict__, "steps": 1},
        "evaluation": {
            "evaluation_interval": 1,
            "validation_trajectories": 4,
            "rollout_horizon": 4,
        },
    }
    config_path = tmp_path / "failure.yaml"
    config_path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    output = tmp_path / "failed-output"

    def fail_train_step(self: H002Trainer, batch: object) -> dict[str, float]:
        raise RuntimeError("injected H002 failure")

    monkeypatch.setattr(H002Trainer, "train_step", fail_train_step)
    with pytest.raises(RuntimeError, match="injected H002 failure"):
        run_h002_preflight(config_path, output)

    result = json.loads((output / "result.json").read_text(encoding="utf-8"))
    assert result["status"] == "execution_failed"
    assert result["exception"]["type"] == "RuntimeError"
    assert result["test_metrics_opened"] is False
