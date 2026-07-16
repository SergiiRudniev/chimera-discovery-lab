from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pytest
import torch

from chimera.meta_world.generators import (
    GeneratedWorldDatasetConfig,
    SplitName,
    WorldGenerationPipeline,
)
from chimera.meta_world.h002.model import RelationalSequenceWorldModel
from chimera.meta_world.h002.windows import materialize_sequence_sample
from chimera.meta_world.h011 import H011RunConfig, run_h011_preflight
from chimera.meta_world.h011.objectives import paired_response_consistency
from chimera.meta_world.h011.windows import make_paired_response_window
from chimera.meta_world.model import MetaWorldOutput

GENERATOR_CONFIG = Path("configs/meta_world/world_generators_h009.yaml")
SMOKE_CONFIG = Path("configs/meta_world/world_h011_development_smoke.yaml")


def _output(effect_mean: torch.Tensor, effect_log_variance: torch.Tensor) -> MetaWorldOutput:
    batch = effect_mean.shape[0]
    return MetaWorldOutput(
        next_state_mean=torch.zeros(batch, 2, 4),
        next_state_log_variance=torch.zeros(batch, 2, 4),
        effect_mean=effect_mean,
        effect_log_variance=effect_log_variance,
        proposal_embedding=torch.zeros(batch, 8),
        final_slot_states=torch.zeros(batch, 2, 8),
        transition_state=torch.zeros(batch, 8),
    )


def test_h011_config_freezes_matched_no_alignment_model() -> None:
    config = H011RunConfig.from_yaml(
        "configs/meta_world/world_h011_development_consistency.yaml"
    )
    control = H011RunConfig.from_yaml(
        "configs/meta_world/world_h011_development_control.yaml"
    )

    assert config.training_seed == control.training_seed == 260934
    assert config.response_consistency_weight == 1.0
    assert control.response_consistency_weight == 0.0
    assert config.common.model == control.common.model
    assert config.common.training == control.common.training
    assert config.common.training.alignment_weight == 0.0


def test_world_instance_keys_pair_renderer_views_only() -> None:
    generator = GeneratedWorldDatasetConfig.from_yaml(GENERATOR_CONFIG)
    sample = materialize_sequence_sample(
        WorldGenerationPipeline(generator),
        SplitName.TRAIN,
        start_index=0,
        batch_size=4,
    )
    window = make_paired_response_window(sample, 0, 4)

    assert sample.world_instance_keys[0] == sample.world_instance_keys[1]
    assert sample.world_instance_keys[2] == sample.world_instance_keys[3]
    assert sample.world_instance_keys[0] != sample.world_instance_keys[2]
    assert torch.equal(window.mechanism_ids, sample.world_instance_keys)
    assert not hasattr(sample.batch, "world_instance_keys")


def test_response_consistency_targets_primary_effect_distribution() -> None:
    effect_mean = torch.zeros(4, 4)
    effect_log_variance = torch.zeros(4, 4)
    effect_mean[:, -1] = torch.tensor([0.0, 2.0, 1.0, 1.0])
    effect_log_variance[:, -1] = torch.tensor([0.0, 2.0, 1.0, 1.0])
    pair_keys = torch.tensor([7, 7, 9, 9])

    inconsistent = paired_response_consistency(
        _output(effect_mean, effect_log_variance),
        pair_keys,
        uncertainty_fraction=0.1,
    )
    consistent = paired_response_consistency(
        _output(torch.zeros(4, 4), torch.zeros(4, 4)),
        pair_keys,
        uncertainty_fraction=0.1,
    )

    assert inconsistent["response_mean_consistency_loss"] > 0.0
    assert inconsistent["response_uncertainty_consistency_loss"] > 0.0
    assert consistent["response_consistency_loss"] == 0.0
    with pytest.raises(ValueError, match="pair_keys"):
        paired_response_consistency(
            _output(effect_mean, effect_log_variance),
            torch.zeros(4, 1),
            uncertainty_fraction=0.1,
        )


def test_pair_labels_do_not_change_model_forward() -> None:
    config = H011RunConfig.from_yaml(SMOKE_CONFIG)
    generator = GeneratedWorldDatasetConfig.from_yaml(GENERATOR_CONFIG)
    sample = materialize_sequence_sample(
        WorldGenerationPipeline(generator),
        SplitName.TRAIN,
        start_index=0,
        batch_size=4,
    )
    window = make_paired_response_window(sample, 0, config.common.model.context_steps)
    altered = replace(window, mechanism_ids=torch.arange(4, dtype=torch.long))
    torch.manual_seed(17)
    model = RelationalSequenceWorldModel(config.common.model).eval()

    with torch.no_grad():
        original_output = model(window)
        altered_output = model(altered)

    assert torch.equal(original_output.effect_mean, altered_output.effect_mean)
    assert torch.equal(
        original_output.effect_log_variance,
        altered_output.effect_log_variance,
    )


def test_h011_smoke_preflight_keeps_test_sealed(tmp_path: Path) -> None:
    result = run_h011_preflight(SMOKE_CONFIG, tmp_path)
    persisted = json.loads((tmp_path / "result.json").read_text(encoding="utf-8"))

    assert result == persisted
    assert result["hypothesis_id"] == "CHM-W-H011"
    assert result["status"] == "completed_preflight"
    assert result["opened_splits"] == ["train", "validation"]
    assert result["test_metrics_opened"] is False
    assert result["response_consistency_weight"] == 1.0
    assert result["paired_effect_mean_disagreement"] >= 0.0
    assert result["paired_effect_uncertainty_disagreement"] >= 0.0
    assert result["final_training"]["response_consistency_loss"] >= 0.0
    assert not (tmp_path / "test_world_transfer.npz").exists()
