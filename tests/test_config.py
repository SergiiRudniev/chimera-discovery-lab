from __future__ import annotations

from pathlib import Path

import pytest

from chimera.config import (
    ExperimentConfig,
    ModelConfig,
    VentureTrialConfig,
)


def test_registered_config_loads() -> None:
    config = ExperimentConfig.from_yaml("configs/venture/venture_m0_20m.yaml")
    assert config.experiment_id == "CHM-V-H001"
    assert config.model.hidden_dim == 384
    assert config.model.max_nodes == 64


def test_unknown_config_field_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "bad.yaml"
    path.write_text("experiment_id: CHM-V-H999\nmodel:\n  surprise: 1\n", encoding="utf-8")
    with pytest.raises(ValueError, match="Unknown ModelConfig"):
        ExperimentConfig.from_yaml(path)


def test_hidden_dimension_must_match_heads() -> None:
    with pytest.raises(ValueError, match="divisible"):
        ModelConfig(hidden_dim=31, num_heads=4)


def test_registered_trial_config_loads() -> None:
    config = VentureTrialConfig.from_yaml("configs/venture/venture_trial_t0.yaml")
    assert config.trial_id == "CHM-V-T000"
    assert config.hypothesis_id == "CHM-V-H001"
    assert config.evaluation.archive_bins == (4, 4)


def test_corrective_trial_config_loads() -> None:
    config = VentureTrialConfig.from_yaml("configs/venture/venture_trial_t1.yaml")
    assert config.trial_id == "CHM-V-T001"
    assert config.training.argument_loss_mode == "operation_conditioned"
    assert config.training.learning_rate_schedule == "cosine"
    assert config.evaluation.checkpoint_selection == "validation_exact_graph"
