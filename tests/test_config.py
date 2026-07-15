from __future__ import annotations

from pathlib import Path

import pytest

from chimera.config import ExperimentConfig, ModelConfig


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
