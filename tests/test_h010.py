from __future__ import annotations

import json
from pathlib import Path

from chimera.meta_world.generators import (
    GeneratedWorldDatasetConfig,
    SplitName,
    WorldGenerationPipeline,
)
from chimera.meta_world.h002.model import RelationalSequenceWorldModel
from chimera.meta_world.h002.windows import (
    make_transition_window,
    materialize_sequence_sample,
)
from chimera.meta_world.h010 import (
    H010ModelVariant,
    H010RunConfig,
    SharedBottleneckRelationalWorldModel,
    run_h010_preflight,
)
from chimera.meta_world.h010.evaluation import projection_prediction_delta

CONFIG = Path("configs/meta_world/world_h010_development_smoke.yaml")


def _window() -> tuple[H010RunConfig, object]:
    config = H010RunConfig.from_yaml(CONFIG)
    generator = GeneratedWorldDatasetConfig.from_yaml(
        config.common.generator_config
    )
    sample = materialize_sequence_sample(
        WorldGenerationPipeline(generator),
        SplitName.TRAIN,
        start_index=0,
        batch_size=4,
    )
    return config, make_transition_window(
        sample,
        prediction_step=2,
        context_steps=config.common.model.context_steps,
    )


def test_h010_config_round_trips_and_keeps_registered_variant() -> None:
    config = H010RunConfig.from_yaml(CONFIG)

    assert config.model_variant is H010ModelVariant.SHARED
    assert H010RunConfig.from_mapping(config.to_dict()) == config
    assert config.common.training.seed == 260930


def test_shared_bottleneck_is_parameter_matched_and_prediction_coupled() -> None:
    config, window = _window()
    separate = RelationalSequenceWorldModel(config.common.model)
    shared = SharedBottleneckRelationalWorldModel(config.common.model)
    shared.load_state_dict(separate.state_dict())

    separate_count = sum(parameter.numel() for parameter in separate.parameters())
    shared_count = sum(parameter.numel() for parameter in shared.parameters())
    separate_delta = projection_prediction_delta(separate, window)  # type: ignore[arg-type]
    shared_delta = projection_prediction_delta(shared, window)  # type: ignore[arg-type]

    assert separate_count == shared_count
    assert separate_delta == 0.0
    assert shared_delta > 1e-6


def test_h010_smoke_preflight_keeps_test_sealed(tmp_path: Path) -> None:
    result = run_h010_preflight(CONFIG, tmp_path)
    persisted = json.loads((tmp_path / "result.json").read_text(encoding="utf-8"))

    assert result["hypothesis_id"] == "CHM-W-H010"
    assert result["model_variant"] == "shared_aligned_bottleneck"
    assert result["projection_prediction_delta"] > 1e-6
    assert result["status"] == "completed_preflight"
    assert result["opened_splits"] == ["train", "validation"]
    assert result["test_metrics_opened"] is False
    assert persisted == result
