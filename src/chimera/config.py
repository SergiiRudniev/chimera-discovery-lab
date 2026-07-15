"""Validated configuration objects for Chimera models and experiments."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any

import yaml


def _validate_fields(
    class_name: str, allowed_fields: set[str], values: Mapping[str, Any]
) -> None:
    unknown = sorted(set(values) - allowed_fields)
    if unknown:
        raise ValueError(f"Unknown {class_name} fields: {', '.join(unknown)}")


@dataclass(frozen=True)
class ModelConfig:
    """Architecture contract for one Chimera Venture model."""

    node_types: int = 12
    edge_types: int = 16
    edit_operations: int = 9
    node_numeric_features: int = 8
    score_dimensions: int = 3
    hidden_dim: int = 384
    num_heads: int = 8
    encoder_layers: int = 5
    decoder_layers: int = 3
    transition_layers: int = 3
    feedforward_multiplier: int = 4
    max_nodes: int = 64
    max_edits: int = 8
    dropout: float = 0.1

    def __post_init__(self) -> None:
        positive = {
            "node_types": self.node_types,
            "edge_types": self.edge_types,
            "edit_operations": self.edit_operations,
            "node_numeric_features": self.node_numeric_features,
            "score_dimensions": self.score_dimensions,
            "hidden_dim": self.hidden_dim,
            "num_heads": self.num_heads,
            "encoder_layers": self.encoder_layers,
            "decoder_layers": self.decoder_layers,
            "transition_layers": self.transition_layers,
            "feedforward_multiplier": self.feedforward_multiplier,
            "max_nodes": self.max_nodes,
            "max_edits": self.max_edits,
        }
        invalid = [name for name, value in positive.items() if value <= 0]
        if invalid:
            raise ValueError(f"ModelConfig values must be positive: {', '.join(invalid)}")
        if self.hidden_dim % self.num_heads:
            raise ValueError("hidden_dim must be divisible by num_heads")
        if self.score_dimensions != 3:
            raise ValueError("Venture M0 defines exactly three score dimensions")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")

    @classmethod
    def from_mapping(cls, values: Mapping[str, Any]) -> ModelConfig:
        _validate_fields("ModelConfig", {item.name for item in fields(ModelConfig)}, values)
        return cls(**values)


@dataclass(frozen=True)
class TrainingConfig:
    """Optimization and reproducibility settings."""

    seed: int = 1701
    batch_size: int = 8
    steps: int = 200
    learning_rate: float = 3e-4
    weight_decay: float = 1e-2
    max_grad_norm: float = 1.0
    target_ema_decay: float = 0.99
    device: str = "auto"

    def __post_init__(self) -> None:
        if self.batch_size <= 0 or self.steps <= 0:
            raise ValueError("batch_size and steps must be positive")
        if self.learning_rate <= 0 or self.weight_decay < 0:
            raise ValueError("learning_rate must be positive and weight_decay non-negative")
        if self.max_grad_norm <= 0:
            raise ValueError("max_grad_norm must be positive")
        if not 0.0 <= self.target_ema_decay < 1.0:
            raise ValueError("target_ema_decay must be in [0, 1)")
        if self.device not in {"auto", "cpu", "cuda", "mps"}:
            raise ValueError("device must be auto, cpu, cuda or mps")

    @classmethod
    def from_mapping(cls, values: Mapping[str, Any]) -> TrainingConfig:
        _validate_fields(
            "TrainingConfig", {item.name for item in fields(TrainingConfig)}, values
        )
        return cls(**values)


@dataclass(frozen=True)
class ExperimentConfig:
    """Complete reproducible experiment configuration."""

    experiment_id: str
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)

    def __post_init__(self) -> None:
        if not self.experiment_id.startswith("CHM-V-H"):
            raise ValueError("Venture experiment IDs must use CHM-V-H###")

    @classmethod
    def from_mapping(cls, values: Mapping[str, Any]) -> ExperimentConfig:
        unknown = sorted(set(values) - {"experiment_id", "model", "training"})
        if unknown:
            raise ValueError(f"Unknown ExperimentConfig fields: {', '.join(unknown)}")
        if "experiment_id" not in values:
            raise ValueError("experiment_id is required")
        model_values = values.get("model", {})
        training_values = values.get("training", {})
        if not isinstance(model_values, Mapping) or not isinstance(training_values, Mapping):
            raise TypeError("model and training must be mappings")
        return cls(
            experiment_id=str(values["experiment_id"]),
            model=ModelConfig.from_mapping(model_values),
            training=TrainingConfig.from_mapping(training_values),
        )

    @classmethod
    def from_yaml(cls, path: str | Path) -> ExperimentConfig:
        with Path(path).open("r", encoding="utf-8") as handle:
            values = yaml.safe_load(handle)
        if not isinstance(values, Mapping):
            raise TypeError("experiment config must contain a mapping")
        return cls.from_mapping(values)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
