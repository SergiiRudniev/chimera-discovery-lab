"""Validated configuration contracts for Chimera Meta-World."""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any

import yaml


def _validated_kwargs(class_name: str, values: Mapping[str, Any], allowed: set[str]) -> None:
    unknown = sorted(set(values) - allowed)
    if unknown:
        raise ValueError(f"Unknown {class_name} fields: {', '.join(unknown)}")


@dataclass(frozen=True)
class MetaWorldModelConfig:
    """Executable W0 tensor and architecture contract."""

    observation_features: int = 12
    relation_features: int = 4
    intervention_types: int = 8
    intervention_parameters: int = 8
    effect_dimensions: int = 4
    domain_count: int = 4
    mechanism_count: int = 4
    hidden_dim: int = 512
    num_heads: int = 8
    spatial_layers: int = 8
    temporal_layers: int = 6
    transition_layers: int = 6
    feedforward_multiplier: int = 4
    max_slots: int = 32
    context_steps: int = 8
    dropout: float = 0.1
    log_variance_min: float = -6.0
    log_variance_max: float = 2.0

    def __post_init__(self) -> None:
        positive = {
            item.name: getattr(self, item.name)
            for item in fields(self)
            if item.name
            not in {
                "dropout",
                "log_variance_min",
                "log_variance_max",
            }
        }
        invalid = [name for name, value in positive.items() if value <= 0]
        if invalid:
            raise ValueError(f"MetaWorldModelConfig values must be positive: {', '.join(invalid)}")
        if self.hidden_dim % self.num_heads:
            raise ValueError("hidden_dim must be divisible by num_heads")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")
        if self.log_variance_min >= self.log_variance_max:
            raise ValueError("log_variance_min must be less than log_variance_max")

    @classmethod
    def from_mapping(cls, values: Mapping[str, Any]) -> MetaWorldModelConfig:
        _validated_kwargs(cls.__name__, values, {item.name for item in fields(cls)})
        return cls(**values)


@dataclass(frozen=True)
class MetaWorldTrainingConfig:
    """Optimization settings for one W0 engineering qualification."""

    seed: int = 260715
    batch_size: int = 8
    active_slots: int = 8
    steps: int = 20
    learning_rate: float = 2e-4
    weight_decay: float = 1e-2
    max_grad_norm: float = 1.0
    next_state_weight: float = 1.0
    effect_weight: float = 0.25
    alignment_weight: float = 0.1
    variance_weight: float = 0.01
    alignment_margin: float = 0.2
    device: str = "cuda"
    precision: str = "bfloat16"

    def __post_init__(self) -> None:
        if self.seed < 0:
            raise ValueError("seed must be non-negative")
        if self.batch_size <= 0 or self.active_slots <= 1 or self.steps <= 0:
            raise ValueError("batch_size, active_slots and steps must be positive")
        if self.learning_rate <= 0 or self.weight_decay < 0 or self.max_grad_norm <= 0:
            raise ValueError("invalid optimizer settings")
        for name in (
            "next_state_weight",
            "effect_weight",
            "alignment_weight",
            "variance_weight",
            "alignment_margin",
        ):
            if getattr(self, name) < 0:
                raise ValueError(f"{name} must be non-negative")
        if self.device not in {"auto", "cpu", "cuda", "mps"}:
            raise ValueError("device must be auto, cpu, cuda or mps")
        if self.precision not in {"float32", "bfloat16"}:
            raise ValueError("precision must be float32 or bfloat16")

    @classmethod
    def from_mapping(cls, values: Mapping[str, Any]) -> MetaWorldTrainingConfig:
        _validated_kwargs(cls.__name__, values, {item.name for item in fields(cls)})
        return cls(**values)


@dataclass(frozen=True)
class MetaWorldQualificationConfig:
    """Predeclared pass/fail gates for W0 engineering H000."""

    minimum_parameters: int = 50_000_000
    maximum_parameters: int = 80_000_000
    minimum_loss_reduction_fraction: float = 0.10
    maximum_replay_delta: float = 0.0
    require_cuda: bool = True
    require_all_finite: bool = True

    def __post_init__(self) -> None:
        if self.minimum_parameters <= 0 or self.maximum_parameters < self.minimum_parameters:
            raise ValueError("invalid parameter qualification range")
        if not 0.0 <= self.minimum_loss_reduction_fraction <= 1.0:
            raise ValueError("minimum_loss_reduction_fraction must be in [0, 1]")
        if self.maximum_replay_delta < 0:
            raise ValueError("maximum_replay_delta must be non-negative")

    @classmethod
    def from_mapping(cls, values: Mapping[str, Any]) -> MetaWorldQualificationConfig:
        _validated_kwargs(cls.__name__, values, {item.name for item in fields(cls)})
        return cls(**values)


@dataclass(frozen=True)
class MetaWorldExperimentConfig:
    """Complete W0 configuration loaded from one immutable YAML file."""

    experiment_id: str
    trial_id: str
    model: MetaWorldModelConfig = field(default_factory=MetaWorldModelConfig)
    training: MetaWorldTrainingConfig = field(default_factory=MetaWorldTrainingConfig)
    qualification: MetaWorldQualificationConfig = field(
        default_factory=MetaWorldQualificationConfig
    )

    def __post_init__(self) -> None:
        if re.fullmatch(r"CHM-W-H\d{3}", self.experiment_id) is None:
            raise ValueError("Meta-World experiment IDs must use CHM-W-H###")
        if re.fullmatch(r"CHM-W-T\d{3}", self.trial_id) is None:
            raise ValueError("Meta-World trial IDs must use CHM-W-T###")
        if self.training.active_slots > self.model.max_slots:
            raise ValueError("active_slots exceeds model max_slots")
        if self.training.batch_size < self.model.mechanism_count * 2:
            raise ValueError("batch_size must provide two domain views per mechanism")

    @classmethod
    def from_mapping(cls, values: Mapping[str, Any]) -> MetaWorldExperimentConfig:
        allowed = {"experiment_id", "trial_id", "model", "training", "qualification"}
        _validated_kwargs(cls.__name__, values, allowed)
        for required in ("experiment_id", "trial_id"):
            if required not in values:
                raise ValueError(f"{required} is required")
        nested = [values.get(name, {}) for name in ("model", "training", "qualification")]
        if not all(isinstance(item, Mapping) for item in nested):
            raise TypeError("model, training and qualification must be mappings")
        model_values, training_values, qualification_values = nested
        return cls(
            experiment_id=str(values["experiment_id"]),
            trial_id=str(values["trial_id"]),
            model=MetaWorldModelConfig.from_mapping(model_values),
            training=MetaWorldTrainingConfig.from_mapping(training_values),
            qualification=MetaWorldQualificationConfig.from_mapping(qualification_values),
        )

    @classmethod
    def from_yaml(cls, path: str | Path) -> MetaWorldExperimentConfig:
        with Path(path).open("r", encoding="utf-8") as handle:
            values = yaml.safe_load(handle)
        if not isinstance(values, Mapping):
            raise TypeError("Meta-World config must contain a mapping")
        return cls.from_mapping(values)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
