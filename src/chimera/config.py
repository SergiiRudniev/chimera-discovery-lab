"""Validated configuration objects for Chimera models and experiments."""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any

import yaml


def _validate_fields(class_name: str, allowed_fields: set[str], values: Mapping[str, Any]) -> None:
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
    argument_loss_mode: str = "all_fields"
    learning_rate_schedule: str = "constant"
    warmup_steps: int = 0
    minimum_learning_rate: float = 0.0
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
        if self.argument_loss_mode not in {"all_fields", "operation_conditioned"}:
            raise ValueError("argument_loss_mode must be all_fields or operation_conditioned")
        if self.learning_rate_schedule not in {"constant", "cosine"}:
            raise ValueError("learning_rate_schedule must be constant or cosine")
        if self.warmup_steps < 0 or self.warmup_steps >= self.steps:
            raise ValueError("warmup_steps must be non-negative and less than steps")
        if not 0.0 <= self.minimum_learning_rate <= self.learning_rate:
            raise ValueError("minimum_learning_rate must be between zero and learning_rate")
        if self.device not in {"auto", "cpu", "cuda", "mps"}:
            raise ValueError("device must be auto, cpu, cuda or mps")

    @classmethod
    def from_mapping(cls, values: Mapping[str, Any]) -> TrainingConfig:
        _validate_fields("TrainingConfig", {item.name for item in fields(TrainingConfig)}, values)
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


@dataclass(frozen=True)
class TrialEvaluationConfig:
    """Evaluation and candidate-generation contract for an engineering trial."""

    corpus_manifest: str = "datasets/venture_corpus_c0/manifest.json"
    eval_interval: int = 25
    evaluation_batch_size: int = 16
    candidates_per_case: int = 16
    generation_temperature: float = 0.75
    generation_seed: int = 1702
    min_edits: int = 1
    max_edits: int = 3
    archive_bins: tuple[int, int] = (4, 4)
    checkpoint_selection: str = "validation_loss"
    memorization_exact_graph_min: float = 0.95
    invalid_candidate_rate_max: float = 0.01

    def __post_init__(self) -> None:
        positive = {
            "eval_interval": self.eval_interval,
            "evaluation_batch_size": self.evaluation_batch_size,
            "candidates_per_case": self.candidates_per_case,
            "max_edits": self.max_edits,
        }
        invalid = [name for name, value in positive.items() if value <= 0]
        if invalid:
            raise ValueError(f"TrialEvaluationConfig values must be positive: {', '.join(invalid)}")
        if self.generation_temperature < 0:
            raise ValueError("generation_temperature must be non-negative")
        if self.min_edits < 0 or self.min_edits > self.max_edits:
            raise ValueError("min_edits must be between zero and max_edits")
        if len(self.archive_bins) != 2 or any(value <= 0 for value in self.archive_bins):
            raise ValueError("archive_bins must contain two positive dimensions")
        if self.checkpoint_selection not in {
            "validation_loss",
            "validation_exact_graph",
        }:
            raise ValueError(
                "checkpoint_selection must be validation_loss or validation_exact_graph"
            )
        for name, value in (
            ("memorization_exact_graph_min", self.memorization_exact_graph_min),
            ("invalid_candidate_rate_max", self.invalid_candidate_rate_max),
        ):
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be in [0, 1]")
        if not self.corpus_manifest:
            raise ValueError("corpus_manifest is required")

    @classmethod
    def from_mapping(cls, values: Mapping[str, Any]) -> TrialEvaluationConfig:
        _validate_fields(
            "TrialEvaluationConfig",
            {item.name for item in fields(TrialEvaluationConfig)},
            values,
        )
        normalized = dict(values)
        if "archive_bins" in normalized:
            normalized["archive_bins"] = tuple(int(value) for value in normalized["archive_bins"])
        return cls(**normalized)


@dataclass(frozen=True)
class VentureTrialConfig:
    """Frozen configuration for a reproducible Venture engineering trial."""

    trial_id: str
    hypothesis_id: str
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    evaluation: TrialEvaluationConfig = field(default_factory=TrialEvaluationConfig)

    def __post_init__(self) -> None:
        if re.fullmatch(r"CHM-V-T\d{3}", self.trial_id) is None:
            raise ValueError("Venture trial IDs must use CHM-V-T###")
        if re.fullmatch(r"CHM-V-H\d{3}", self.hypothesis_id) is None:
            raise ValueError("Venture hypothesis IDs must use CHM-V-H###")
        if self.evaluation.max_edits > self.model.max_edits:
            raise ValueError("trial max_edits exceeds model capacity")

    @classmethod
    def from_mapping(cls, values: Mapping[str, Any]) -> VentureTrialConfig:
        unknown = sorted(
            set(values) - {"trial_id", "hypothesis_id", "model", "training", "evaluation"}
        )
        if unknown:
            raise ValueError(f"Unknown VentureTrialConfig fields: {', '.join(unknown)}")
        for required in ("trial_id", "hypothesis_id"):
            if required not in values:
                raise ValueError(f"{required} is required")
        model_values = values.get("model", {})
        training_values = values.get("training", {})
        evaluation_values = values.get("evaluation", {})
        if not all(
            isinstance(item, Mapping) for item in (model_values, training_values, evaluation_values)
        ):
            raise TypeError("model, training and evaluation must be mappings")
        return cls(
            trial_id=str(values["trial_id"]),
            hypothesis_id=str(values["hypothesis_id"]),
            model=ModelConfig.from_mapping(model_values),
            training=TrainingConfig.from_mapping(training_values),
            evaluation=TrialEvaluationConfig.from_mapping(evaluation_values),
        )

    @classmethod
    def from_yaml(cls, path: str | Path) -> VentureTrialConfig:
        with Path(path).open("r", encoding="utf-8") as handle:
            values = yaml.safe_load(handle)
        if not isinstance(values, Mapping):
            raise TypeError("trial config must contain a mapping")
        return cls.from_mapping(values)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
