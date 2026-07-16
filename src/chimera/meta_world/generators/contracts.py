"""Numerical contracts for procedurally generated Meta-World environments."""

from __future__ import annotations

from dataclasses import dataclass, fields
from enum import Enum, IntEnum
from typing import Protocol

import numpy as np
import torch
from numpy.typing import NDArray
from torch import Tensor

FloatArray = NDArray[np.float32]
BoolArray = NDArray[np.bool_]
IntArray = NDArray[np.int64]


class WorldFamily(IntEnum):
    """Numeric family identifiers used by the generator, never by the model."""

    FLOW = 0
    COMPETITION = 1
    FUNNEL = 2


class SplitName(str, Enum):
    """Registered generated-world dataset partitions."""

    TRAIN = "train"
    VALIDATION = "validation"
    TEST_WORLD_TRANSFER = "test_world_transfer"
    TEST_MECHANISM = "test_mechanism"
    TEST_RENDERER = "test_renderer"


class ViewCoupling(str, Enum):
    """How multiple observations of one sampled mechanism are coupled."""

    MECHANISM_ONLY = "mechanism_only"
    PAIRED_WORLD_RENDERERS = "paired_world_renderers"


class TrainingFamilyPolicy(str, Enum):
    """Evaluator-controlled family allocation for online training only."""

    CROSS_WORLD = "cross_world"
    HELD_TARGET = "held_target"


@dataclass(frozen=True)
class MechanismConfig:
    """Hidden family-agnostic transition law sampled by MechanismGenerator."""

    mechanism_id: str
    template_id: int
    retention: float
    nonlinearity: float
    threshold: float
    delay_steps: int
    positive_feedback: float
    negative_feedback: float
    saturation: float
    competition: float
    interaction: float
    hidden_coupling: float
    event_rate: float
    latent_weights: FloatArray

    def __post_init__(self) -> None:
        if not self.mechanism_id:
            raise ValueError("mechanism_id cannot be empty")
        if self.template_id < 0 or self.delay_steps < 0:
            raise ValueError("template_id and delay_steps must be non-negative")
        if not 0.0 < self.retention <= 1.0:
            raise ValueError("retention must be in (0, 1]")
        if self.saturation <= 0.0 or not 0.0 <= self.event_rate <= 1.0:
            raise ValueError("invalid saturation or event rate")
        if tuple(self.latent_weights.shape) != (4,):
            raise ValueError("latent_weights must have shape [4]")


@dataclass(frozen=True)
class WorldConfig:
    """Concrete numerical environment sampled from one hidden mechanism."""

    world_instance_id: str
    family_id: WorldFamily
    objects: int
    capacity: FloatArray
    topology: FloatArray
    edge_capacity: FloatArray
    rates: FloatArray
    initial_state: FloatArray
    event_scale: float

    def __post_init__(self) -> None:
        objects = self.objects
        if not self.world_instance_id or objects <= 1:
            raise ValueError("world ID must be set and objects must exceed one")
        expected = {
            "capacity": (objects,),
            "topology": (objects, objects),
            "edge_capacity": (objects, objects),
            "rates": (objects, 4),
            "initial_state": (objects, 4),
        }
        for name, shape in expected.items():
            if tuple(getattr(self, name).shape) != shape:
                raise ValueError(f"{name} must have shape {shape}")
        if np.any(self.capacity <= 0.0) or self.event_scale < 0.0:
            raise ValueError("capacity must be positive and event_scale non-negative")


@dataclass(frozen=True)
class RendererConfig:
    """Observation-only transform; it never changes the hidden world law."""

    renderer_id: str
    profile_id: int
    object_permutation: IntArray
    feature_permutation: IntArray
    relation_permutation: IntArray
    feature_scale: FloatArray
    feature_offset: FloatArray
    visibility: BoolArray
    nonlinear_kind: int
    noise_std: float
    nuisance_features: int
    time_scale: float

    def __post_init__(self) -> None:
        objects, features = self.visibility.shape
        relation_features = int(self.relation_permutation.size)
        if not self.renderer_id:
            raise ValueError("renderer_id cannot be empty")
        if tuple(self.object_permutation.shape) != (objects,):
            raise ValueError("object_permutation shape does not match visibility")
        if tuple(self.feature_permutation.shape) != (features,):
            raise ValueError("feature_permutation shape does not match visibility")
        if tuple(self.feature_scale.shape) != (features,) or tuple(
            self.feature_offset.shape
        ) != (features,):
            raise ValueError("feature scale and offset must match hidden features")
        if sorted(self.object_permutation.tolist()) != list(range(objects)):
            raise ValueError("object_permutation must be bijective")
        if sorted(self.feature_permutation.tolist()) != list(range(features)):
            raise ValueError("feature_permutation must be bijective")
        if sorted(self.relation_permutation.tolist()) != list(range(relation_features)):
            raise ValueError("relation_permutation must be bijective")
        if self.nonlinear_kind not in {0, 1, 2}:
            raise ValueError("unknown renderer nonlinearity")
        if self.noise_std < 0.0 or self.nuisance_features < 0 or self.time_scale <= 0.0:
            raise ValueError("invalid renderer scale, noise or nuisance count")


@dataclass(frozen=True)
class WorldAction:
    """Legal numeric intervention expressed in rendered object coordinates."""

    source: int
    target: int
    magnitude: float
    control: float

    def vector(self) -> FloatArray:
        return np.asarray([self.magnitude, self.control], dtype=np.float32)


@dataclass(frozen=True)
class WorldObservation:
    """Rendered observation with no hidden generator metadata."""

    values: FloatArray
    object_mask: BoolArray
    relations: FloatArray
    relation_mask: BoolArray
    delta_time: float

    def __post_init__(self) -> None:
        if self.values.ndim != 2:
            raise ValueError("observation values must have shape [objects, features]")
        objects = self.values.shape[0]
        if tuple(self.object_mask.shape) != (objects,):
            raise ValueError("object_mask must have shape [objects]")
        if self.relations.ndim != 3 or tuple(self.relations.shape[:2]) != (
            objects,
            objects,
        ):
            raise ValueError("relations must have shape [objects, objects, features]")
        if tuple(self.relation_mask.shape) != (objects, objects):
            raise ValueError("relation_mask must have shape [objects, objects]")
        if self.delta_time <= 0.0:
            raise ValueError("delta_time must be positive")


@dataclass(frozen=True)
class WorldTransition:
    """One action, its next rendered observation and numerical outcome."""

    action: WorldAction
    observation: WorldObservation
    outcome: FloatArray
    counterfactual_no_op_observation: WorldObservation | None = None

    def __post_init__(self) -> None:
        if self.outcome.ndim != 1:
            raise ValueError("outcome must be one-dimensional")


@dataclass(frozen=True)
class TrajectoryMetadata:
    """Evaluator-only provenance that must never enter the model batch."""

    split: SplitName
    world_family_id: int
    world_instance_id: str
    mechanism_id: str
    mechanism_template_id: int
    renderer_id: str
    renderer_profile_id: int
    generation_seed: int
    mechanism_seed: int
    world_seed: int
    renderer_seed: int
    mechanism_config_sha256: str
    world_config_sha256: str
    renderer_config_sha256: str


@dataclass(frozen=True)
class WorldTrajectory:
    """Initial observation followed by action-conditioned transitions."""

    initial_observation: WorldObservation
    transitions: tuple[WorldTransition, ...]
    metadata: TrajectoryMetadata

    def __post_init__(self) -> None:
        if not self.transitions:
            raise ValueError("trajectory must contain at least one transition")


@dataclass(frozen=True)
class DatasetSplitConfig:
    """Immutable mechanism, renderer and seed allocation for one split."""

    name: SplitName
    seed_offset: int
    mechanism_template_ids: tuple[int, ...]
    renderer_profiles: tuple[int, ...]

    def __post_init__(self) -> None:
        if self.seed_offset < 0 or not self.mechanism_template_ids:
            raise ValueError("split seed and mechanism templates must be configured")
        if not self.renderer_profiles:
            raise ValueError("split renderer profiles must be configured")


@dataclass(frozen=True)
class WorldDatasetManifest:
    """Serializable manifest for one fixed generated-world evaluation dataset."""

    dataset_id: str
    schema_version: int
    hypothesis_id: str
    base_seed: int
    trajectory_steps: int
    configuration: dict[str, object]
    counts: dict[str, int]
    shards: dict[str, dict[str, object]]
    tensor_contract: dict[str, list[str]]
    split_policy: dict[str, object]
    source_hashes: dict[str, str]
    claim_boundary: str


@dataclass(frozen=True)
class GeneratedWorldBatch:
    """Only numerical model inputs and targets; no generator IDs or text."""

    observations: Tensor
    object_mask: Tensor
    relations: Tensor
    relation_mask: Tensor
    actions: Tensor
    action_targets: Tensor
    delta_time: Tensor
    outcomes: Tensor
    sequence_mask: Tensor
    counterfactual_no_op_observations: Tensor | None = None

    @property
    def batch_size(self) -> int:
        return int(self.observations.shape[0])

    def validate(self) -> None:
        if self.observations.ndim != 4:
            raise ValueError("observations must have shape [batch, time, objects, features]")
        batch, time, objects, _ = self.observations.shape
        expected_prefixes = {
            "object_mask": (batch, time, objects),
            "relation_mask": (batch, time, objects, objects),
            "action_targets": (batch, time, objects),
            "delta_time": (batch, time),
            "sequence_mask": (batch, time),
        }
        for name, shape in expected_prefixes.items():
            if tuple(getattr(self, name).shape) != shape:
                raise ValueError(f"{name} must have shape {shape}")
        if self.relations.ndim != 5 or tuple(self.relations.shape[:4]) != (
            batch,
            time,
            objects,
            objects,
        ):
            raise ValueError(
                "relations must have shape [batch, time, objects, objects, features]"
            )
        if self.actions.ndim != 3 or tuple(self.actions.shape[:2]) != (batch, time):
            raise ValueError("actions must have shape [batch, time, features]")
        if self.outcomes.ndim != 3 or tuple(self.outcomes.shape[:2]) != (batch, time):
            raise ValueError("outcomes must have shape [batch, time, features]")
        if self.counterfactual_no_op_observations is not None and tuple(
            self.counterfactual_no_op_observations.shape
        ) != tuple(self.observations.shape):
            raise ValueError(
                "counterfactual_no_op_observations must match observations"
            )
        for name in ("object_mask", "relation_mask", "sequence_mask"):
            if getattr(self, name).dtype != torch.bool:
                raise TypeError(f"{name} must be boolean")
        if torch.any(self.object_mask & ~self.sequence_mask.unsqueeze(-1)):
            raise ValueError("objects cannot be active outside sequence_mask")
        if torch.any(self.relation_mask & ~self.object_mask.unsqueeze(-1)) or torch.any(
            self.relation_mask & ~self.object_mask.unsqueeze(-2)
        ):
            raise ValueError("relations can only join active objects")
        if not torch.isfinite(self.observations).all() or not torch.isfinite(
            self.outcomes
        ).all():
            raise ValueError("generated batch tensors must be finite")
        if (
            self.counterfactual_no_op_observations is not None
            and not torch.isfinite(self.counterfactual_no_op_observations).all()
        ):
            raise ValueError("counterfactual no-op tensors must be finite")

    def to(self, device: torch.device | str) -> GeneratedWorldBatch:
        values = {
            item.name: (
                value.to(device)
                if isinstance(value := getattr(self, item.name), Tensor)
                else value
            )
            for item in fields(self)
        }
        return GeneratedWorldBatch(**values)


class GeneratedWorld(Protocol):
    """Common runtime interface for all rendered numerical worlds."""

    mechanism: MechanismConfig
    config: WorldConfig
    renderer_config: RendererConfig

    def reset(self, seed: int) -> WorldObservation:
        ...

    def step(self, action: WorldAction) -> WorldTransition:
        ...

    def sample_action(self, rng: np.random.Generator) -> WorldAction:
        ...

    def sample_latent_action(self, rng: np.random.Generator) -> WorldAction:
        ...

    def render_action(self, action: WorldAction) -> WorldAction:
        ...


class WorldActionPolicy(Protocol):
    """Evaluator-selected numeric excitation policy; never a model input."""

    @property
    def policy_id(self) -> str:
        ...

    def sample_action(
        self,
        world: GeneratedWorld,
        rng: np.random.Generator,
        step: int,
    ) -> WorldAction:
        ...
