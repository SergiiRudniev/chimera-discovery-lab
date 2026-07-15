"""Online generation and fixed SHA-256 datasets for Meta-World H002."""

from __future__ import annotations

import hashlib
import io
import json
import zipfile
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, cast

import numpy as np
import torch
import yaml
from numpy.typing import NDArray

from chimera.meta_world.generators.contracts import (
    DatasetSplitConfig,
    GeneratedWorldBatch,
    SplitName,
    TrajectoryMetadata,
    WorldDatasetManifest,
    WorldFamily,
    WorldTrajectory,
)
from chimera.meta_world.generators.fingerprints import (
    mechanism_config_sha256,
    renderer_config_sha256,
    world_config_sha256,
)
from chimera.meta_world.generators.mechanisms import MechanismGenerator
from chimera.meta_world.generators.worlds import WorldGenerator

MODEL_FIELDS = (
    "observations",
    "object_mask",
    "relations",
    "relation_mask",
    "actions",
    "action_targets",
    "delta_time",
    "outcomes",
    "sequence_mask",
)
METADATA_FIELDS = (
    "world_family_ids",
    "world_instance_ids",
    "mechanism_ids",
    "mechanism_template_ids",
    "renderer_ids",
    "renderer_profile_ids",
    "generation_seeds",
    "mechanism_seeds",
    "world_seeds",
    "renderer_seeds",
    "mechanism_config_sha256",
    "world_config_sha256",
    "renderer_config_sha256",
)


def _as_mapping(value: object, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{name} must be a mapping")
    return cast(Mapping[str, Any], value)


@dataclass(frozen=True)
class GeneratedWorldDatasetConfig:
    """Executable generator settings loaded from the immutable H002 YAML."""

    hypothesis_id: str
    dataset_id: str
    schema_version: int
    base_seed: int
    trajectory_steps: int
    min_objects: int
    max_objects: int
    state_features: int
    observation_features: int
    relation_features: int
    action_features: int
    outcome_features: int
    views_per_mechanism: int
    fixed_trajectories_per_split: int
    split_configs: tuple[DatasetSplitConfig, ...]
    held_family_by_template: dict[int, int]

    def __post_init__(self) -> None:
        if self.hypothesis_id != "CHM-W-H002" or self.schema_version != 1:
            raise ValueError("unexpected H002 generator identity or schema")
        if self.base_seed < 0 or self.trajectory_steps <= 1:
            raise ValueError("base_seed and trajectory_steps are invalid")
        if self.min_objects <= 1 or self.max_objects < self.min_objects:
            raise ValueError("invalid generated-world object range")
        if (
            self.state_features != 4
            or self.observation_features < self.state_features
            or self.relation_features != 4
            or self.action_features != 2
            or self.outcome_features != 4
        ):
            raise ValueError("H002 tensor dimensions do not match the generator laws")
        if self.views_per_mechanism < 2:
            raise ValueError("mechanism alignment requires at least two views")
        if self.fixed_trajectories_per_split % self.views_per_mechanism:
            raise ValueError("fixed split size must be divisible by views_per_mechanism")
        names = [item.name for item in self.split_configs]
        if set(names) != set(SplitName) or len(names) != len(set(names)):
            raise ValueError("every registered split must appear exactly once")
        known_templates = set(self.split(SplitName.TRAIN).mechanism_template_ids)
        if set(self.held_family_by_template) != known_templates:
            raise ValueError("held family map must cover every known mechanism template")
        if any(
            not 0 <= value < len(WorldFamily)
            for value in self.held_family_by_template.values()
        ):
            raise ValueError("held family map contains an unknown world family")

    def split(self, name: SplitName) -> DatasetSplitConfig:
        return next(item for item in self.split_configs if item.name is name)

    @classmethod
    def from_mapping(cls, values: Mapping[str, Any]) -> GeneratedWorldDatasetConfig:
        generation = _as_mapping(values.get("generation"), "generation")
        split_values = _as_mapping(values.get("splits"), "splits")
        split_configs: list[DatasetSplitConfig] = []
        for split_name in SplitName:
            raw = _as_mapping(split_values.get(split_name.value), split_name.value)
            split_configs.append(
                DatasetSplitConfig(
                    name=split_name,
                    seed_offset=int(raw["seed_offset"]),
                    mechanism_template_ids=tuple(
                        int(item) for item in raw["mechanism_template_ids"]
                    ),
                    renderer_profiles=tuple(int(item) for item in raw["renderer_profiles"]),
                )
            )
        transfer = _as_mapping(values.get("transfer"), "transfer")
        held_raw = _as_mapping(
            transfer.get("held_family_by_template"), "held_family_by_template"
        )
        return cls(
            hypothesis_id=str(values["hypothesis_id"]),
            dataset_id=str(values["dataset_id"]),
            schema_version=int(values["schema_version"]),
            base_seed=int(generation["base_seed"]),
            trajectory_steps=int(generation["trajectory_steps"]),
            min_objects=int(generation["min_objects"]),
            max_objects=int(generation["max_objects"]),
            state_features=int(generation["state_features"]),
            observation_features=int(generation["observation_features"]),
            relation_features=int(generation["relation_features"]),
            action_features=int(generation["action_features"]),
            outcome_features=int(generation["outcome_features"]),
            views_per_mechanism=int(generation["views_per_mechanism"]),
            fixed_trajectories_per_split=int(
                generation["fixed_trajectories_per_split"]
            ),
            split_configs=tuple(split_configs),
            held_family_by_template={int(key): int(value) for key, value in held_raw.items()},
        )

    @classmethod
    def from_yaml(cls, path: str | Path) -> GeneratedWorldDatasetConfig:
        with Path(path).open("r", encoding="utf-8") as handle:
            values = yaml.safe_load(handle)
        return cls.from_mapping(_as_mapping(values, "generated-world config"))

    def to_dict(self) -> dict[str, object]:
        return {
            "hypothesis_id": self.hypothesis_id,
            "dataset_id": self.dataset_id,
            "schema_version": self.schema_version,
            "generation": {
                "base_seed": self.base_seed,
                "trajectory_steps": self.trajectory_steps,
                "min_objects": self.min_objects,
                "max_objects": self.max_objects,
                "state_features": self.state_features,
                "observation_features": self.observation_features,
                "relation_features": self.relation_features,
                "action_features": self.action_features,
                "outcome_features": self.outcome_features,
                "views_per_mechanism": self.views_per_mechanism,
                "fixed_trajectories_per_split": self.fixed_trajectories_per_split,
            },
            "splits": {
                item.name.value: {
                    "seed_offset": item.seed_offset,
                    "mechanism_template_ids": list(item.mechanism_template_ids),
                    "renderer_profiles": list(item.renderer_profiles),
                }
                for item in self.split_configs
            },
            "transfer": {
                "held_family_by_template": {
                    str(key): value
                    for key, value in sorted(self.held_family_by_template.items())
                }
            },
        }


class WorldGenerationPipeline:
    """Seed-addressable trajectory materialization for online and fixed modes."""

    def __init__(self, config: GeneratedWorldDatasetConfig) -> None:
        self.config = config
        self.mechanisms = MechanismGenerator()
        self.worlds = WorldGenerator(
            min_objects=config.min_objects,
            max_objects=config.max_objects,
            observation_features=config.observation_features,
            relation_features=config.relation_features,
        )

    def materialize(self, split: SplitName, index: int) -> WorldTrajectory:
        if not 0 <= index < 1_000_000:
            raise ValueError("trajectory index must be in [0, 1000000)")
        spec = self.config.split(split)
        group = index // self.config.views_per_mechanism
        view = index % self.config.views_per_mechanism
        template_id = spec.mechanism_template_ids[group % len(spec.mechanism_template_ids)]
        prefix = (self.config.base_seed + spec.seed_offset) * 1_000_000
        mechanism_seed = prefix + group * 16 + 1
        world_seed = prefix + 100_000_000 + index * 16 + 2
        renderer_seed = prefix + 200_000_000 + index * 16 + 3
        generation_seed = prefix + 300_000_000 + index * 16 + 4
        mechanism = self.mechanisms.generate(template_id, mechanism_seed)
        family_id = self._family(split, template_id, group, view)
        renderer_profile = spec.renderer_profiles[view % len(spec.renderer_profiles)]
        world = self.worlds.generate(
            mechanism,
            family_id,
            world_seed=world_seed,
            renderer_seed=renderer_seed,
            renderer_profile=renderer_profile,
        )
        initial = world.reset(generation_seed)
        action_rng = np.random.default_rng(generation_seed + 1)
        transitions = tuple(
            world.step(world.sample_action(action_rng))
            for _ in range(self.config.trajectory_steps)
        )
        metadata = TrajectoryMetadata(
            split=split,
            world_family_id=int(family_id),
            world_instance_id=world.config.world_instance_id,
            mechanism_id=mechanism.mechanism_id,
            mechanism_template_id=mechanism.template_id,
            renderer_id=world.renderer_config.renderer_id,
            renderer_profile_id=world.renderer_config.profile_id,
            generation_seed=generation_seed,
            mechanism_seed=mechanism_seed,
            world_seed=world_seed,
            renderer_seed=renderer_seed,
            mechanism_config_sha256=mechanism_config_sha256(mechanism),
            world_config_sha256=world_config_sha256(world.config),
            renderer_config_sha256=renderer_config_sha256(world.renderer_config),
        )
        return WorldTrajectory(
            initial_observation=initial,
            transitions=transitions,
            metadata=metadata,
        )

    def online_batch(
        self,
        split: SplitName,
        batch_size: int,
        *,
        start_index: int = 0,
        device: torch.device | str = "cpu",
    ) -> GeneratedWorldBatch:
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        trajectories = [
            self.materialize(split, start_index + offset) for offset in range(batch_size)
        ]
        return collate_trajectories(
            trajectories,
            max_objects=self.config.max_objects,
        ).to(device)

    def _family(
        self,
        split: SplitName,
        template_id: int,
        group: int,
        view: int,
    ) -> WorldFamily:
        held = self.config.held_family_by_template.get(template_id)
        if split is SplitName.TEST_WORLD_TRANSFER:
            if held is None:
                raise ValueError("world-transfer templates require a held family")
            return WorldFamily(held)
        if held is not None:
            available = [family for family in WorldFamily if int(family) != held]
            return available[view % len(available)]
        return WorldFamily((group + view) % len(WorldFamily))


def collate_trajectories(
    trajectories: Sequence[WorldTrajectory],
    *,
    max_objects: int,
) -> GeneratedWorldBatch:
    """Pad variable worlds into the H002 language-free tensor contract."""

    if not trajectories:
        raise ValueError("at least one trajectory is required")
    batch_size = len(trajectories)
    time = max(len(item.transitions) for item in trajectories)
    observation_features = trajectories[0].initial_observation.values.shape[-1]
    relation_features = trajectories[0].initial_observation.relations.shape[-1]
    outcome_features = trajectories[0].transitions[0].outcome.size
    observations = np.zeros(
        (batch_size, time, max_objects, observation_features), dtype=np.float32
    )
    object_mask = np.zeros((batch_size, time, max_objects), dtype=np.bool_)
    relations = np.zeros(
        (batch_size, time, max_objects, max_objects, relation_features),
        dtype=np.float32,
    )
    relation_mask = np.zeros(
        (batch_size, time, max_objects, max_objects), dtype=np.bool_
    )
    actions = np.zeros((batch_size, time, 2), dtype=np.float32)
    action_targets = np.zeros((batch_size, time, max_objects), dtype=np.float32)
    delta_time = np.zeros((batch_size, time), dtype=np.float32)
    outcomes = np.zeros((batch_size, time, outcome_features), dtype=np.float32)
    sequence_mask = np.zeros((batch_size, time), dtype=np.bool_)
    for batch_index, trajectory in enumerate(trajectories):
        current = trajectory.initial_observation
        for step, transition in enumerate(trajectory.transitions):
            objects = current.values.shape[0]
            if objects > max_objects:
                raise ValueError("trajectory exceeds max_objects")
            observations[batch_index, step, :objects] = current.values
            object_mask[batch_index, step, :objects] = current.object_mask
            relations[batch_index, step, :objects, :objects] = current.relations
            relation_mask[batch_index, step, :objects, :objects] = current.relation_mask
            actions[batch_index, step] = transition.action.vector()
            action_targets[batch_index, step, transition.action.source] = -1.0
            action_targets[batch_index, step, transition.action.target] = 1.0
            delta_time[batch_index, step] = current.delta_time
            outcomes[batch_index, step] = transition.outcome
            sequence_mask[batch_index, step] = True
            current = transition.observation
    batch = GeneratedWorldBatch(
        observations=torch.from_numpy(observations),
        object_mask=torch.from_numpy(object_mask),
        relations=torch.from_numpy(relations),
        relation_mask=torch.from_numpy(relation_mask),
        actions=torch.from_numpy(actions),
        action_targets=torch.from_numpy(action_targets),
        delta_time=torch.from_numpy(delta_time),
        outcomes=torch.from_numpy(outcomes),
        sequence_mask=torch.from_numpy(sequence_mask),
    )
    batch.validate()
    return batch


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, payload: Mapping[str, object]) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _write_deterministic_npz(
    path: Path, arrays: Mapping[str, NDArray[np.generic]]
) -> None:
    """Write object-free NPZ files with fixed member order and ZIP timestamps."""

    with zipfile.ZipFile(path, mode="w", compression=zipfile.ZIP_DEFLATED) as archive:
        for name in sorted(arrays):
            buffer = io.BytesIO()
            np.lib.format.write_array(  # type: ignore[no-untyped-call]
                buffer, np.asarray(arrays[name]), allow_pickle=False
            )
            info = zipfile.ZipInfo(f"{name}.npy", date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o600 << 16
            archive.writestr(info, buffer.getvalue(), compress_type=zipfile.ZIP_DEFLATED)


def _batch_arrays(batch: GeneratedWorldBatch) -> dict[str, NDArray[np.generic]]:
    return {
        name: cast(torch.Tensor, getattr(batch, name)).detach().cpu().numpy()
        for name in MODEL_FIELDS
    }


def _metadata_arrays(
    trajectories: Sequence[WorldTrajectory],
) -> dict[str, NDArray[np.generic]]:
    metadata = [item.metadata for item in trajectories]
    return {
        "world_family_ids": np.asarray(
            [item.world_family_id for item in metadata], dtype=np.int64
        ),
        "world_instance_ids": np.asarray(
            [item.world_instance_id for item in metadata], dtype="<U32"
        ),
        "mechanism_ids": np.asarray(
            [item.mechanism_id for item in metadata], dtype="<U32"
        ),
        "mechanism_template_ids": np.asarray(
            [item.mechanism_template_id for item in metadata], dtype=np.int64
        ),
        "renderer_ids": np.asarray([item.renderer_id for item in metadata], dtype="<U32"),
        "renderer_profile_ids": np.asarray(
            [item.renderer_profile_id for item in metadata], dtype=np.int64
        ),
        "generation_seeds": np.asarray(
            [item.generation_seed for item in metadata], dtype=np.int64
        ),
        "mechanism_seeds": np.asarray(
            [item.mechanism_seed for item in metadata], dtype=np.int64
        ),
        "world_seeds": np.asarray([item.world_seed for item in metadata], dtype=np.int64),
        "renderer_seeds": np.asarray(
            [item.renderer_seed for item in metadata], dtype=np.int64
        ),
        "mechanism_config_sha256": np.asarray(
            [item.mechanism_config_sha256 for item in metadata], dtype="<U64"
        ),
        "world_config_sha256": np.asarray(
            [item.world_config_sha256 for item in metadata], dtype="<U64"
        ),
        "renderer_config_sha256": np.asarray(
            [item.renderer_config_sha256 for item in metadata], dtype="<U64"
        ),
    }


def _source_hashes() -> dict[str, str]:
    root = Path(__file__).parent
    names = (
        "contracts.py",
        "dataset.py",
        "fingerprints.py",
        "mechanisms.py",
        "renderer.py",
        "worlds.py",
    )
    return {name: _sha256(root / name) for name in names}


def build_generated_world_dataset(
    output_dir: str | Path,
    config_path: str | Path,
    *,
    trajectories_per_split: int | None = None,
) -> dict[str, object]:
    """Build a small fixed H002 dataset; online training uses the same pipeline."""

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    config_file = Path(config_path)
    config = GeneratedWorldDatasetConfig.from_yaml(config_file)
    count = trajectories_per_split or config.fixed_trajectories_per_split
    minimum = max(
        len(item.mechanism_template_ids) * config.views_per_mechanism
        for item in config.split_configs
    )
    if count < minimum or count % config.views_per_mechanism:
        raise ValueError(
            "trajectories_per_split must cover every template and complete view group"
        )
    pipeline = WorldGenerationPipeline(config)
    shards: dict[str, dict[str, object]] = {}
    counts: dict[str, int] = {"total": 0}
    for split in SplitName:
        trajectories = [pipeline.materialize(split, index) for index in range(count)]
        batch = collate_trajectories(trajectories, max_objects=config.max_objects)
        arrays = {**_batch_arrays(batch), **_metadata_arrays(trajectories)}
        shard_path = output / f"{split.value}.npz"
        _write_deterministic_npz(shard_path, arrays)
        metadata = [item.metadata for item in trajectories]
        shards[split.value] = {
            "file": shard_path.name,
            "sha256": _sha256(shard_path),
            "trajectories": count,
            "mechanism_template_ids": sorted(
                {item.mechanism_template_id for item in metadata}
            ),
            "world_family_ids": sorted({item.world_family_id for item in metadata}),
            "renderer_profile_ids": sorted(
                {item.renderer_profile_id for item in metadata}
            ),
        }
        counts[split.value] = count
        counts["total"] += count
    manifest = WorldDatasetManifest(
        dataset_id=config.dataset_id,
        schema_version=config.schema_version,
        hypothesis_id=config.hypothesis_id,
        base_seed=config.base_seed,
        trajectory_steps=config.trajectory_steps,
        configuration=config.to_dict(),
        counts=counts,
        shards=shards,
        tensor_contract={
            "observations": ["batch", "time", "objects", "features"],
            "object_mask": ["batch", "time", "objects"],
            "relations": ["batch", "time", "objects", "objects", "relation_features"],
            "relation_mask": ["batch", "time", "objects", "objects"],
            "actions": ["batch", "time", "action_features"],
            "action_targets": ["batch", "time", "objects"],
            "delta_time": ["batch", "time"],
            "outcomes": ["batch", "time", "outcome_features"],
            "sequence_mask": ["batch", "time"],
        },
        split_policy={
            "known_mechanism_templates": list(
                config.split(SplitName.TRAIN).mechanism_template_ids
            ),
            "unseen_mechanism_templates": list(
                config.split(SplitName.TEST_MECHANISM).mechanism_template_ids
            ),
            "held_family_by_template": {
                str(key): value
                for key, value in sorted(config.held_family_by_template.items())
            },
            "known_renderer_profiles": list(
                config.split(SplitName.TRAIN).renderer_profiles
            ),
            "unseen_renderer_profiles": list(
                config.split(SplitName.TEST_RENDERER).renderer_profiles
            ),
            "exact_isolation": [
                "mechanism_id",
                "world_instance_id",
                "seed",
                "world_config_sha256",
                "renderer_config_sha256",
            ],
            "config_sha256": _sha256(config_file),
        },
        source_hashes=_source_hashes(),
        claim_boundary=(
            "Generated numerical simulator data for H002 transfer evaluation; "
            "not evidence of real-world causality, business utility or profitable ideas."
        ),
    )
    payload = cast(dict[str, object], asdict(manifest))
    _write_json(output / "manifest.json", payload)
    return payload


def _pairwise_disjoint(values: Mapping[str, set[object]]) -> bool:
    names = list(values)
    return all(
        values[names[left]].isdisjoint(values[names[right]])
        for left in range(len(names))
        for right in range(left + 1, len(names))
    )


def validate_generated_world_dataset(manifest_path: str | Path) -> dict[str, object]:
    """Validate files, tensor shapes, replay and all registered split boundaries."""

    manifest_file = Path(manifest_path)
    manifest = cast(
        dict[str, Any], json.loads(manifest_file.read_text(encoding="utf-8"))
    )
    config = GeneratedWorldDatasetConfig.from_mapping(
        _as_mapping(manifest.get("configuration"), "manifest configuration")
    )
    root = manifest_file.parent
    pipeline = WorldGenerationPipeline(config)
    file_integrity = True
    source_root = Path(__file__).parent
    source_integrity = all(
        (source_root / name).is_file() and _sha256(source_root / name) == expected
        for name, expected in manifest["source_hashes"].items()
    )
    shape_validity = True
    tensors_finite = True
    replay_exact = True
    counts_valid = True
    metadata_sets: dict[str, dict[str, set[object]]] = {
        field: {} for field in METADATA_FIELDS
    }
    template_sets: dict[str, set[int]] = {}
    family_pairs: dict[str, set[tuple[int, int]]] = {}
    renderer_profiles: dict[str, set[int]] = {}
    for split in SplitName:
        shard = manifest["shards"][split.value]
        path = root / shard["file"]
        file_integrity = file_integrity and path.is_file() and _sha256(path) == shard["sha256"]
        with np.load(path, allow_pickle=False) as arrays:
            if not {*MODEL_FIELDS, *METADATA_FIELDS} <= set(arrays.files):
                shape_validity = False
                continue
            batch = GeneratedWorldBatch(
                **{
                    name: torch.from_numpy(np.asarray(arrays[name]).copy())
                    for name in MODEL_FIELDS
                }
            )
            try:
                batch.validate()
            except (TypeError, ValueError):
                shape_validity = False
            tensors_finite = tensors_finite and all(
                np.isfinite(arrays[name]).all()
                for name in ("observations", "relations", "actions", "outcomes")
            )
            count = int(arrays["mechanism_ids"].shape[0])
            counts_valid = counts_valid and count == int(shard["trajectories"])
            trajectories = [pipeline.materialize(split, index) for index in range(count)]
            expected = {
                **_batch_arrays(
                    collate_trajectories(trajectories, max_objects=config.max_objects)
                ),
                **_metadata_arrays(trajectories),
            }
            replay_exact = replay_exact and all(
                np.array_equal(arrays[name], expected[name]) for name in expected
            )
            for field in METADATA_FIELDS:
                metadata_sets[field][split.value] = set(arrays[field].tolist())
            templates = arrays["mechanism_template_ids"].astype(np.int64)
            families = arrays["world_family_ids"].astype(np.int64)
            template_sets[split.value] = {int(item) for item in templates.tolist()}
            family_pairs[split.value] = set(
                zip(templates.tolist(), families.tolist(), strict=True)
            )
            renderer_profiles[split.value] = {
                int(item) for item in arrays["renderer_profile_ids"].tolist()
            }
    mechanism_isolation = _pairwise_disjoint(metadata_sets["mechanism_ids"])
    world_isolation = _pairwise_disjoint(metadata_sets["world_instance_ids"])
    seed_isolation = all(
        _pairwise_disjoint(metadata_sets[field])
        for field in (
            "generation_seeds",
            "mechanism_seeds",
            "world_seeds",
            "renderer_seeds",
        )
    )
    configuration_isolation = all(
        _pairwise_disjoint(metadata_sets[field])
        for field in (
            "mechanism_config_sha256",
            "world_config_sha256",
            "renderer_config_sha256",
        )
    )
    train_templates = template_sets[SplitName.TRAIN.value]
    known_template_policy = all(
        template_sets[name.value] == train_templates
        for name in (
            SplitName.VALIDATION,
            SplitName.TEST_WORLD_TRANSFER,
            SplitName.TEST_RENDERER,
        )
    )
    unseen_mechanism_policy = template_sets[SplitName.TEST_MECHANISM.value].isdisjoint(
        train_templates
    )
    renderer_policy = renderer_profiles[SplitName.TEST_RENDERER.value].isdisjoint(
        renderer_profiles[SplitName.TRAIN.value]
    )
    held_pairs = {
        (template, family)
        for template, family in config.held_family_by_template.items()
    }
    transfer_policy = (
        family_pairs[SplitName.TEST_WORLD_TRANSFER.value] == held_pairs
        and family_pairs[SplitName.TRAIN.value].isdisjoint(held_pairs)
        and family_pairs[SplitName.VALIDATION.value].isdisjoint(held_pairs)
    )
    checks = {
        "file_integrity": file_integrity,
        "generator_source_integrity": source_integrity,
        "count_consistency": counts_valid,
        "tensor_shapes": shape_validity,
        "tensors_finite": tensors_finite,
        "deterministic_replay": replay_exact,
        "mechanism_id_isolation": mechanism_isolation,
        "world_instance_isolation": world_isolation,
        "seed_isolation": seed_isolation,
        "exact_configuration_isolation": configuration_isolation,
        "known_template_policy": known_template_policy,
        "unseen_mechanism_policy": unseen_mechanism_policy,
        "unseen_renderer_policy": renderer_policy,
        "held_world_family_policy": transfer_policy,
        "service_metadata_excluded_from_model_batch": not set(METADATA_FIELDS)
        & set(MODEL_FIELDS),
    }
    return {
        "dataset_id": config.dataset_id,
        "hypothesis_id": config.hypothesis_id,
        "status": "passed" if all(checks.values()) else "failed",
        "manifest_sha256": _sha256(manifest_file),
        "counts": manifest["counts"],
        "checks": checks,
        "claim_boundary": manifest["claim_boundary"],
    }
