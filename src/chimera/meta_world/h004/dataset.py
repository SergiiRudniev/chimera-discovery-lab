"""WG1 online/fixed datasets with active-identification action policies."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import numpy as np
import yaml
from numpy.typing import NDArray

from chimera.meta_world.generators import (
    GeneratedWorldDatasetConfig,
    SplitName,
    build_generated_world_dataset,
    validate_generated_world_dataset,
)
from chimera.meta_world.generators.contracts import WorldActionPolicy
from chimera.meta_world.h004.probes import (
    HybridProbePolicy,
    SeededRandomPolicy,
    SystemIdentificationProbePolicy,
)


def _mapping(value: object, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{name} must be a mapping")
    return cast(Mapping[str, Any], value)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


@dataclass(frozen=True)
class H004DatasetConfig:
    """Base world generator plus train/evaluation excitation policies."""

    worlds: GeneratedWorldDatasetConfig
    training_policy: str
    evaluation_policy: str
    probe_prefix_steps: int

    def __post_init__(self) -> None:
        if self.worlds.hypothesis_id != "CHM-W-H004":
            raise ValueError("WG1 must belong to CHM-W-H004")
        if self.worlds.dataset_id != "CHM-W-WG1":
            raise ValueError("H004 dataset_id must be CHM-W-WG1")
        if self.worlds.trajectory_steps < 8:
            raise ValueError("WG1 requires at least eight excitation steps")
        if self.training_policy != "deterministic_system_identification_probe_v1":
            raise ValueError("unknown H004 training action policy")
        if self.evaluation_policy != "probe_prefix_then_seeded_random_v1":
            raise ValueError("unknown H004 evaluation action policy")
        if not 0 < self.probe_prefix_steps < self.worlds.trajectory_steps:
            raise ValueError("probe prefix must fit inside a WG1 trajectory")

    @classmethod
    def from_yaml(cls, path: str | Path) -> H004DatasetConfig:
        with Path(path).open("r", encoding="utf-8") as handle:
            values = yaml.safe_load(handle)
        root = _mapping(values, "H004 dataset config")
        policies = _mapping(root.get("action_policies"), "action_policies")
        return cls(
            worlds=GeneratedWorldDatasetConfig.from_mapping(root),
            training_policy=str(policies["training"]),
            evaluation_policy=str(policies["evaluation"]),
            probe_prefix_steps=int(policies["probe_prefix_steps"]),
        )

    def policies(self) -> dict[SplitName, WorldActionPolicy]:
        training = SystemIdentificationProbePolicy()
        evaluation = HybridProbePolicy(self.probe_prefix_steps)
        return {
            split: training if split is SplitName.TRAIN else evaluation
            for split in SplitName
        }

    def policy_ids(self) -> dict[SplitName, str]:
        return {split: policy.policy_id for split, policy in self.policies().items()}

    def random_training_policies(self) -> dict[SplitName, WorldActionPolicy]:
        evaluation = HybridProbePolicy(self.probe_prefix_steps)
        return {
            split: SeededRandomPolicy() if split is SplitName.TRAIN else evaluation
            for split in SplitName
        }


def build_h004_probe_dataset(
    output_dir: str | Path,
    config_path: str | Path,
    *,
    trajectories_per_split: int | None = None,
) -> dict[str, object]:
    """Build deterministic WG1 shards with probe train and hybrid evaluation."""

    config = H004DatasetConfig.from_yaml(config_path)
    policy_path = Path(__file__).with_name("probes.py")
    return build_generated_world_dataset(
        output_dir,
        config_path,
        trajectories_per_split=trajectories_per_split,
        action_policies=config.policies(),
        action_policy_ids=config.policy_ids(),
        additional_source_hashes={"../h004/probes.py": _sha256(policy_path)},
        claim_boundary=(
            "Generated numerical active-identification data for H004 simulator transfer; "
            "not evidence of real-world causality, experiment safety or business utility."
        ),
    )


def _policy_from_id(policy_id: str) -> WorldActionPolicy:
    if policy_id == SystemIdentificationProbePolicy().policy_id:
        return SystemIdentificationProbePolicy()
    if policy_id == SeededRandomPolicy().policy_id:
        return SeededRandomPolicy()
    prefix = "probe_prefix_"
    suffix = "_then_seeded_random_v1"
    if policy_id.startswith(prefix) and policy_id.endswith(suffix):
        steps = int(policy_id[len(prefix) : -len(suffix)])
        return HybridProbePolicy(steps)
    raise ValueError(f"unknown action policy ID: {policy_id}")


def _probe_response_separation(
    outcomes: NDArray[np.generic],
    views: int,
) -> float:
    effect = np.asarray(outcomes[..., 3], dtype=np.float64)
    groups = effect.reshape(effect.shape[0] // views, views, effect.shape[1])
    centroids = groups.mean(axis=1)
    within = np.linalg.norm(groups - centroids[:, None, :], axis=-1).mean()
    if centroids.shape[0] <= 1:
        return 0.0
    differences = centroids[:, None, :] - centroids[None, :, :]
    mask = ~np.eye(centroids.shape[0], dtype=bool)
    between = np.linalg.norm(differences, axis=-1)[mask].mean()
    return float(between / max(float(within), 1e-8))


def validate_h004_probe_dataset(manifest_path: str | Path) -> dict[str, object]:
    """Validate WG1 replay, isolation, probe coverage and response excitation."""

    manifest_file = Path(manifest_path)
    manifest = cast(
        dict[str, Any],
        json.loads(manifest_file.read_text(encoding="utf-8")),
    )
    configuration = _mapping(manifest.get("configuration"), "configuration")
    raw_policy_ids = _mapping(configuration.get("action_policies"), "action_policies")
    policy_ids = {
        SplitName(split): str(policy_id) for split, policy_id in raw_policy_ids.items()
    }
    policies = {split: _policy_from_id(policy_ids[split]) for split in SplitName}
    report = validate_generated_world_dataset(
        manifest_file,
        action_policies=policies,
    )
    train_path = manifest_file.parent / str(
        cast(Mapping[str, Any], manifest["shards"])[SplitName.TRAIN.value]["file"]
    )
    validation_path = manifest_file.parent / str(
        cast(Mapping[str, Any], manifest["shards"])[SplitName.VALIDATION.value]["file"]
    )
    with np.load(train_path, allow_pickle=False) as arrays:
        train_actions = np.asarray(arrays["actions"])
        response_separation = _probe_response_separation(
            np.asarray(arrays["outcomes"]),
            GeneratedWorldDatasetConfig.from_mapping(configuration).views_per_mechanism,
        )
    with np.load(validation_path, allow_pickle=False) as arrays:
        validation_actions = np.asarray(arrays["actions"])
    expected_magnitude = np.asarray([0.0, 0.25, 0.85, 0.85], dtype=np.float32)
    expected_control = np.asarray([0.0, 0.0, 1.0, -1.0], dtype=np.float32)
    train_prefix = bool(
        np.allclose(train_actions[:, :4, 0], expected_magnitude)
        and np.allclose(train_actions[:, :4, 1], expected_control)
    )
    evaluation_prefix = bool(
        np.allclose(validation_actions[:, :4, 0], expected_magnitude)
        and np.allclose(validation_actions[:, :4, 1], expected_control)
    )
    checks = cast(dict[str, bool], report["checks"])
    checks.update(
        {
            "action_policy_manifest": set(policy_ids) == set(SplitName),
            "train_probe_prefix": train_prefix,
            "evaluation_probe_prefix": evaluation_prefix,
            "probe_response_finite": bool(np.isfinite(response_separation)),
            "probe_response_nonzero": response_separation > 0.0,
        }
    )
    report["checks"] = checks
    report["status"] = "passed" if all(checks.values()) else "failed"
    report["probe_response_separation"] = response_separation
    report["action_policies"] = {
        split.value: policy_ids[split] for split in SplitName
    }
    return report
