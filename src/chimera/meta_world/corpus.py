"""Procedural, hash-locked trajectory corpus for Chimera Meta-World."""

from __future__ import annotations

import hashlib
import json
import platform
from collections.abc import Sequence
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
import torch

from chimera.meta_world.batch import MetaWorldBatch
from chimera.meta_world.config import MetaWorldExperimentConfig, MetaWorldModelConfig
from chimera.meta_world.model import ChimeraMetaWorld
from chimera.meta_world.synthetic import (
    intervention_parameter_sensitivity,
    make_indexed_batch,
)

CORPUS_ID = "CHM-W-C000"
SPLITS = ("train", "validation", "test", "transfer")
TRANSFER_PAIRS = frozenset({(0, 0), (1, 3), (2, 2), (3, 1)})
SPLIT_ERAS = {
    "train": tuple(range(10)),
    "validation": (10,),
    "test": (11,),
    "transfer": (12,),
}
DEFAULT_REPEATS = {
    "train": 1_280,
    "validation": 128,
    "test": 128,
    "transfer": 512,
}
INDEX_FIELDS = (
    "record_ids",
    "record_seeds",
    "domain_ids",
    "mechanism_ids",
    "intervention_types",
    "eras",
)

IntArray = npt.NDArray[np.int64]

MECHANISM_SEMANTICS = {
    "0": "0.12 * (signed.T @ state - state)",
    "1": "0.08 * tanh(signed.T @ state) + 0.04 * state - 0.025 * state^3",
    "2": (
        "0.06 * state * (1 - adjacency.T @ abs(state)) + "
        "0.02 * tanh(signed.T @ state)"
    ),
    "3": "0.10 * (signed.T @ state) - 0.08 * signed_capacity_overflow",
}
INTERVENTION_PARAMETER_SEMANTICS = {
    "0": "amplitude = 0.05 + 0.20 * sigmoid(p0)",
    "1": "primary_feature = floor(observation_features * sigmoid(p1))",
    "2": "secondary_feature = floor(observation_features * sigmoid(p2))",
    "3": "scope = 0.25 + 0.75 * sigmoid(p3)",
    "4": "edge_gain = 0.50 + sigmoid(p4)",
    "5": "mixing = 0.05 + 0.45 * sigmoid(p5)",
    "6": "delay_strength = 0.10 + 0.60 * sigmoid(p6)",
    "7": "polarity_gain = 0.50 + sigmoid(p7)",
}
INTERVENTION_SEMANTICS = {
    "0": "inject primary at source and scoped secondary at target",
    "1": "dampen primary at source and scoped secondary at target",
    "2": "transfer primary and secondary mass from source to target",
    "3": "reinforce edge magnitude, signed influence and capacity",
    "4": "weaken every feature of one existing edge",
    "5": "equalize primary and secondary features across active slots",
    "6": "mix the intervened transition with the pre-transition state",
    "7": "invert and scale one edge's signed polarity",
}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _pair_list(pairs: set[tuple[int, int]] | frozenset[tuple[int, int]]) -> list[list[int]]:
    return [[domain, mechanism] for domain, mechanism in sorted(pairs)]


class MetaWorldCorpusSplit:
    """Compact numeric index that materializes deterministic trajectories on demand."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        with np.load(self.path, allow_pickle=False) as payload:
            missing = sorted(set(INDEX_FIELDS) - set(payload.files))
            if missing:
                raise ValueError(f"corpus shard is missing fields: {', '.join(missing)}")
            self.arrays: dict[str, IntArray] = {
                name: np.asarray(payload[name], dtype=np.int64).copy() for name in INDEX_FIELDS
            }
        lengths = {len(values) for values in self.arrays.values()}
        if len(lengths) != 1 or not lengths or next(iter(lengths)) <= 0:
            raise ValueError("corpus shard fields must have one shared non-zero length")
        if any(values.ndim != 1 for values in self.arrays.values()):
            raise ValueError("corpus shard fields must be one-dimensional")

    def __len__(self) -> int:
        return len(self.arrays["record_ids"])

    def batch(
        self,
        indices: Sequence[int] | IntArray,
        config: MetaWorldModelConfig,
        active_slots: int,
        transform_seed: int,
        device: torch.device | str = "cpu",
    ) -> MetaWorldBatch:
        selected = np.asarray(indices, dtype=np.int64)
        if selected.ndim != 1 or selected.size == 0:
            raise ValueError("indices must be a non-empty one-dimensional sequence")
        if np.any(selected < 0) or np.any(selected >= len(self)):
            raise IndexError("corpus batch index is outside the shard")
        tensors = {
            name: torch.from_numpy(values[selected].copy()) for name, values in self.arrays.items()
        }
        return make_indexed_batch(
            config,
            record_seeds=tensors["record_seeds"],
            domain_ids=tensors["domain_ids"],
            mechanism_ids=tensors["mechanism_ids"],
            intervention_types=tensors["intervention_types"],
            eras=tensors["eras"],
            active_slots=active_slots,
            transform_seed=transform_seed,
            device=device,
        )


def _split_pairs(config: MetaWorldModelConfig, split: str) -> tuple[tuple[int, int], ...]:
    all_pairs = {
        (domain, mechanism)
        for domain in range(config.domain_count)
        for mechanism in range(config.mechanism_count)
    }
    if config.domain_count != 4 or config.mechanism_count != 4:
        raise ValueError("Corpus C0 registers exactly four domains and four mechanisms")
    selected = TRANSFER_PAIRS if split == "transfer" else all_pairs - TRANSFER_PAIRS
    return tuple(sorted(selected))


def _validate_c0_config(config: MetaWorldModelConfig) -> None:
    exact_values = {
        "observation_features": 12,
        "relation_features": 4,
        "intervention_types": 8,
        "intervention_parameters": 8,
        "effect_dimensions": 4,
        "domain_count": 4,
        "mechanism_count": 4,
        "context_steps": 8,
    }
    mismatches = [
        name
        for name, expected in exact_values.items()
        if getattr(config, name) != expected
    ]
    if mismatches:
        raise ValueError(
            "Corpus C0 requires its registered tensor contract: "
            + ", ".join(mismatches)
        )


def _build_split_index(
    config: MetaWorldModelConfig,
    split: str,
    repeats: int,
    first_record_id: int,
    base_seed: int,
) -> tuple[dict[str, IntArray], int]:
    if repeats <= 0:
        raise ValueError("split repeats must be positive")
    rows: dict[str, list[int]] = {name: [] for name in INDEX_FIELDS}
    record_id = first_record_id
    pairs = _split_pairs(config, split)
    for domain_id, mechanism_id in pairs:
        for intervention_type in range(config.intervention_types):
            for repeat in range(repeats):
                rows["record_ids"].append(record_id)
                rows["record_seeds"].append(base_seed + record_id * 104_729 + 17)
                rows["domain_ids"].append(domain_id)
                rows["mechanism_ids"].append(mechanism_id)
                rows["intervention_types"].append(intervention_type)
                eras = SPLIT_ERAS[split]
                rows["eras"].append(eras[repeat % len(eras)])
                record_id += 1
    arrays = {name: np.asarray(values, dtype=np.int64) for name, values in rows.items()}
    generator = np.random.default_rng(base_seed + 1_000 * (SPLITS.index(split) + 1))
    permutation = generator.permutation(len(arrays["record_ids"]))
    return {name: values[permutation] for name, values in arrays.items()}, record_id


def _distribution(arrays: dict[str, IntArray]) -> dict[str, Any]:
    combination = np.stack(
        [
            arrays["domain_ids"],
            arrays["mechanism_ids"],
            arrays["intervention_types"],
        ],
        axis=1,
    )
    _, counts = np.unique(combination, axis=0, return_counts=True)
    era_combination = np.column_stack((combination, arrays["eras"]))
    _, era_counts = np.unique(era_combination, axis=0, return_counts=True)
    return {
        "rows": len(arrays["record_ids"]),
        "domains": int(np.unique(arrays["domain_ids"]).size),
        "mechanisms": int(np.unique(arrays["mechanism_ids"]).size),
        "interventions": int(np.unique(arrays["intervention_types"]).size),
        "eras": sorted(int(value) for value in np.unique(arrays["eras"])),
        "combination_count_min": int(counts.min()),
        "combination_count_max": int(counts.max()),
        "era_combination_count_min": int(era_counts.min()),
        "era_combination_count_max": int(era_counts.max()),
    }


def _parameter_count(config: MetaWorldModelConfig) -> int:
    with torch.device("meta"):
        model = ChimeraMetaWorld(config)
    return model.trainable_parameter_count()


def _dataset_readme(shards: dict[str, dict[str, Any]]) -> str:
    total = sum(int(metadata["rows"]) for metadata in shards.values())
    rows = "\n".join(
        f"| {split} | {int(shards[split]['rows']):,} | "
        f"{', '.join(str(era) for era in shards[split]['eras'])} |"
        for split in SPLITS
    )
    return f"""# Meta-World Corpus C0

`{CORPUS_ID}` is the first procedural trajectory corpus for Chimera Meta-World W0.

## Grain

One row is one numeric, intervention-conditioned trajectory identified by a unique
`record_id` and `record_seed`. The model receives no text, names or language-model
embeddings.

## Size

| Split | Trajectories | Eras |
|---|---:|---|
{rows}

Total: **{total:,} trajectories**.

Train, validation and test use twelve domain x mechanism pairs. Transfer uses four
disjoint pairs while retaining domains and mechanisms seen in other combinations.

## Files

- `manifest.json`: hashes, split policy, model tensor contract and claim boundary.
- `generator_contract.json`: exact numerical dynamics and intervention semantics.
- `*.npz`: compact `int64` indices; tensors are materialized deterministically.
- `quality_report.json`: automated integrity, leakage and distribution gates.

## Boundary

C0 qualifies mechanistic dynamics learning and combinatorial transfer only. It is
not evidence of real-world causality or production idea quality.

## Commands

```console
chimera validate-meta-world-corpus
```
"""


def build_meta_world_corpus(
    output_dir: str | Path,
    config_path: str | Path,
    *,
    base_seed: int = 260_800,
    active_slots: int = 8,
    train_repeats: int = DEFAULT_REPEATS["train"],
    evaluation_repeats: int = DEFAULT_REPEATS["validation"],
    transfer_repeats: int = DEFAULT_REPEATS["transfer"],
    source_revision: str | None = None,
) -> dict[str, Any]:
    """Build the compact C0 index and its inspectable quality profile."""

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    config_file = Path(config_path)
    experiment = MetaWorldExperimentConfig.from_yaml(config_file)
    config = experiment.model
    _validate_c0_config(config)
    if base_seed < 0:
        raise ValueError("base_seed must be non-negative")
    if active_slots <= 1 or active_slots > config.max_slots:
        raise ValueError("active_slots must be within the model contract")
    repeats = {
        "train": train_repeats,
        "validation": evaluation_repeats,
        "test": evaluation_repeats,
        "transfer": transfer_repeats,
    }
    for split, repeat_count in repeats.items():
        era_count = len(SPLIT_ERAS[split])
        if repeat_count <= 0 or repeat_count % era_count:
            raise ValueError(
                f"{split} repeats must be a positive multiple of its {era_count} eras"
            )
    record_id = 0
    shard_metadata: dict[str, dict[str, Any]] = {}
    for split in SPLITS:
        arrays, record_id = _build_split_index(
            config,
            split,
            repeats[split],
            record_id,
            base_seed,
        )
        shard_path = output / f"{split}.npz"
        np.savez_compressed(
            shard_path,
            record_ids=arrays["record_ids"],
            record_seeds=arrays["record_seeds"],
            domain_ids=arrays["domain_ids"],
            mechanism_ids=arrays["mechanism_ids"],
            intervention_types=arrays["intervention_types"],
            eras=arrays["eras"],
        )
        shard_metadata[split] = {
            "file": shard_path.name,
            "sha256": _sha256(shard_path),
            **_distribution(arrays),
        }

    generator_paths = [Path(__file__), Path(__file__).with_name("synthetic.py")]
    contract: dict[str, Any] = {
        "corpus_id": CORPUS_ID,
        "schema_version": 1,
        "grain": "one intervention-conditioned numeric trajectory",
        "index_fields": list(INDEX_FIELDS),
        "base_seed": base_seed,
        "transform_seed": base_seed,
        "active_slots": active_slots,
        "context_steps": config.context_steps,
        "observation_features": config.observation_features,
        "relation_features": config.relation_features,
        "effect_dimensions": config.effect_dimensions,
        "numeric_semantics": {
            "domain_observation": (
                "tanh(state @ (QR(seed + 10000 * (domain + 1)) * "
                "linspace(0.8, 1.2)) + bias)"
            ),
            "domain_ids": {
                str(domain): f"fixed observation transform {domain}"
                for domain in range(config.domain_count)
            },
            "relation_features": {
                "0": "row-normalized non-negative adjacency magnitude",
                "1": "signed adjacency influence",
                "2": "capacity coefficient in [0.5, 1.5] times adjacency",
                "3": "observed nuisance coefficient in [0, 1] times adjacency",
            },
            "mechanism_deltas": MECHANISM_SEMANTICS,
            "era_multiplier": "1 + 0.015 * era",
            "intervention_parameters": INTERVENTION_PARAMETER_SEMANTICS,
            "interventions": INTERVENTION_SEMANTICS,
            "effect_target": [
                "signed_mean(delta)",
                "mean(abs(delta))",
                "sqrt(mean(delta^2))",
                "max(abs(delta))",
            ],
            "context_observation_probability": 0.90,
            "target_observation_probability": 0.95,
        },
        "transfer_pairs": _pair_list(TRANSFER_PAIRS),
        "split_eras": {name: list(values) for name, values in SPLIT_ERAS.items()},
        "source_hashes": {path.name: _sha256(path) for path in generator_paths},
        "runtime": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "torch": torch.__version__,
        },
        "language_input": False,
    }
    if source_revision is not None:
        contract["source_revision"] = source_revision
    contract_path = output / "generator_contract.json"
    _write_json(contract_path, contract)
    readme_path = output / "README.md"
    readme_path.write_text(_dataset_readme(shard_metadata), encoding="utf-8")
    manifest: dict[str, Any] = {
        "corpus_id": CORPUS_ID,
        "schema_version": 1,
        "storage": "procedural_index",
        "model": {
            "config": asdict(config),
            "source_config": config_file.as_posix(),
            "config_sha256": _sha256(config_file),
            "parameters": _parameter_count(config),
        },
        "generation": {
            "base_seed": base_seed,
            "transform_seed": base_seed,
            "active_slots": active_slots,
            "repeats": repeats,
            "source_revision": source_revision,
        },
        "split_policy": {
            "seen_pairs": _pair_list(
                {
                    (domain, mechanism)
                    for domain in range(config.domain_count)
                    for mechanism in range(config.mechanism_count)
                }
                - TRANSFER_PAIRS
            ),
            "transfer_pairs": _pair_list(TRANSFER_PAIRS),
            "eras": {name: list(values) for name, values in SPLIT_ERAS.items()},
        },
        "counts": {
            "total": sum(int(metadata["rows"]) for metadata in shard_metadata.values()),
            **{name: int(metadata["rows"]) for name, metadata in shard_metadata.items()},
        },
        "shards": shard_metadata,
        "files": {
            contract_path.name: {"sha256": _sha256(contract_path)},
            readme_path.name: {"sha256": _sha256(readme_path)},
            **{
                metadata["file"]: {"sha256": metadata["sha256"]}
                for metadata in shard_metadata.values()
            },
        },
        "quality": {"status": "pending", "report": "quality_report.json"},
        "claim_boundary": (
            "Mechanistic procedural data for dynamics and transfer research; "
            "not evidence of real-world causality or production idea quality."
        ),
    }
    manifest_path = output / "manifest.json"
    _write_json(manifest_path, manifest)
    quality = validate_meta_world_corpus(manifest_path)
    quality_path = output / "quality_report.json"
    _write_json(quality_path, quality)
    manifest["files"][quality_path.name] = {"sha256": _sha256(quality_path)}
    manifest["quality"] = {
        "status": quality["status"],
        "report": quality_path.name,
        "report_sha256": _sha256(quality_path),
    }
    _write_json(manifest_path, manifest)
    return manifest


def _trajectory_hashes(batch: MetaWorldBatch) -> list[str]:
    hashes: list[str] = []
    for index in range(batch.batch_size):
        digest = hashlib.sha256()
        for tensor in (
            batch.observations[index],
            batch.observation_mask[index],
            batch.relations[index],
            batch.intervention_parameters[index],
            batch.next_observations[index],
            batch.effect_targets[index],
        ):
            digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
        hashes.append(digest.hexdigest())
    return hashes


def _stratified_indices(split: MetaWorldCorpusSplit, limit: int) -> IntArray:
    if limit <= 0:
        raise ValueError("sample_per_split must be positive")
    combinations = np.column_stack(
        (
            split.arrays["domain_ids"],
            split.arrays["mechanism_ids"],
            split.arrays["intervention_types"],
        )
    )
    _, combination_indices = np.unique(combinations, axis=0, return_index=True)
    _, era_indices = np.unique(split.arrays["eras"], return_index=True)
    mandatory = np.unique(np.concatenate((combination_indices, era_indices)))
    if mandatory.size >= limit:
        return mandatory[:limit].astype(np.int64, copy=False)
    selected = mandatory.tolist()
    selected_set = set(selected)
    candidates = np.linspace(0, len(split) - 1, limit, dtype=np.int64)
    for candidate in candidates.tolist():
        if candidate not in selected_set:
            selected.append(candidate)
            selected_set.add(candidate)
        if len(selected) == limit:
            break
    if len(selected) < limit:
        for candidate in range(len(split)):
            if candidate not in selected_set:
                selected.append(candidate)
                selected_set.add(candidate)
            if len(selected) == limit:
                break
    return np.asarray(selected, dtype=np.int64)


def _effect_invariants(effect_targets: torch.Tensor) -> bool:
    if effect_targets.shape[1] != 4:
        return False
    signed_mean, mean_absolute, root_mean_square, maximum_absolute = (
        effect_targets.float().unbind(dim=1)
    )
    tolerance = 1e-6
    return bool(
        torch.all(mean_absolute >= -tolerance)
        and torch.all(root_mean_square >= -tolerance)
        and torch.all(maximum_absolute >= -tolerance)
        and torch.all(signed_mean.abs() <= mean_absolute + tolerance)
        and torch.all(mean_absolute <= root_mean_square + tolerance)
        and torch.all(root_mean_square <= maximum_absolute + tolerance)
    )


def validate_meta_world_corpus(
    manifest_path: str | Path,
    *,
    sample_per_split: int = 128,
) -> dict[str, Any]:
    """Profile C0 integrity, isolation, coverage and sampled tensor quality."""

    manifest_file = Path(manifest_path)
    manifest = json.loads(manifest_file.read_text(encoding="utf-8"))
    if manifest.get("corpus_id") != CORPUS_ID:
        raise ValueError("unexpected Meta-World corpus ID")
    root = manifest_file.parent
    expected_files = {
        "generator_contract.json",
        "README.md",
        *(manifest["shards"][name]["file"] for name in SPLITS),
    }
    required_files_declared = expected_files <= set(manifest["files"])
    file_integrity = all(
        (root / name).is_file() and _sha256(root / name) == metadata["sha256"]
        for name, metadata in manifest["files"].items()
    )
    contract = json.loads(
        (root / "generator_contract.json").read_text(encoding="utf-8")
    )
    generator_paths = [Path(__file__), Path(__file__).with_name("synthetic.py")]
    generator_source_integrity = all(
        contract["source_hashes"].get(path.name) == _sha256(path)
        for path in generator_paths
    )
    config = MetaWorldModelConfig.from_mapping(manifest["model"]["config"])
    _validate_c0_config(config)
    parameter_count_matches = (
        int(manifest["model"]["parameters"]) == _parameter_count(config)
    )
    active_slots = int(manifest["generation"]["active_slots"])
    transform_seed = int(manifest["generation"]["transform_seed"])
    splits = {
        name: MetaWorldCorpusSplit(root / manifest["shards"][name]["file"])
        for name in SPLITS
    }
    actual_counts = {name: len(split) for name, split in splits.items()}
    count_consistency = (
        all(actual_counts[name] == int(manifest["counts"][name]) for name in SPLITS)
        and sum(actual_counts.values()) == int(manifest["counts"]["total"])
    )
    shard_profile_consistency = all(
        all(
            manifest["shards"][name][field] == value
            for field, value in _distribution(split.arrays).items()
        )
        for name, split in splits.items()
    )
    all_record_ids = np.concatenate(
        [split.arrays["record_ids"] for split in splits.values()]
    )
    all_record_seeds = np.concatenate(
        [split.arrays["record_seeds"] for split in splits.values()]
    )
    unique_record_ids = np.unique(all_record_ids).size == all_record_ids.size
    unique_record_seeds = np.unique(all_record_seeds).size == all_record_seeds.size
    seen_pairs = {
        tuple(values)
        for values in manifest["split_policy"]["seen_pairs"]
    }
    transfer_pairs = {
        tuple(values)
        for values in manifest["split_policy"]["transfer_pairs"]
    }
    observed_pairs = {
        name: set(
            zip(
                split.arrays["domain_ids"].tolist(),
                split.arrays["mechanism_ids"].tolist(),
                strict=True,
            )
        )
        for name, split in splits.items()
    }
    pair_isolation = (
        observed_pairs["train"] == seen_pairs
        and observed_pairs["validation"] == seen_pairs
        and observed_pairs["test"] == seen_pairs
        and observed_pairs["transfer"] == transfer_pairs
        and seen_pairs.isdisjoint(transfer_pairs)
    )
    temporal_isolation = all(
        set(split.arrays["eras"].tolist())
        == set(manifest["split_policy"]["eras"][name])
        for name, split in splits.items()
    )
    enum_validity = all(
        np.all((split.arrays["domain_ids"] >= 0) & (split.arrays["domain_ids"] < 4))
        and np.all(
            (split.arrays["mechanism_ids"] >= 0) & (split.arrays["mechanism_ids"] < 4)
        )
        and np.all(
            (split.arrays["intervention_types"] >= 0)
            & (split.arrays["intervention_types"] < config.intervention_types)
        )
        for split in splits.values()
    )
    balanced = all(
        metadata["combination_count_min"] == metadata["combination_count_max"]
        for metadata in manifest["shards"].values()
    )
    era_balanced = all(
        metadata["era_combination_count_min"]
        == metadata["era_combination_count_max"]
        for metadata in manifest["shards"].values()
    )

    sampled_hashes: list[str] = []
    replay_exact = True
    tensors_finite = True
    squared_error = 0.0
    persistence_values = 0
    observed_values = 0
    possible_values = 0
    effect_nonzero = 0
    sampled_rows = 0
    split_profiles: dict[str, Any] = {}
    effect_invariants = True
    sampled_combination_coverage = True
    sampled_era_coverage = True
    intervention_magnitudes: dict[int, list[float]] = {
        intervention: [] for intervention in range(config.intervention_types)
    }
    mechanism_intervention_magnitudes: dict[tuple[int, int], list[float]] = {
        (mechanism, intervention): []
        for mechanism in range(config.mechanism_count)
        for intervention in range(config.intervention_types)
    }
    for name, split in splits.items():
        sample_count = min(sample_per_split, len(split))
        indices = _stratified_indices(split, sample_count)
        first = split.batch(indices, config, active_slots, transform_seed)
        second = split.batch(indices, config, active_slots, transform_seed)
        replay_exact = replay_exact and all(
            torch.equal(getattr(first, field), getattr(second, field))
            for field in first.__dataclass_fields__
        )
        tensors_finite = tensors_finite and all(
            torch.isfinite(tensor).all().item()
            for tensor in (
                first.observations,
                first.relations,
                first.intervention_parameters,
                first.next_observations,
                first.effect_targets,
            )
        )
        effect_invariants = effect_invariants and _effect_invariants(
            first.effect_targets
        )
        sampled_hashes.extend(_trajectory_hashes(first))
        final_observation = first.observations[:, -1]
        persistence_mask = first.observation_mask[:, -1] & first.next_observation_mask
        errors = (final_observation - first.next_observations).square()
        squared_error += float(errors[persistence_mask].sum())
        persistence_values += int(persistence_mask.sum())
        observed_values += int(first.observation_mask.sum() + first.next_observation_mask.sum())
        possible_values += int(
            first.slot_mask.sum() * config.observation_features
            + first.next_observation_mask.shape[0] * active_slots * config.observation_features
        )
        nonzero = first.effect_targets.abs().amax(dim=1) > 1e-8
        effect_nonzero += int(nonzero.sum())
        sampled_rows += first.batch_size
        sampled_combinations = set(
            zip(
                first.domain_ids.tolist(),
                first.mechanism_ids.tolist(),
                first.intervention_types.tolist(),
                strict=True,
            )
        )
        full_combinations = set(
            zip(
                split.arrays["domain_ids"].tolist(),
                split.arrays["mechanism_ids"].tolist(),
                split.arrays["intervention_types"].tolist(),
                strict=True,
            )
        )
        combination_coverage_rate = len(sampled_combinations) / len(full_combinations)
        sampled_combination_coverage = (
            sampled_combination_coverage and combination_coverage_rate == 1.0
        )
        sampled_eras = set(split.arrays["eras"][indices].tolist())
        full_eras = set(split.arrays["eras"].tolist())
        era_coverage_rate = len(sampled_eras) / len(full_eras)
        sampled_era_coverage = sampled_era_coverage and era_coverage_rate == 1.0
        maximum_effect = first.effect_targets[:, 3].float()
        for intervention in range(config.intervention_types):
            selection = first.intervention_types == intervention
            intervention_magnitudes[intervention].extend(
                maximum_effect[selection].tolist()
            )
        for mechanism in range(config.mechanism_count):
            for intervention in range(config.intervention_types):
                selection = (first.mechanism_ids == mechanism) & (
                    first.intervention_types == intervention
                )
                mechanism_intervention_magnitudes[(mechanism, intervention)].extend(
                    maximum_effect[selection].tolist()
                )
        split_profiles[name] = {
            "sampled_rows": first.batch_size,
            "combination_coverage_rate": combination_coverage_rate,
            "era_coverage_rate": era_coverage_rate,
            "effect_nonzero_rate": float(nonzero.float().mean()),
            "observation_min": float(first.observations.min()),
            "observation_max": float(first.observations.max()),
        }

    sample_duplicates = len(sampled_hashes) - len(set(sampled_hashes))
    completeness_rate = observed_values / max(possible_values, 1)
    persistence_rmse = (squared_error / max(persistence_values, 1)) ** 0.5
    effect_nonzero_rate = effect_nonzero / max(sampled_rows, 1)
    intervention_profiles = {
        str(intervention): {
            "sampled_rows": len(values),
            "nonzero_rate": float(np.mean(np.asarray(values) > 1e-8)),
            "q10": float(np.quantile(values, 0.10)),
            "median": float(np.median(values)),
            "q90": float(np.quantile(values, 0.90)),
        }
        for intervention, values in intervention_magnitudes.items()
        if values
    }
    intervention_coverage = len(intervention_profiles) == config.intervention_types
    intervention_nonzero = intervention_coverage and all(
        profile["nonzero_rate"] >= 0.85
        for profile in intervention_profiles.values()
    )
    intervention_variation = intervention_coverage and all(
        profile["q90"] - profile["q10"] > 1e-6
        for profile in intervention_profiles.values()
    )
    mechanism_intervention_profiles = {
        f"{mechanism}:{intervention}": {
            "sampled_rows": len(values),
            "nonzero_rate": float(np.mean(np.asarray(values) > 1e-8)),
            "q10": float(np.quantile(values, 0.10)),
            "median": float(np.median(values)),
            "q90": float(np.quantile(values, 0.90)),
        }
        for (mechanism, intervention), values in (
            mechanism_intervention_magnitudes.items()
        )
        if values
    }
    mechanism_intervention_coverage = len(mechanism_intervention_profiles) == (
        config.mechanism_count * config.intervention_types
    )
    mechanism_intervention_nonzero = mechanism_intervention_coverage and all(
        profile["nonzero_rate"] >= 0.80
        for profile in mechanism_intervention_profiles.values()
    )
    mechanism_intervention_variation = mechanism_intervention_coverage and all(
        profile["q90"] - profile["q10"] > 1e-7
        for profile in mechanism_intervention_profiles.values()
    )
    parameter_sensitivity = intervention_parameter_sensitivity(config)
    checks = {
        "required_files_declared": required_files_declared,
        "file_integrity": file_integrity,
        "generator_source_integrity": generator_source_integrity,
        "model_parameter_count": parameter_count_matches,
        "count_consistency": count_consistency,
        "shard_profile_consistency": shard_profile_consistency,
        "record_id_uniqueness": unique_record_ids,
        "record_seed_uniqueness": unique_record_seeds,
        "enum_validity": enum_validity,
        "domain_mechanism_pair_isolation": pair_isolation,
        "era_isolation": temporal_isolation,
        "combination_balance": balanced,
        "era_combination_balance": era_balanced,
        "deterministic_replay": replay_exact,
        "sampled_tensors_finite": tensors_finite,
        "sampled_exact_duplicates_zero": sample_duplicates == 0,
        "sampled_combination_coverage": sampled_combination_coverage,
        "sampled_era_coverage": sampled_era_coverage,
        "effect_summary_invariants": effect_invariants,
        "effect_nonzero_rate": effect_nonzero_rate >= 0.95,
        "effect_nonzero_by_intervention": intervention_nonzero,
        "effect_variation_by_intervention": intervention_variation,
        "effect_nonzero_by_mechanism_intervention": mechanism_intervention_nonzero,
        "effect_variation_by_mechanism_intervention": (
            mechanism_intervention_variation
        ),
        "intervention_parameter_sensitivity": all(parameter_sensitivity),
        "observation_completeness": 0.85 <= completeness_rate <= 0.98,
    }
    return {
        "corpus_id": CORPUS_ID,
        "status": "passed" if all(checks.values()) else "failed",
        "grain": "one intervention-conditioned numeric trajectory",
        "counts": manifest["counts"],
        "checks": checks,
        "profile": {
            "sampled_rows": sampled_rows,
            "sampled_exact_duplicates": sample_duplicates,
            "observation_completeness_rate": completeness_rate,
            "effect_nonzero_rate": effect_nonzero_rate,
            "persistence_rmse": persistence_rmse,
            "split_profiles": split_profiles,
            "intervention_profiles": intervention_profiles,
            "mechanism_intervention_profiles": mechanism_intervention_profiles,
            "intervention_parameter_sensitivity": list(parameter_sensitivity),
        },
        "risks": [
            {
                "severity": "high",
                "finding": (
                    "C0 is generated from known mechanisms rather than observed "
                    "real-world events."
                ),
                "impact": (
                    "It can qualify dynamics learning but cannot establish real-world "
                    "causality or idea quality."
                ),
            },
            {
                "severity": "medium",
                "finding": "Domain transforms are fixed within C0.",
                "impact": (
                    "Transfer evidence is combinatorial, not transfer to an entirely "
                    "unseen sensor domain."
                ),
            },
        ],
    }
