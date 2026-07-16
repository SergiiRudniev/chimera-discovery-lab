"""Online and fixed H018 compositional-world datasets."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, cast

import numpy as np

from chimera.meta_world.generators import (
    GeneratedWorldDatasetConfig,
    TrainingFamilyPolicy,
    TransferTemplatePolicy,
    WorldGenerationPipeline,
    build_generated_world_dataset,
    validate_generated_world_dataset,
)
from chimera.meta_world.h018.programs import (
    MECHANISM_TEST_PROGRAM_IDS,
    PROGRAM_SPECS,
    TRAIN_PROGRAM_IDS,
    TRANSFER_PROGRAM_IDS,
    MechanismProgramGenerator,
    operator_ids,
    program_catalogue,
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _assert_h018_config(config: GeneratedWorldDatasetConfig) -> None:
    if (
        config.hypothesis_id != "CHM-W-H018"
        or config.dataset_id != "CHM-W-WG5"
        or config.schema_version != 4
        or config.transfer_template_policy
        is not TransferTemplatePolicy.HELD_OUT_COMPOSITION
    ):
        raise ValueError("H018 requires the registered WG5 schema-v4 generator")
    actual = {
        "train": set(config.split_configs[0].mechanism_template_ids),
        "transfer": set(config.split_configs[2].mechanism_template_ids),
        "mechanism": set(config.split_configs[3].mechanism_template_ids),
    }
    expected = {
        "train": set(TRAIN_PROGRAM_IDS),
        "transfer": set(TRANSFER_PROGRAM_IDS),
        "mechanism": set(MECHANISM_TEST_PROGRAM_IDS),
    }
    if actual != expected:
        raise ValueError("H018 program split differs from preregistration")


def make_h018_pipeline(
    config: GeneratedWorldDatasetConfig,
    *,
    training_family_policy: TrainingFamilyPolicy = TrainingFamilyPolicy.CROSS_WORLD,
) -> WorldGenerationPipeline:
    """Construct the shared world stack with the H018 mechanism compiler."""

    _assert_h018_config(config)
    return WorldGenerationPipeline(
        config,
        training_family_policy=training_family_policy,
        mechanism_generator=MechanismProgramGenerator(),
    )


def build_h018_smoke_dataset(
    output_dir: str | Path,
    config_path: str | Path,
    *,
    trajectories_per_split: int = 24,
) -> dict[str, Any]:
    """Build a deterministic WG5 engineering dataset without opening metrics."""

    output = Path(output_dir)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("H018 smoke output directory must be empty")
    config = GeneratedWorldDatasetConfig.from_yaml(config_path)
    _assert_h018_config(config)
    programs_file = Path(__file__).with_name("programs.py")
    build_generated_world_dataset(
        output,
        config_path,
        trajectories_per_split=trajectories_per_split,
        additional_source_hashes={"../h018/programs.py": _sha256(programs_file)},
        pipeline_factory=make_h018_pipeline,
        claim_boundary=(
            "H018 compositional-world engineering smoke only; no transfer, causal, "
            "business-utility, language-independence or production claim."
        ),
    )
    manifest_path = output / "manifest.json"
    manifest = cast(
        dict[str, Any], json.loads(manifest_path.read_text(encoding="utf-8"))
    )
    manifest["mechanism_program_catalogue"] = program_catalogue()
    manifest["program_split_policy"] = {
        "train": sorted(TRAIN_PROGRAM_IDS),
        "test_world_transfer": sorted(TRANSFER_PROGRAM_IDS),
        "test_mechanism": sorted(MECHANISM_TEST_PROGRAM_IDS),
    }
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    report = validate_h018_dataset(manifest_path)
    if report["status"] != "passed":
        raise RuntimeError("H018 smoke dataset failed integrity validation")
    return report


def validate_h018_dataset(manifest_path: str | Path) -> dict[str, Any]:
    """Validate generic isolation plus the held-out program boundary."""

    manifest_file = Path(manifest_path)
    report = cast(
        dict[str, Any],
        validate_generated_world_dataset(
            manifest_file,
            pipeline_factory=make_h018_pipeline,
        ),
    )
    manifest = cast(
        dict[str, Any], json.loads(manifest_file.read_text(encoding="utf-8"))
    )
    policy = cast(dict[str, list[int]], manifest.get("program_split_policy", {}))
    catalogue = cast(dict[str, list[int]], manifest.get("mechanism_program_catalogue", {}))
    expected_catalogue = program_catalogue()
    train = set(policy.get("train", []))
    transfer = set(policy.get("test_world_transfer", []))
    mechanism = set(policy.get("test_mechanism", []))
    shard_programs: dict[str, set[int]] = {}
    for split in ("train", "test_world_transfer", "test_mechanism"):
        shard = cast(dict[str, Any], manifest["shards"][split])
        with np.load(manifest_file.parent / str(shard["file"]), allow_pickle=False) as arrays:
            shard_programs[split] = {
                int(value) for value in arrays["mechanism_template_ids"].tolist()
            }
    checks = cast(dict[str, bool], report["checks"])
    checks.update(
        {
            "mechanism_program_catalogue_exact": catalogue == expected_catalogue,
            "registered_program_split_exact": (
                train == set(TRAIN_PROGRAM_IDS)
                and transfer == set(TRANSFER_PROGRAM_IDS)
                and mechanism == set(MECHANISM_TEST_PROGRAM_IDS)
                and shard_programs["train"] == train
                and shard_programs["test_world_transfer"] == transfer
                and shard_programs["test_mechanism"] == mechanism
            ),
            "exact_train_transfer_program_overlap_zero": train.isdisjoint(transfer),
            "all_transfer_primitives_seen_in_train": operator_ids(transfer).issubset(
                operator_ids(train)
            ),
            "program_depth_policy": (
                all(len(PROGRAM_SPECS[item].operators) == 2 for item in train | transfer)
                and all(len(PROGRAM_SPECS[item].operators) == 3 for item in mechanism)
            ),
        }
    )
    report["status"] = "passed" if all(checks.values()) else "failed"
    report["manifest_sha256"] = _sha256(manifest_file)
    report["programs"] = {
        "train": sorted(train),
        "test_world_transfer": sorted(transfer),
        "test_mechanism": sorted(mechanism),
        "exact_train_transfer_overlap": len(train & transfer),
        "train_primitive_ids": sorted(operator_ids(train)),
        "transfer_primitive_ids": sorted(operator_ids(transfer)),
    }
    return report
