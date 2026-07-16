"""Auditable ranking-head training and checkpoint loading for H016."""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path
from typing import Any

import torch

from chimera.meta_world.generators import (
    GeneratedWorldDatasetConfig,
    SplitName,
    WorldGenerationPipeline,
)
from chimera.meta_world.h014.model import (
    ResponseConditionedEffectWorldModel,
    ResponseSource,
)
from chimera.meta_world.h016.config import (
    H016BackboneConfig,
    H016SuiteConfig,
)
from chimera.meta_world.h016.dataset import (
    materialize_ranking_group,
    ranking_group_replay_summary,
)
from chimera.meta_world.h016.model import WithinStateActionRanker
from chimera.meta_world.h016.trainer import H016RankingTrainer


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def model_tensor_sha256(model: torch.nn.Module) -> str:
    """Hash ordered tensor names, shapes, dtypes and exact values."""

    digest = hashlib.sha256()
    for name, value in sorted(model.state_dict().items()):
        tensor = value.detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(tuple(tensor.shape)).encode("ascii"))
        digest.update(str(tensor.dtype).encode("ascii"))
        digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def build_h016_ranker(
    suite: H016SuiteConfig,
    backbone_checkpoint: str | Path,
) -> WithinStateActionRanker:
    """Strictly load the selected H016 backbone and initialize its rank head."""

    backbone_config = H016BackboneConfig.from_yaml(suite.backbone_config)
    torch.manual_seed(suite.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(suite.seed)
    backbone = ResponseConditionedEffectWorldModel(
        backbone_config.paired_runtime.runtime.model,
        response_source=ResponseSource.FACTUAL_RESIDUAL,
    )
    checkpoint = torch.load(
        backbone_checkpoint,
        map_location="cpu",
        weights_only=True,
    )
    backbone.load_state_dict(checkpoint["model"], strict=True)
    return WithinStateActionRanker(backbone)


def run_h016_ranking_training(
    suite: H016SuiteConfig,
    *,
    backbone_checkpoint: str | Path,
    output_dir: str | Path,
    device: torch.device,
    use_autocast: bool,
    steps: int | None = None,
) -> tuple[WithinStateActionRanker, dict[str, Any]]:
    """Train the rank head while proving exact targets and frozen backbone."""

    training_steps = suite.ranking.steps if steps is None else steps
    if training_steps <= 0 or training_steps > suite.ranking.steps:
        raise ValueError("H016 ranking engineering steps are outside the frozen limit")
    output = Path(output_dir)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("H016 ranking output must be empty")
    output.mkdir(parents=True, exist_ok=True)
    ranker = build_h016_ranker(suite, backbone_checkpoint)
    before_hash = model_tensor_sha256(ranker.backbone)
    trainer = H016RankingTrainer(
        ranker,
        suite.ranking,
        device=device,
        use_autocast=use_autocast,
    )
    pipeline = WorldGenerationPipeline(
        GeneratedWorldDatasetConfig.from_yaml(suite.generator_config)
    )
    metric_rows: list[dict[str, Any]] = []
    replay_matches = 0
    group_count = 0
    audit_examples: list[dict[str, object]] = []
    first_metrics: dict[str, float] | None = None
    final_metrics: dict[str, float] | None = None
    started = time.perf_counter()
    for step in range(1, training_steps + 1):
        groups = []
        for offset in range(suite.ranking.states_per_step):
            state_ordinal = (step - 1) * suite.ranking.states_per_step + offset
            group = materialize_ranking_group(
                pipeline,
                SplitName.TRAIN,
                state_ordinal=state_ordinal,
                seed=suite.seed,
                config=suite.ranking,
                audit_replay=True,
            )
            groups.append(group)
            group_count += 1
            replay_matches += int(group.deterministic_replay)
            if len(audit_examples) < 2 or step == training_steps:
                audit_examples.append(
                    ranking_group_replay_summary(pipeline.config, group)
                )
        metrics = trainer.train_step(groups)
        if first_metrics is None:
            first_metrics = metrics
        final_metrics = metrics
        metric_rows.append({"phase": "ranking_train", "step": step, **metrics})
    after_hash = model_tensor_sha256(ranker.backbone)
    replay_rate = replay_matches / max(group_count, 1)
    rank_checkpoint = output / "rank_head.pt"
    torch.save(
        {
            "hypothesis_id": "CHM-W-H016",
            "step": training_steps,
            "rank_head": ranker.rank_head.state_dict(),
            "backbone_tensor_sha256": before_hash,
            "backbone_checkpoint_sha256": _sha256(Path(backbone_checkpoint)),
            "ranking_protocol": {
                "states_per_step": suite.ranking.states_per_step,
                "candidates_per_state": suite.ranking.candidates_per_state,
                "listnet_target_temperature": (
                    suite.ranking.listnet_target_temperature
                ),
                "pairwise_weight": suite.ranking.pairwise_weight,
            },
        },
        rank_checkpoint,
    )
    metrics_path = output / "metrics.jsonl"
    metrics_path.write_bytes(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in metric_rows).encode(
            "utf-8"
        )
    )
    summary: dict[str, Any] = {
        "status": "completed_ranking_training",
        "steps": training_steps,
        "state_groups": group_count,
        "candidates_per_state": suite.ranking.candidates_per_state,
        "candidate_labels_generated": group_count
        * suite.ranking.candidates_per_state,
        "simulator_executions_for_labels_and_replay": group_count
        * suite.ranking.candidates_per_state
        * 2,
        "deterministic_training_candidate_replay_rate": replay_rate,
        "backbone_trainable": False,
        "backbone_tensor_sha256_before": before_hash,
        "backbone_tensor_sha256_after": after_hash,
        "backbone_unchanged": before_hash == after_hash,
        "trainable_parameters": ranker.trainable_parameter_count(),
        "frozen_backbone_parameters": ranker.frozen_backbone_parameter_count(),
        "total_parameters": ranker.total_parameter_count(),
        "first_training": first_metrics,
        "final_training": final_metrics,
        "runtime_seconds": time.perf_counter() - started,
        "peak_memory_bytes": trainer.peak_memory_bytes(),
        "audit_examples": audit_examples,
        "checkpoint": {
            "file": rank_checkpoint.name,
            "sha256": _sha256(rank_checkpoint),
            "promoted": False,
        },
        "metrics_file": metrics_path.name,
    }
    (output / "result.json").write_bytes(
        (json.dumps(summary, indent=2, sort_keys=True) + "\n").encode("utf-8")
    )
    return ranker, summary


def load_h016_rank_head(
    ranker: WithinStateActionRanker,
    checkpoint_path: str | Path,
) -> None:
    """Strictly restore only the rank head and verify its backbone binding."""

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if checkpoint["backbone_tensor_sha256"] != model_tensor_sha256(ranker.backbone):
        raise ValueError("H016 rank head does not match the loaded backbone")
    ranker.rank_head.load_state_dict(checkpoint["rank_head"], strict=True)
