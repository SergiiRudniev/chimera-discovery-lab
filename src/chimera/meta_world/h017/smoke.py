"""Postcommit GPU engineering smoke for H017 pool generation and reranking."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from chimera.meta_world.generators import (
    GeneratedWorldDatasetConfig,
    SplitName,
    WorldGenerationPipeline,
)
from chimera.meta_world.h002.windows import (
    make_transition_window,
    materialize_sequence_sample,
)
from chimera.meta_world.h015.evaluation import slice_sequence_sample
from chimera.meta_world.h016.config import H016BackboneConfig, H016SuiteConfig
from chimera.meta_world.h016.evaluation import H016CandidatePredictor
from chimera.meta_world.h016.preflight import run_h016_backbone_preflight
from chimera.meta_world.h016.run import run_h016_ranking_training
from chimera.meta_world.h017.config import H017SuiteConfig
from chimera.meta_world.h017.pool import (
    balanced_support_pool,
    support_pool_diagnostics,
)
from chimera.meta_world.h017.rerank import one_pass_qd_rerank
from chimera.meta_world.trainer import resolve_device


def run_h017_engineering_smoke(
    config_path: str | Path,
    backbone_smoke_config_path: str | Path,
    output_dir: str | Path,
) -> dict[str, Any]:
    """Train two rank steps and score one complete support pool on GPU."""

    suite = H017SuiteConfig.from_yaml(config_path)
    critic_suite = H016SuiteConfig.from_yaml(suite.critic_suite_config)
    output = Path(output_dir)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("H017 smoke output must be empty")
    output.mkdir(parents=True, exist_ok=True)
    backbone_result = run_h016_backbone_preflight(
        backbone_smoke_config_path,
        output / "backbone",
    )
    smoke_backbone = H016BackboneConfig.from_yaml(backbone_smoke_config_path)
    runtime = smoke_backbone.paired_runtime.runtime
    device = resolve_device(runtime.training.device)
    ranker, ranking_result = run_h016_ranking_training(
        critic_suite,
        backbone_checkpoint=output / "backbone" / "checkpoint.pt",
        output_dir=output / "ranking",
        device=device,
        use_autocast=(
            runtime.training.precision == "bfloat16" and device.type == "cuda"
        ),
        steps=2,
    )
    generator = GeneratedWorldDatasetConfig.from_yaml(suite.generator_config)
    pipeline = WorldGenerationPipeline(generator)
    grouped = materialize_sequence_sample(
        pipeline,
        SplitName.VALIDATION,
        start_index=0,
        batch_size=generator.views_per_mechanism,
    )
    window = make_transition_window(
        slice_sequence_sample(grouped, 0),
        prediction_step=critic_suite.ranking.prediction_step,
        context_steps=critic_suite.ranking.context_steps,
    )
    final_step = int(window.time_mask[0].sum().item()) - 1
    objects = int(window.slot_mask[0, final_step].sum().item())
    seed = suite.seed + suite.support_pool.seed_offset
    candidates = balanced_support_pool(
        objects=objects,
        count=suite.support_pool.candidates_per_state,
        seed=seed,
    )
    replay = balanced_support_pool(
        objects=objects,
        count=suite.support_pool.candidates_per_state,
        seed=seed,
    )
    predictor = H016CandidatePredictor(
        model=ranker,
        device=device,
        use_autocast=(
            runtime.training.precision == "bfloat16" and device.type == "cuda"
        ),
    )
    scores, _ = predictor.predict_rank(window, candidates)
    selected = one_pass_qd_rerank(
        candidates,
        scores,
        executions=suite.pool_reranking.simulator_executions_per_state,
    )
    result = {
        "status": "completed_engineering_smoke",
        "hypothesis_id": "CHM-W-H017",
        "backbone_parameters": backbone_result["parameters"],
        "ranking_checkpoint_sha256": ranking_result["checkpoint"]["sha256"],
        "ranking_trainable_parameters": ranking_result["trainable_parameters"],
        "support_pool_replay": candidates == replay,
        "support_pool": support_pool_diagnostics(candidates).to_dict(),
        "model_scores": selected.model_scores,
        "selected_candidates": len(selected.selected),
        "archive_cells": selected.archive_cells,
        "unique_source_target_pairs": selected.unique_source_target_pairs,
        "peak_memory_bytes": max(
            int(backbone_result["peak_memory_bytes"]),
            int(ranking_result["peak_memory_bytes"]),
        ),
        "frozen_validation_seeds_opened": False,
        "test_metrics_opened": False,
        "checkpoint_promoted": False,
    }
    (output / "smoke_result.json").write_bytes(
        (json.dumps(result, indent=2, sort_keys=True) + "\n").encode("utf-8")
    )
    return result
