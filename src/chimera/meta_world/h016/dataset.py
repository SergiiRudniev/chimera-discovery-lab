"""Exact shared-state/shared-event numerical ranking groups."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from chimera.meta_world.batch import MetaWorldBatch
from chimera.meta_world.generators import (
    GeneratedWorldDatasetConfig,
    SplitName,
    WorldGenerationPipeline,
)
from chimera.meta_world.h002.windows import (
    make_transition_window,
    materialize_sequence_sample,
)
from chimera.meta_world.h015.evaluation import (
    realized_candidate_effect,
    slice_sequence_sample,
    uniform_legal_pool,
)
from chimera.meta_world.h015.search import InterventionCandidate
from chimera.meta_world.h016.config import H016RankingTrainingConfig


@dataclass(frozen=True)
class H016RankingGroup:
    """One model state plus evaluator-only candidates and effects."""

    state_ordinal: int
    trajectory_index: int
    renderer_view: int
    candidate_seed: int
    window: MetaWorldBatch
    candidates: tuple[InterventionCandidate, ...]
    realized_effects: NDArray[np.float64]
    deterministic_replay: bool


def materialize_ranking_group(
    pipeline: WorldGenerationPipeline,
    split: SplitName,
    *,
    state_ordinal: int,
    seed: int,
    config: H016RankingTrainingConfig,
    audit_replay: bool,
) -> H016RankingGroup:
    """Generate alternatives from one state while keeping labels evaluator-only."""

    if state_ordinal < 0 or seed < 0:
        raise ValueError("H016 state ordinal and seed must be non-negative")
    generator = pipeline.config
    views = generator.views_per_mechanism
    group_index, renderer_view = divmod(state_ordinal, views)
    group_start = group_index * views
    trajectory_index = group_start + renderer_view
    trajectory = pipeline.materialize(split, trajectory_index)
    grouped = materialize_sequence_sample(
        pipeline,
        split,
        start_index=group_start,
        batch_size=views,
    )
    sample = slice_sequence_sample(grouped, renderer_view)
    window = make_transition_window(
        sample,
        prediction_step=config.prediction_step,
        context_steps=config.context_steps,
    )
    final_step = int(window.time_mask[0].sum().item()) - 1
    objects = int(window.slot_mask[0, final_step].sum().item())
    candidate_seed = (
        seed
        + state_ordinal * config.candidate_seed_stride
        + config.candidate_seed_offset
    )
    candidates = uniform_legal_pool(
        objects=objects,
        count=config.candidates_per_state,
        seed=candidate_seed,
    )

    def effects() -> NDArray[np.float64]:
        return np.asarray(
            [
                realized_candidate_effect(
                    generator,
                    trajectory,
                    prediction_step=config.prediction_step,
                    candidate=candidate,
                )
                for candidate in candidates
            ],
            dtype=np.float64,
        )

    realized = effects()
    replay_exact = not audit_replay or np.array_equal(realized, effects())
    if not np.isfinite(realized).all():
        raise FloatingPointError("H016 generated a non-finite ranking target")
    return H016RankingGroup(
        state_ordinal=state_ordinal,
        trajectory_index=trajectory_index,
        renderer_view=renderer_view,
        candidate_seed=candidate_seed,
        window=window,
        candidates=candidates,
        realized_effects=realized,
        deterministic_replay=replay_exact,
    )


def ranking_group_replay_summary(
    generator: GeneratedWorldDatasetConfig,
    group: H016RankingGroup,
) -> dict[str, object]:
    """Expose only evaluator provenance needed to audit group generation."""

    return {
        "dataset_id": generator.dataset_id,
        "state_ordinal": group.state_ordinal,
        "trajectory_index": group.trajectory_index,
        "renderer_view": group.renderer_view,
        "candidate_seed": group.candidate_seed,
        "candidates": len(group.candidates),
        "deterministic_replay": group.deterministic_replay,
        "targets_finite": bool(np.isfinite(group.realized_effects).all()),
    }
