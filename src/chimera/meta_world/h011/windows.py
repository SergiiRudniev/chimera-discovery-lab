"""Evaluator-only pairing labels for H011 training windows."""

from __future__ import annotations

from dataclasses import replace

from chimera.meta_world.batch import MetaWorldBatch
from chimera.meta_world.h002.windows import (
    GeneratedSequenceSample,
    make_transition_window,
)


def make_paired_response_window(
    sample: GeneratedSequenceSample,
    prediction_step: int,
    context_steps: int,
) -> MetaWorldBatch:
    window = make_transition_window(
        sample,
        prediction_step=prediction_step,
        context_steps=context_steps,
    )
    return replace(window, mechanism_ids=sample.world_instance_keys)
