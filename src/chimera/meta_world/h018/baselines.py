"""Evaluator-only legal random intervention baseline for H018."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

from chimera.meta_world.generators import (
    GeneratedWorldDatasetConfig,
    SplitName,
    WorldFamily,
    WorldGenerator,
)
from chimera.meta_world.h018.dataset import make_h018_pipeline
from chimera.meta_world.h018.programs import MechanismProgramGenerator


@dataclass(frozen=True)
class H018RandomInterventionMetrics:
    samples: int
    candidates_per_sample: int
    legal_action_rate: float
    mean_selected_effect: float
    mean_best_candidate_effect: float
    mean_intervention_regret: float

    def to_dict(self) -> dict[str, int | float]:
        return asdict(self)


def evaluate_h018_random_interventions(
    config: GeneratedWorldDatasetConfig,
    *,
    split: SplitName = SplitName.VALIDATION,
    samples: int = 4,
    candidates_per_sample: int = 8,
) -> H018RandomInterventionMetrics:
    """Evaluate deterministic one-step regret with compositional mechanisms."""

    if split is SplitName.TRAIN:
        raise ValueError("random intervention evaluation requires a held-out split")
    if samples <= 0 or candidates_per_sample < 2:
        raise ValueError("random baseline requires samples and at least two candidates")
    pipeline = make_h018_pipeline(config)
    mechanisms = MechanismProgramGenerator()
    worlds = WorldGenerator(
        min_objects=config.min_objects,
        max_objects=config.max_objects,
        observation_features=config.observation_features,
        relation_features=config.relation_features,
    )
    selected_effects: list[float] = []
    best_effects: list[float] = []
    legal_actions = 0
    total_actions = samples * candidates_per_sample
    stride = config.render_views_per_world
    for sample_index in range(samples):
        metadata = pipeline.materialize(split, sample_index * stride).metadata
        mechanism = mechanisms.generate(
            metadata.mechanism_template_id,
            metadata.mechanism_seed,
        )
        effects: list[float] = []
        for candidate in range(candidates_per_sample):
            world = worlds.generate(
                mechanism,
                WorldFamily(metadata.world_family_id),
                world_seed=metadata.world_seed,
                renderer_seed=metadata.renderer_seed,
                renderer_profile=metadata.renderer_profile_id,
                independent_renderer_rng=True,
            )
            world.reset(metadata.generation_seed)
            action_rng = np.random.default_rng(
                metadata.generation_seed + 10_000_019 + candidate
            )
            action = world.render_action(world.sample_latent_action(action_rng))
            if (
                action.source != action.target
                and 0.0 <= action.magnitude <= 1.0
                and -1.0 <= action.control <= 1.0
            ):
                legal_actions += 1
            effects.append(float(world.step(action).outcome[-1]))
        selected_effects.append(effects[0])
        best_effects.append(max(effects))
    selected = np.asarray(selected_effects, dtype=np.float64)
    best = np.asarray(best_effects, dtype=np.float64)
    if not np.isfinite(selected).all() or not np.isfinite(best).all():
        raise RuntimeError("H018 random baseline produced non-finite effects")
    return H018RandomInterventionMetrics(
        samples=samples,
        candidates_per_sample=candidates_per_sample,
        legal_action_rate=legal_actions / total_actions,
        mean_selected_effect=float(selected.mean()),
        mean_best_candidate_effect=float(best.mean()),
        mean_intervention_regret=float((best - selected).mean()),
    )
