"""Evaluator-only legal intervention baseline for CHM-W-H008."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

from chimera.meta_world.generators import (
    GeneratedWorldDatasetConfig,
    MechanismGenerator,
    SplitName,
    WorldFamily,
    WorldGenerationPipeline,
    WorldGenerator,
)


@dataclass(frozen=True)
class RandomInterventionMetrics:
    """Regret of one seeded legal action against sampled alternatives."""

    samples: int
    candidates_per_sample: int
    legal_action_rate: float
    mean_selected_effect: float
    mean_best_candidate_effect: float
    mean_intervention_regret: float

    def to_dict(self) -> dict[str, int | float]:
        return asdict(self)


def evaluate_legal_random_interventions(
    config: GeneratedWorldDatasetConfig,
    *,
    samples: int,
    candidates_per_sample: int,
) -> RandomInterventionMetrics:
    """Replay validation worlds and score deterministic random legal actions."""

    if samples <= 0 or candidates_per_sample < 2:
        raise ValueError("random baseline requires samples and at least two candidates")
    pipeline = WorldGenerationPipeline(config)
    mechanisms = MechanismGenerator()
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
    for sample_index in range(samples):
        metadata = pipeline.materialize(
            SplitName.VALIDATION,
            sample_index * config.views_per_mechanism,
        ).metadata
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
            )
            world.reset(metadata.generation_seed)
            action_rng = np.random.default_rng(
                metadata.generation_seed + 10_000_019 + candidate
            )
            action = world.sample_action(action_rng)
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
        raise RuntimeError("random intervention baseline produced non-finite effects")
    return RandomInterventionMetrics(
        samples=samples,
        candidates_per_sample=candidates_per_sample,
        legal_action_rate=legal_actions / total_actions,
        mean_selected_effect=float(selected.mean()),
        mean_best_candidate_effect=float(best.mean()),
        mean_intervention_regret=float((best - selected).mean()),
    )
