"""Deterministic numerical excitation policies for generated dynamic worlds."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from chimera.meta_world.generators.contracts import GeneratedWorld, WorldAction


@dataclass(frozen=True)
class SeededRandomPolicy:
    """Explicit name for the generator's existing seeded legal-action policy."""

    policy_id: str = "seeded_random"

    def sample_action(
        self,
        world: GeneratedWorld,
        rng: np.random.Generator,
        step: int,
    ) -> WorldAction:
        if step < 0:
            raise ValueError("probe step must be non-negative")
        return world.sample_action(rng)


@dataclass(frozen=True)
class SystemIdentificationProbePolicy:
    """Repeat zero, impulse, saturation, polarity and reversal probes."""

    policy_id: str = "deterministic_system_identification_probe_v1"

    @staticmethod
    def _pair(objects: int, block: int, *, alternate: bool) -> tuple[int, int]:
        source = (2 * block + int(alternate)) % objects
        target = (source + 1 + block % max(objects - 1, 1)) % objects
        if target == source:
            target = (source + 1) % objects
        return source, target

    def sample_action(
        self,
        world: GeneratedWorld,
        rng: np.random.Generator,
        step: int,
    ) -> WorldAction:
        del rng
        if step < 0:
            raise ValueError("probe step must be non-negative")
        objects = world.config.objects
        if objects <= 1:
            raise ValueError("system-identification probes require two objects")
        phase = step % 8
        block = step // 8
        alternate = phase >= 5
        source, target = self._pair(objects, block, alternate=alternate)
        if phase in {0, 4}:
            magnitude, control = 0.0, 0.0
        elif phase in {1, 5}:
            magnitude, control = 0.25, 0.0
        elif phase in {2, 6}:
            magnitude, control = 0.85, 1.0 if phase == 2 else -1.0
        else:
            source, target = target, source
            magnitude, control = 0.85, -1.0 if phase == 3 else 1.0
        return WorldAction(
            source=source,
            target=target,
            magnitude=magnitude,
            control=control,
        )


@dataclass(frozen=True)
class HybridProbePolicy:
    """Identify for a fixed prefix, then evaluate seeded random interventions."""

    probe_prefix_steps: int = 4

    def __post_init__(self) -> None:
        if self.probe_prefix_steps <= 0:
            raise ValueError("probe_prefix_steps must be positive")

    @property
    def policy_id(self) -> str:
        return f"probe_prefix_{self.probe_prefix_steps}_then_seeded_random_v1"

    def sample_action(
        self,
        world: GeneratedWorld,
        rng: np.random.Generator,
        step: int,
    ) -> WorldAction:
        if step < self.probe_prefix_steps:
            return SystemIdentificationProbePolicy().sample_action(world, rng, step)
        return world.sample_action(rng)
