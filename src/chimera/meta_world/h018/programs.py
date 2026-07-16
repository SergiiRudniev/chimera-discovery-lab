"""Seeded hidden mechanism programs for H018 compositional transfer."""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum

import numpy as np
from numpy.typing import NDArray

from chimera.meta_world.generators import MechanismConfig
from chimera.meta_world.generators.fingerprints import numeric_sha256


class MechanismOperator(IntEnum):
    """Evaluator-only operator identities; none enter the model batch."""

    RETENTION_LOSS = 0
    THRESHOLD_RESPONSE = 1
    DELAYED_FEEDBACK = 2
    SATURATION = 3
    COMPETITION = 4
    HIDDEN_COUPLING = 5
    EXOGENOUS_EVENTS = 6


@dataclass(frozen=True)
class MechanismProgramSpec:
    """One immutable composition of dynamic operators."""

    program_id: int
    operators: tuple[MechanismOperator, ...]

    def __post_init__(self) -> None:
        if self.program_id < 0 or len(self.operators) < 2:
            raise ValueError("mechanism programs require an ID and two operators")
        if len(set(self.operators)) != len(self.operators):
            raise ValueError("mechanism program operators must be unique")


_PROGRAM_SPECS = (
    MechanismProgramSpec(
        0, (MechanismOperator.RETENTION_LOSS, MechanismOperator.THRESHOLD_RESPONSE)
    ),
    MechanismProgramSpec(
        1, (MechanismOperator.RETENTION_LOSS, MechanismOperator.DELAYED_FEEDBACK)
    ),
    MechanismProgramSpec(
        2, (MechanismOperator.THRESHOLD_RESPONSE, MechanismOperator.SATURATION)
    ),
    MechanismProgramSpec(
        3, (MechanismOperator.DELAYED_FEEDBACK, MechanismOperator.COMPETITION)
    ),
    MechanismProgramSpec(
        4, (MechanismOperator.SATURATION, MechanismOperator.HIDDEN_COUPLING)
    ),
    MechanismProgramSpec(
        5, (MechanismOperator.COMPETITION, MechanismOperator.EXOGENOUS_EVENTS)
    ),
    MechanismProgramSpec(
        6, (MechanismOperator.RETENTION_LOSS, MechanismOperator.COMPETITION)
    ),
    MechanismProgramSpec(
        7, (MechanismOperator.THRESHOLD_RESPONSE, MechanismOperator.HIDDEN_COUPLING)
    ),
    MechanismProgramSpec(
        8, (MechanismOperator.DELAYED_FEEDBACK, MechanismOperator.EXOGENOUS_EVENTS)
    ),
    MechanismProgramSpec(
        9,
        (
            MechanismOperator.RETENTION_LOSS,
            MechanismOperator.SATURATION,
            MechanismOperator.EXOGENOUS_EVENTS,
        ),
    ),
    MechanismProgramSpec(
        10,
        (
            MechanismOperator.THRESHOLD_RESPONSE,
            MechanismOperator.DELAYED_FEEDBACK,
            MechanismOperator.COMPETITION,
        ),
    ),
)

PROGRAM_SPECS = {spec.program_id: spec for spec in _PROGRAM_SPECS}
TRAIN_PROGRAM_IDS = frozenset(range(6))
TRANSFER_PROGRAM_IDS = frozenset({6, 7, 8})
MECHANISM_TEST_PROGRAM_IDS = frozenset({9, 10})


def _readonly_float(values: NDArray[np.generic]) -> NDArray[np.float32]:
    result = np.asarray(values, dtype=np.float32).copy()
    result.flags.writeable = False
    return result


def operator_ids(program_ids: frozenset[int] | set[int]) -> frozenset[int]:
    """Return the primitive support of a registered program set."""

    return frozenset(
        int(operator)
        for program_id in program_ids
        for operator in PROGRAM_SPECS[program_id].operators
    )


def program_catalogue() -> dict[str, list[int]]:
    """Serializable evaluator catalogue for deterministic manifests."""

    return {
        str(program_id): [int(operator) for operator in spec.operators]
        for program_id, spec in sorted(PROGRAM_SPECS.items())
    }


class MechanismProgramGenerator:
    """Compile held-out compositions into family-agnostic numerical laws."""

    def generate(self, template_id: int, seed: int) -> MechanismConfig:
        if template_id not in PROGRAM_SPECS:
            raise ValueError("unknown H018 mechanism program")
        if seed < 0:
            raise ValueError("seed must be non-negative")
        spec = PROGRAM_SPECS[template_id]
        rng = np.random.default_rng(seed)
        active = set(spec.operators)

        retention = 0.965
        nonlinearity = 0.12
        threshold = 0.50
        delay_steps = 0
        positive_feedback = 0.08
        negative_feedback = 0.05
        saturation = 1.28
        competition = 0.06
        interaction = 0.10
        hidden_coupling = 0.06
        event_rate = 0.02

        if MechanismOperator.RETENTION_LOSS in active:
            retention -= 0.105
            negative_feedback += 0.12
        if MechanismOperator.THRESHOLD_RESPONSE in active:
            nonlinearity += 0.43
            threshold -= 0.14
        if MechanismOperator.DELAYED_FEEDBACK in active:
            delay_steps = 2
            positive_feedback += 0.25
            negative_feedback += 0.06
        if MechanismOperator.SATURATION in active:
            saturation -= 0.48
            nonlinearity += 0.13
        if MechanismOperator.COMPETITION in active:
            competition += 0.43
            negative_feedback += 0.09
        if MechanismOperator.HIDDEN_COUPLING in active:
            interaction += 0.33
            hidden_coupling += 0.36
        if MechanismOperator.EXOGENOUS_EVENTS in active:
            event_rate += 0.105
            positive_feedback += 0.07

        if {
            MechanismOperator.THRESHOLD_RESPONSE,
            MechanismOperator.SATURATION,
        } <= active:
            nonlinearity += 0.10
        if {
            MechanismOperator.DELAYED_FEEDBACK,
            MechanismOperator.EXOGENOUS_EVENTS,
        } <= active:
            delay_steps = 3
            positive_feedback += 0.08
        if {
            MechanismOperator.COMPETITION,
            MechanismOperator.HIDDEN_COUPLING,
        } <= active:
            interaction += 0.10

        jitter = rng.uniform(-1.0, 1.0, size=10)
        retention = float(np.clip(retention + 0.012 * jitter[0], 0.75, 0.995))
        nonlinearity = float(np.clip(nonlinearity + 0.025 * jitter[1], 0.05, 0.95))
        threshold = float(np.clip(threshold + 0.03 * jitter[2], 0.10, 0.90))
        positive_feedback = float(
            np.clip(positive_feedback + 0.02 * jitter[3], 0.02, 0.60)
        )
        negative_feedback = float(
            np.clip(negative_feedback + 0.02 * jitter[4], 0.01, 0.60)
        )
        saturation = float(np.clip(saturation + 0.035 * jitter[5], 0.50, 1.50))
        competition = float(np.clip(competition + 0.025 * jitter[6], 0.02, 0.80))
        interaction = float(np.clip(interaction + 0.025 * jitter[7], 0.05, 0.80))
        hidden_coupling = float(
            np.clip(hidden_coupling + 0.025 * jitter[8], 0.02, 0.70)
        )
        event_rate = float(np.clip(event_rate + 0.01 * jitter[9], 0.01, 0.20))
        weights = rng.normal(0.0, 1.0, size=4)
        weights /= max(float(np.linalg.norm(weights)), 1e-8)
        latent_weights = _readonly_float(weights)
        signature = tuple(int(operator) for operator in spec.operators)
        config_hash = numeric_sha256(
            "chimera-compositional-mechanism-v1",
            (
                template_id,
                *signature,
                retention,
                nonlinearity,
                threshold,
                delay_steps,
                positive_feedback,
                negative_feedback,
                saturation,
                competition,
                interaction,
                hidden_coupling,
                event_rate,
            ),
            (latent_weights,),
        )
        return MechanismConfig(
            mechanism_id=f"CHM-W-CM-{config_hash[:20]}",
            template_id=template_id,
            retention=retention,
            nonlinearity=nonlinearity,
            threshold=threshold,
            delay_steps=delay_steps,
            positive_feedback=positive_feedback,
            negative_feedback=negative_feedback,
            saturation=saturation,
            competition=competition,
            interaction=interaction,
            hidden_coupling=hidden_coupling,
            event_rate=event_rate,
            latent_weights=latent_weights,
        )
