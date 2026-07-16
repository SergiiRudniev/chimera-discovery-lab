"""Family-agnostic hidden mechanism generation."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from chimera.meta_world.generators.contracts import FloatArray, MechanismConfig
from chimera.meta_world.generators.fingerprints import numeric_sha256

_TEMPLATE_COUNT = 6


def _readonly_float(values: NDArray[np.generic]) -> FloatArray:
    result = np.asarray(values, dtype=np.float32).copy()
    result.flags.writeable = False
    return result


class MechanismGenerator:
    """Sample hidden dynamic laws from six registered mechanism templates."""

    def generate(self, template_id: int, seed: int) -> MechanismConfig:
        if not 0 <= template_id < _TEMPLATE_COUNT:
            raise ValueError(f"template_id must be in [0, {_TEMPLATE_COUNT})")
        if seed < 0:
            raise ValueError("seed must be non-negative")
        rng = np.random.default_rng(seed)
        base = np.asarray(
            [
                [0.96, 0.18, 0.40, 0, 0.18, 0.05, 1.20, 0.10, 0.24, 0.10, 0.04],
                [0.91, 0.42, 0.55, 1, 0.32, 0.12, 1.05, 0.18, 0.30, 0.18, 0.07],
                [0.87, 0.35, 0.32, 2, 0.14, 0.30, 0.92, 0.44, 0.22, 0.20, 0.09],
                [0.94, 0.62, 0.68, 1, 0.38, 0.16, 0.80, 0.26, 0.48, 0.26, 0.06],
                [0.82, 0.78, 0.28, 3, 0.45, 0.08, 0.72, 0.58, 0.36, 0.42, 0.12],
                [0.89, 0.52, 0.76, 2, 0.10, 0.42, 0.66, 0.34, 0.62, 0.38, 0.14],
            ],
            dtype=np.float64,
        )[template_id]
        jitter = rng.uniform(-1.0, 1.0, size=base.shape)
        retention = float(np.clip(base[0] + 0.015 * jitter[0], 0.75, 0.995))
        nonlinearity = float(np.clip(base[1] + 0.04 * jitter[1], 0.05, 0.95))
        threshold = float(np.clip(base[2] + 0.04 * jitter[2], 0.10, 0.90))
        delay_steps = int(base[3])
        positive_feedback = float(np.clip(base[4] + 0.03 * jitter[4], 0.02, 0.60))
        negative_feedback = float(np.clip(base[5] + 0.03 * jitter[5], 0.01, 0.60))
        saturation = float(np.clip(base[6] + 0.05 * jitter[6], 0.50, 1.50))
        competition = float(np.clip(base[7] + 0.04 * jitter[7], 0.02, 0.80))
        interaction = float(np.clip(base[8] + 0.04 * jitter[8], 0.05, 0.80))
        hidden_coupling = float(np.clip(base[9] + 0.04 * jitter[9], 0.02, 0.70))
        event_rate = float(np.clip(base[10] + 0.015 * jitter[10], 0.01, 0.20))
        weights = rng.normal(0.0, 1.0, size=4)
        weights /= max(float(np.linalg.norm(weights)), 1e-8)
        latent_weights = _readonly_float(weights)
        config_hash = numeric_sha256(
            "chimera-mechanism-v1",
            (
                template_id,
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
            mechanism_id=f"CHM-W-M-{config_hash[:20]}",
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
