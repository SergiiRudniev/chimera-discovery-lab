"""Stable SHA-256 fingerprints for generated numerical configurations."""

from __future__ import annotations

import hashlib
from collections.abc import Iterable

import numpy as np
from numpy.typing import NDArray

from chimera.meta_world.generators.contracts import (
    MechanismConfig,
    RendererConfig,
    WorldConfig,
)


def numeric_sha256(
    tag: str,
    scalars: Iterable[int | float | str],
    arrays: Iterable[NDArray[np.generic]],
) -> str:
    """Hash typed scalars and arrays without relying on Python object serialization."""

    digest = hashlib.sha256(tag.encode("utf-8"))
    for scalar in scalars:
        digest.update(type(scalar).__name__.encode("ascii"))
        digest.update(b":")
        digest.update(repr(scalar).encode("utf-8"))
        digest.update(b"\0")
    for array in arrays:
        contiguous = np.ascontiguousarray(array)
        digest.update(contiguous.dtype.str.encode("ascii"))
        digest.update(repr(tuple(contiguous.shape)).encode("ascii"))
        digest.update(contiguous.tobytes())
    return digest.hexdigest()


def mechanism_config_sha256(config: MechanismConfig) -> str:
    return numeric_sha256(
        "chimera-mechanism-v1",
        (
            config.template_id,
            config.retention,
            config.nonlinearity,
            config.threshold,
            config.delay_steps,
            config.positive_feedback,
            config.negative_feedback,
            config.saturation,
            config.competition,
            config.interaction,
            config.hidden_coupling,
            config.event_rate,
        ),
        (config.latent_weights,),
    )


def world_config_sha256(config: WorldConfig) -> str:
    return numeric_sha256(
        "chimera-world-v1",
        (int(config.family_id), config.objects, config.event_scale),
        (
            config.capacity,
            config.topology,
            config.edge_capacity,
            config.rates,
            config.initial_state,
        ),
    )


def renderer_config_sha256(config: RendererConfig) -> str:
    return numeric_sha256(
        "chimera-renderer-v1",
        (
            config.profile_id,
            config.nonlinear_kind,
            config.noise_std,
            config.nuisance_features,
            config.time_scale,
        ),
        (
            config.object_permutation,
            config.feature_permutation,
            config.relation_permutation,
            config.feature_scale,
            config.feature_offset,
            config.visibility,
        ),
    )

