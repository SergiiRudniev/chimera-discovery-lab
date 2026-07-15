"""Observation rendering that changes representation but not hidden dynamics."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from chimera.meta_world.generators.contracts import (
    BoolArray,
    FloatArray,
    IntArray,
    RendererConfig,
    WorldAction,
    WorldObservation,
)
from chimera.meta_world.generators.fingerprints import numeric_sha256


def _readonly_int(values: NDArray[np.generic]) -> IntArray:
    result = np.asarray(values, dtype=np.int64).copy()
    result.flags.writeable = False
    return result


def _readonly_float(values: NDArray[np.generic]) -> FloatArray:
    result = np.asarray(values, dtype=np.float32).copy()
    result.flags.writeable = False
    return result


def _readonly_bool(values: NDArray[np.generic]) -> BoolArray:
    result = np.asarray(values, dtype=np.bool_).copy()
    result.flags.writeable = False
    return result


def build_renderer_config(
    *,
    objects: int,
    hidden_features: int,
    relation_features: int,
    output_features: int,
    profile_id: int,
    seed: int,
) -> RendererConfig:
    """Generate one deterministic observation transform from a registered profile."""

    if objects <= 1 or hidden_features <= 0 or relation_features <= 0:
        raise ValueError("renderer dimensions must be positive")
    if output_features < hidden_features or profile_id not in {0, 1, 2}:
        raise ValueError("invalid output feature count or renderer profile")
    if seed < 0:
        raise ValueError("seed must be non-negative")
    rng = np.random.default_rng(seed)
    object_permutation = _readonly_int(rng.permutation(objects))
    feature_permutation = _readonly_int(rng.permutation(hidden_features))
    relation_permutation = _readonly_int(rng.permutation(relation_features))
    if profile_id == 0:
        nonlinear_kind = int(seed % 2)
        scale_range = (0.75, 1.35)
        offset_scale = 0.08
        visibility_rate = 0.96
        noise_std = 0.005
        nuisance_features = min(1, output_features - hidden_features)
        time_range = (0.80, 1.25)
    elif profile_id == 1:
        nonlinear_kind = 1
        scale_range = (0.35, 2.25)
        offset_scale = 0.20
        visibility_rate = 0.88
        noise_std = 0.015
        nuisance_features = min(2, output_features - hidden_features)
        time_range = (0.50, 2.00)
    else:
        nonlinear_kind = 2
        scale_range = (0.15, 3.75)
        offset_scale = 0.35
        visibility_rate = 0.72
        noise_std = 0.030
        nuisance_features = min(4, output_features - hidden_features)
        time_range = (0.25, 4.00)
    feature_scale = _readonly_float(rng.uniform(*scale_range, size=hidden_features))
    feature_offset = _readonly_float(
        rng.normal(0.0, offset_scale, size=hidden_features)
    )
    visibility = rng.random((objects, hidden_features)) < visibility_rate
    visibility[:, 0] = True
    visibility = _readonly_bool(visibility)
    time_scale = float(rng.uniform(*time_range))
    config_hash = numeric_sha256(
        "chimera-renderer-v1",
        (
            profile_id,
            nonlinear_kind,
            noise_std,
            nuisance_features,
            time_scale,
        ),
        (
            object_permutation,
            feature_permutation,
            relation_permutation,
            feature_scale,
            feature_offset,
            visibility,
        ),
    )
    return RendererConfig(
        renderer_id=f"CHM-W-R-{config_hash[:20]}",
        profile_id=profile_id,
        object_permutation=object_permutation.astype(np.int64, copy=False),
        feature_permutation=feature_permutation.astype(np.int64, copy=False),
        relation_permutation=relation_permutation.astype(np.int64, copy=False),
        feature_scale=feature_scale.astype(np.float32, copy=False),
        feature_offset=feature_offset.astype(np.float32, copy=False),
        visibility=visibility.astype(np.bool_, copy=False),
        nonlinear_kind=nonlinear_kind,
        noise_std=noise_std,
        nuisance_features=nuisance_features,
        time_scale=time_scale,
    )


class ObservationRenderer:
    """Render hidden states under object, channel, unit and time transformations."""

    def __init__(self, config: RendererConfig, output_features: int) -> None:
        hidden_features = int(config.feature_permutation.size)
        if output_features < hidden_features + config.nuisance_features:
            raise ValueError("output_features cannot hold the configured renderer channels")
        self.config = config
        self.output_features = output_features

    def to_latent_action(self, action: WorldAction) -> WorldAction:
        objects = int(self.config.object_permutation.size)
        if not 0 <= action.source < objects or not 0 <= action.target < objects:
            raise ValueError("action object pointer is outside the rendered world")
        return WorldAction(
            source=int(self.config.object_permutation[action.source]),
            target=int(self.config.object_permutation[action.target]),
            magnitude=action.magnitude,
            control=action.control,
        )

    def action_targets(self, action: WorldAction) -> FloatArray:
        objects = int(self.config.object_permutation.size)
        targets = np.zeros(objects, dtype=np.float32)
        targets[action.source] = -1.0
        targets[action.target] = 1.0
        return targets

    def render(
        self,
        state: FloatArray,
        relations: FloatArray,
        rng: np.random.Generator,
    ) -> WorldObservation:
        config = self.config
        objects = state.shape[0]
        hidden_features = state.shape[1]
        if tuple(state.shape) != (objects, config.feature_permutation.size):
            raise ValueError("state shape does not match renderer configuration")
        if relations.ndim != 3 or tuple(relations.shape[:2]) != (objects, objects):
            raise ValueError("relation shape does not match renderer configuration")
        permutation = config.object_permutation
        rendered = state[permutation][:, config.feature_permutation].astype(
            np.float32, copy=True
        )
        rendered = rendered * config.feature_scale + config.feature_offset
        if config.nonlinear_kind == 1:
            rendered = np.arcsinh(rendered).astype(np.float32)
        elif config.nonlinear_kind == 2:
            rendered = (
                np.sign(rendered) * np.sqrt(np.abs(rendered) + np.float32(1e-6))
            ).astype(np.float32)
        if config.noise_std:
            rendered += rng.normal(0.0, config.noise_std, size=rendered.shape).astype(
                np.float32
            )
        rendered = np.where(config.visibility, rendered, 0.0).astype(np.float32)
        values = np.zeros((objects, self.output_features), dtype=np.float32)
        values[:, :hidden_features] = rendered
        start = hidden_features
        stop = start + config.nuisance_features
        if stop > start:
            nuisance = rng.normal(0.0, 1.0, size=(objects, stop - start)).astype(np.float32)
            values[:, start:stop] = nuisance

        relation_mask = np.any(np.abs(relations) > 1e-8, axis=-1)
        rendered_relations = relations[permutation][:, permutation]
        rendered_relations = rendered_relations[:, :, config.relation_permutation].astype(
            np.float32, copy=True
        )
        rendered_relation_mask = relation_mask[permutation][:, permutation].astype(
            np.bool_, copy=True
        )
        return WorldObservation(
            values=values,
            object_mask=np.ones(objects, dtype=np.bool_),
            relations=rendered_relations,
            relation_mask=rendered_relation_mask,
            delta_time=config.time_scale,
        )
