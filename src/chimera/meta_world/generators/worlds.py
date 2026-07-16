"""Three numerical dynamic-world families sharing one hidden-law contract."""

from __future__ import annotations

import copy
from abc import ABC, abstractmethod

import numpy as np
from numpy.typing import NDArray

from chimera.meta_world.generators.contracts import (
    FloatArray,
    GeneratedWorld,
    MechanismConfig,
    WorldAction,
    WorldConfig,
    WorldFamily,
    WorldObservation,
    WorldTransition,
)
from chimera.meta_world.generators.fingerprints import numeric_sha256
from chimera.meta_world.generators.renderer import (
    ObservationRenderer,
    build_renderer_config,
)


def _readonly_float(values: NDArray[np.generic]) -> FloatArray:
    result = np.asarray(values, dtype=np.float32).copy()
    result.flags.writeable = False
    return result


def _sigmoid(values: NDArray[np.generic]) -> FloatArray:
    clipped = np.clip(values, -20.0, 20.0)
    return np.asarray(1.0 / (1.0 + np.exp(-clipped)), dtype=np.float32)


def _softmax(values: NDArray[np.generic]) -> FloatArray:
    shifted = values - np.max(values)
    exponential = np.exp(np.clip(shifted, -20.0, 20.0))
    return np.asarray(
        exponential / np.maximum(exponential.sum(), 1e-8), dtype=np.float32
    )


class _BaseWorld(ABC):
    """Shared deterministic reset, paired counterfactual and renderer handling."""

    def __init__(
        self,
        mechanism: MechanismConfig,
        config: WorldConfig,
        renderer: ObservationRenderer,
        *,
        independent_renderer_rng: bool = False,
    ) -> None:
        self.mechanism = mechanism
        self.config = config
        self.renderer_config = renderer.config
        self._renderer = renderer
        self._independent_renderer_rng = independent_renderer_rng
        self._state = config.initial_state.copy()
        self._rng: np.random.Generator | None = None
        self._renderer_rng: np.random.Generator | None = None
        self._relations = self._make_relations()

    def _make_relations(self) -> FloatArray:
        config = self.config
        mechanism = self.mechanism
        maximum_edge = max(float(config.edge_capacity.max()), 1e-8)
        relation = np.zeros((config.objects, config.objects, 4), dtype=np.float32)
        relation[..., 0] = config.topology
        relation[..., 1] = config.edge_capacity / maximum_edge
        relation[..., 2] = config.topology * (
            mechanism.positive_feedback - mechanism.negative_feedback
        )
        relation[..., 3] = config.topology * (
            mechanism.delay_steps / max(mechanism.delay_steps + 1, 1)
        )
        return relation

    def reset(self, seed: int) -> WorldObservation:
        if seed < 0:
            raise ValueError("seed must be non-negative")
        self._rng = np.random.default_rng(seed)
        self._renderer_rng = (
            np.random.default_rng(seed + 1_000_000_007)
            if self._independent_renderer_rng
            else self._rng
        )
        jitter = self._rng.normal(0.0, 0.01, size=self.config.initial_state.shape).astype(
            np.float32
        )
        self._state = self._clip_state(self.config.initial_state + jitter)
        return self._renderer.render(self._state, self._relations, self._renderer_rng)

    def sample_action(self, rng: np.random.Generator) -> WorldAction:
        objects = self.config.objects
        source = int(rng.integers(0, objects))
        target = int(rng.integers(0, objects - 1))
        if target >= source:
            target += 1
        return WorldAction(
            source=source,
            target=target,
            magnitude=float(rng.uniform(0.05, 1.0)),
            control=float(rng.uniform(-1.0, 1.0)),
        )

    def sample_latent_action(self, rng: np.random.Generator) -> WorldAction:
        """Sample a renderer-independent intervention in latent coordinates."""

        objects = self.config.objects
        source = int(rng.integers(0, objects))
        target = int(rng.integers(0, objects - 1))
        if target >= source:
            target += 1
        return WorldAction(
            source=source,
            target=target,
            magnitude=float(rng.uniform(0.05, 1.0)),
            control=float(rng.uniform(-1.0, 1.0)),
        )

    def render_action(self, action: WorldAction) -> WorldAction:
        return self._renderer.from_latent_action(action)

    def step(self, action: WorldAction) -> WorldTransition:
        if self._rng is None or self._renderer_rng is None:
            raise RuntimeError("reset must be called before step")
        if action.source == action.target:
            raise ValueError("source and target must differ")
        if not 0.0 <= action.magnitude <= 1.0 or not -1.0 <= action.control <= 1.0:
            raise ValueError("action magnitude or control is outside the legal range")
        latent_action = self._renderer.to_latent_action(action)
        event = np.zeros(self.config.objects, dtype=np.float32)
        active = self._rng.random(self.config.objects) < self.mechanism.event_rate
        event[active] = self._rng.normal(
            0.0, self.config.event_scale, size=int(active.sum())
        ).astype(np.float32)
        next_state, metrics = self._advance(self._state.copy(), latent_action, event)
        no_op = WorldAction(
            source=latent_action.source,
            target=latent_action.target,
            magnitude=0.0,
            control=0.0,
        )
        no_op_state, baseline_metrics = self._advance(
            self._state.copy(), no_op, event
        )
        self._state = self._clip_state(next_state)
        outcome = np.asarray(
            [metrics[0], metrics[1], metrics[2], metrics[0] - baseline_metrics[0]],
            dtype=np.float32,
        )
        renderer_state = copy.deepcopy(self._renderer_rng.bit_generator.state)
        factual_observation = self._renderer.render(
            self._state,
            self._relations,
            self._renderer_rng,
        )
        factual_renderer_state = copy.deepcopy(
            self._renderer_rng.bit_generator.state
        )
        self._renderer_rng.bit_generator.state = renderer_state
        no_op_observation = self._renderer.render(
            self._clip_state(no_op_state),
            self._relations,
            self._renderer_rng,
        )
        self._renderer_rng.bit_generator.state = factual_renderer_state
        return WorldTransition(
            action=action,
            observation=factual_observation,
            outcome=outcome,
            counterfactual_no_op_observation=no_op_observation,
        )

    def _clip_state(self, state: NDArray[np.generic]) -> FloatArray:
        clipped = np.asarray(state, dtype=np.float32).copy()
        upper = self.config.capacity * 1.5
        clipped[:, 0] = np.clip(clipped[:, 0], 0.0, upper)
        clipped[:, 1] = np.clip(clipped[:, 1], 0.0, upper)
        clipped[:, 2] = np.clip(clipped[:, 2], -2.0, 2.0)
        clipped[:, 3] = np.clip(clipped[:, 3], -2.0, 2.0)
        return clipped

    @abstractmethod
    def _advance(
        self,
        state: FloatArray,
        action: WorldAction,
        event: FloatArray,
    ) -> tuple[FloatArray, FloatArray]:
        """Return next latent state and [utility, throughput, constraint]."""


class FlowWorld(_BaseWorld):
    """Finite resource flow with loss, queues, feedback and bottlenecks."""

    def _advance(
        self,
        state: FloatArray,
        action: WorldAction,
        event: FloatArray,
    ) -> tuple[FloatArray, FloatArray]:
        config = self.config
        mechanism = self.mechanism
        resource = state[:, 0].copy()
        pending = state[:, 1].copy()
        attraction = state[:, 2].copy()
        hidden = state[:, 3].copy()
        edge_limit = float(config.edge_capacity[action.source, action.target])
        requested = action.magnitude * float(config.capacity[action.source])
        transferred = min(float(resource[action.source]), edge_limit, requested)
        retention = float(
            np.clip(mechanism.retention * (1.0 + 0.08 * action.control), 0.0, 1.0)
        )
        delivered = transferred * retention
        delay_fraction = mechanism.delay_steps / max(mechanism.delay_steps + 1, 1)
        resource[action.source] -= transferred
        resource[action.target] += delivered * (1.0 - delay_fraction)
        pending[action.target] += delivered * delay_fraction

        release_rate = 1.0 / (mechanism.delay_steps + 1.0)
        released = pending * release_rate
        pending -= released
        resource += released
        utilization = resource / np.maximum(config.capacity, 1e-6)
        pressure = attraction[None, :] - attraction[:, None]
        propensity = config.topology * _sigmoid(mechanism.nonlinearity * pressure)
        network_flow = propensity * utilization[:, None] * config.rates[:, 0, None]
        network_flow = np.minimum(network_flow, config.edge_capacity)
        outbound = network_flow.sum(axis=1)
        inbound = network_flow.sum(axis=0) * mechanism.retention
        resource += inbound - np.minimum(outbound, resource)
        threshold_signal = np.tanh(utilization - mechanism.threshold)
        attraction += (
            mechanism.positive_feedback * threshold_signal
            - mechanism.negative_feedback * (attraction - config.rates[:, 2])
            + mechanism.hidden_coupling * hidden
        )
        hidden += mechanism.interaction * (
            config.topology.T @ utilization - config.topology @ utilization
        ) + event
        resource += event * config.capacity
        overflow = np.maximum(resource - config.capacity, 0.0)
        utility = float(
            np.sum(resource * (0.5 + 0.5 * np.tanh(attraction)))
            / np.maximum(config.capacity.sum(), 1e-6)
            - mechanism.saturation * overflow.sum() / np.maximum(config.capacity.sum(), 1e-6)
        )
        throughput = float(transferred + network_flow.sum())
        constraint = float(np.max(resource / np.maximum(config.capacity, 1e-6)))
        next_state = np.stack([resource, pending, attraction, hidden], axis=1)
        return self._clip_state(next_state), np.asarray(
            [utility, throughput, constraint], dtype=np.float32
        )


class CompetitionWorld(_BaseWorld):
    """Agents compete and sometimes cooperate for one capacity-limited pool."""

    def _advance(
        self,
        state: FloatArray,
        action: WorldAction,
        event: FloatArray,
    ) -> tuple[FloatArray, FloatArray]:
        config = self.config
        mechanism = self.mechanism
        allocation = state[:, 0].copy()
        momentum = state[:, 1].copy()
        demand = state[:, 2].copy()
        hidden = state[:, 3].copy()
        intervention = action.magnitude * float(config.capacity[action.source])
        momentum[action.source] += intervention * (0.75 + 0.25 * action.control)
        momentum[action.target] -= intervention * (0.35 - 0.20 * action.control)
        normalized = allocation / np.maximum(config.capacity, 1e-6)
        rival_pressure = config.topology @ normalized
        scores = (
            momentum
            + mechanism.positive_feedback * normalized
            - mechanism.competition * rival_pressure
            + mechanism.hidden_coupling * hidden
        )
        shares = _softmax(scores)
        pool = float(config.capacity.sum() * 0.72)
        target_allocation = np.minimum(shares * pool, config.capacity)
        blend = np.clip(config.rates[:, 0] + 0.15 * mechanism.nonlinearity, 0.05, 0.65)
        allocation = (1.0 - blend) * allocation + blend * target_allocation
        momentum = (
            mechanism.retention * momentum
            - mechanism.negative_feedback * normalized
            + mechanism.interaction * (config.topology.T @ shares - rival_pressure)
        )
        demand += config.rates[:, 1] * (
            _sigmoid(hidden + event) - _sigmoid(demand)
        )
        hidden += mechanism.hidden_coupling * (shares - shares.mean()) + event
        served = np.minimum(allocation, config.capacity * _sigmoid(demand))
        throughput = float(served.sum())
        concentration = float(np.square(shares).sum())
        utility = float(
            throughput / np.maximum(config.capacity.sum(), 1e-6)
            - mechanism.competition * concentration
            + 0.05 * action.control * action.magnitude
        )
        next_state = np.stack([allocation, momentum, demand, hidden], axis=1)
        return self._clip_state(next_state), np.asarray(
            [utility, throughput, concentration], dtype=np.float32
        )


class FunnelWorld(_BaseWorld):
    """Sequential stages with conversion, returns, queues and delayed effects."""

    def _advance(
        self,
        state: FloatArray,
        action: WorldAction,
        event: FloatArray,
    ) -> tuple[FloatArray, FloatArray]:
        config = self.config
        mechanism = self.mechanism
        queue = state[:, 0].copy()
        delayed = state[:, 1].copy()
        conversion = state[:, 2].copy()
        hidden = state[:, 3].copy()
        queue[action.source] += action.magnitude * float(config.capacity[action.source])
        conversion[action.target] += 0.20 * action.control * action.magnitude
        flow = np.zeros_like(config.topology)
        for source in range(config.objects):
            destinations = np.flatnonzero(config.topology[source] > 0.0)
            if not destinations.size:
                continue
            available = float(queue[source])
            weights = config.topology[source, destinations]
            weights = weights / np.maximum(weights.sum(), 1e-8)
            for destination, weight in zip(destinations.tolist(), weights.tolist(), strict=True):
                candidate = (
                    available
                    * float(config.rates[source, 0])
                    * float(weight)
                    * float(_sigmoid(np.asarray(conversion[destination])))
                )
                flow[source, destination] = min(
                    candidate,
                    float(config.edge_capacity[source, destination]),
                )
            queue[source] -= min(float(flow[source].sum()), available)
        incoming = flow.sum(axis=0) * mechanism.retention
        delay_fraction = mechanism.delay_steps / max(mechanism.delay_steps + 1, 1)
        delayed += incoming * delay_fraction
        queue += incoming * (1.0 - delay_fraction)
        released = delayed / (mechanism.delay_steps + 1.0)
        delayed -= released
        queue += released
        completion = min(
            float(queue[-1]) * float(config.rates[-1, 1]),
            float(config.capacity[-1]) * 0.30,
        )
        queue[-1] -= completion
        utilization = queue / np.maximum(config.capacity, 1e-6)
        conversion += (
            mechanism.positive_feedback * np.tanh(flow.sum(axis=0))
            - mechanism.negative_feedback * (conversion - config.rates[:, 2])
            + mechanism.hidden_coupling * hidden
        )
        hidden += mechanism.interaction * (utilization - utilization.mean()) + event
        queue += np.maximum(event, 0.0) * config.capacity
        backlog = float(np.mean(utilization))
        utility = float(
            completion / np.maximum(config.capacity[-1], 1e-6)
            - mechanism.saturation * max(backlog - mechanism.threshold, 0.0)
        )
        throughput = float(flow.sum() + completion)
        next_state = np.stack([queue, delayed, conversion, hidden], axis=1)
        return self._clip_state(next_state), np.asarray(
            [utility, throughput, backlog], dtype=np.float32
        )


class WorldGenerator:
    """Map one hidden mechanism into a concrete family and observation renderer."""

    def __init__(
        self,
        *,
        min_objects: int = 4,
        max_objects: int = 10,
        observation_features: int = 8,
        relation_features: int = 4,
    ) -> None:
        if min_objects <= 1 or max_objects < min_objects:
            raise ValueError("invalid world object range")
        if observation_features < 4 or relation_features != 4:
            raise ValueError("generated worlds require four state and relation channels")
        self.min_objects = min_objects
        self.max_objects = max_objects
        self.observation_features = observation_features
        self.relation_features = relation_features

    def generate(
        self,
        mechanism: MechanismConfig,
        family_id: WorldFamily,
        *,
        world_seed: int,
        renderer_seed: int,
        renderer_profile: int,
        independent_renderer_rng: bool = False,
    ) -> GeneratedWorld:
        if world_seed < 0 or renderer_seed < 0:
            raise ValueError("world and renderer seeds must be non-negative")
        config = self._world_config(mechanism, family_id, world_seed)
        renderer_config = build_renderer_config(
            objects=config.objects,
            hidden_features=4,
            relation_features=self.relation_features,
            output_features=self.observation_features,
            profile_id=renderer_profile,
            seed=renderer_seed,
        )
        renderer = ObservationRenderer(renderer_config, self.observation_features)
        world_types: dict[WorldFamily, type[_BaseWorld]] = {
            WorldFamily.FLOW: FlowWorld,
            WorldFamily.COMPETITION: CompetitionWorld,
            WorldFamily.FUNNEL: FunnelWorld,
        }
        return world_types[family_id](
            mechanism,
            config,
            renderer,
            independent_renderer_rng=independent_renderer_rng,
        )

    def _world_config(
        self,
        mechanism: MechanismConfig,
        family_id: WorldFamily,
        seed: int,
    ) -> WorldConfig:
        rng = np.random.default_rng(seed)
        objects = int(rng.integers(self.min_objects, self.max_objects + 1))
        capacity = rng.uniform(0.8, 2.0, size=objects).astype(np.float32)
        topology = np.zeros((objects, objects), dtype=np.float32)
        if family_id is WorldFamily.FLOW:
            raw = rng.uniform(0.15, 1.0, size=(objects, objects)).astype(np.float32)
            topology[:, :] = np.where(
                rng.random((objects, objects)) > 0.55, raw, 0.0
            )
            np.fill_diagonal(topology, 0.0)
            for source in range(objects):
                topology[source, (source + 1) % objects] = max(
                    float(topology[source, (source + 1) % objects]), 0.35
                )
        elif family_id is WorldFamily.COMPETITION:
            raw = rng.uniform(0.2, 1.0, size=(objects, objects)).astype(np.float32)
            topology[:, :] = (0.5 * (raw + raw.T)).astype(np.float32)
            np.fill_diagonal(topology, 0.0)
        else:
            for source in range(objects - 1):
                topology[source, source + 1] = float(rng.uniform(0.65, 1.0))
                if source > 0 and rng.random() < 0.55:
                    topology[source, source - 1] = float(rng.uniform(0.05, 0.30))
                if source + 2 < objects and rng.random() < 0.35:
                    topology[source, source + 2] = float(rng.uniform(0.05, 0.25))
        edge_scale = rng.uniform(0.15, 0.75, size=(objects, objects)).astype(np.float32)
        edge_capacity = topology * edge_scale * capacity[:, None]
        rates = rng.uniform(0.05, 0.40, size=(objects, 4)).astype(np.float32)
        initial_state = np.zeros((objects, 4), dtype=np.float32)
        initial_state[:, 0] = rng.uniform(0.15, 0.70, size=objects) * capacity
        initial_state[:, 1] = rng.uniform(0.0, 0.20, size=objects) * capacity
        initial_state[:, 2] = rng.uniform(0.15, 0.85, size=objects)
        initial_state[:, 3] = rng.normal(0.0, 0.08, size=objects)
        event_scale = float(rng.uniform(0.01, 0.05))
        config_hash = numeric_sha256(
            "chimera-world-v1",
            (int(family_id), objects, event_scale),
            (capacity, topology, edge_capacity, rates, initial_state),
        )
        return WorldConfig(
            world_instance_id=f"CHM-W-I-{config_hash[:20]}",
            family_id=family_id,
            objects=objects,
            capacity=_readonly_float(capacity),
            topology=_readonly_float(topology),
            edge_capacity=_readonly_float(edge_capacity),
            rates=_readonly_float(rates),
            initial_state=_readonly_float(initial_state),
            event_scale=event_scale,
        )
