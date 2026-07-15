"""Deterministic mechanistic systems used only for W0 engineering validation."""

from __future__ import annotations

import torch
from torch import Tensor

from chimera.meta_world.batch import MetaWorldBatch
from chimera.meta_world.config import MetaWorldModelConfig


def _domain_transform(domain_id: int, features: int, seed: int) -> tuple[Tensor, Tensor]:
    generator = torch.Generator(device="cpu").manual_seed(seed + 10_000 * (domain_id + 1))
    matrix = torch.randn(features, features, generator=generator)
    orthogonal, _ = torch.linalg.qr(matrix)
    scale = torch.linspace(0.8, 1.2, features)
    transform = orthogonal * scale.unsqueeze(0)
    bias = torch.randn(features, generator=generator) * 0.05
    return transform, bias


def _observe(state: Tensor, transform: Tensor, bias: Tensor) -> Tensor:
    return torch.tanh(state @ transform + bias)


def _make_relations(
    active_slots: int, relation_features: int, generator: torch.Generator
) -> Tensor:
    relations = torch.zeros(active_slots, active_slots, relation_features)
    raw = torch.rand(active_slots, active_slots, generator=generator)
    adjacency = torch.where(raw > 0.55, raw, torch.zeros_like(raw))
    adjacency.fill_diagonal_(0.0)
    rows_without_edges = adjacency.sum(dim=1) == 0
    for row in torch.where(rows_without_edges)[0].tolist():
        adjacency[row, (row + 1) % active_slots] = 0.5
    adjacency = adjacency / adjacency.sum(dim=1, keepdim=True).clamp_min(1e-6)
    signs = torch.where(
        torch.rand(active_slots, active_slots, generator=generator) > 0.25,
        torch.ones_like(adjacency),
        -torch.ones_like(adjacency),
    )
    relations[..., 0] = adjacency
    if relation_features > 1:
        relations[..., 1] = adjacency * signs
    if relation_features > 2:
        relations[..., 2] = adjacency * (
            0.5 + torch.rand(active_slots, active_slots, generator=generator)
        )
    if relation_features > 3:
        relations[..., 3] = adjacency * torch.rand(
            active_slots, active_slots, generator=generator
        )
    return relations


def _evolve(state: Tensor, relations: Tensor, mechanism_id: int, era: int = 0) -> Tensor:
    adjacency = relations[..., 0]
    signed = relations[..., 1] if relations.shape[-1] > 1 else adjacency
    incoming = signed.transpose(0, 1) @ state
    if mechanism_id == 0:
        delta = 0.12 * (incoming - state)
    elif mechanism_id == 1:
        delta = 0.08 * torch.tanh(incoming) + 0.04 * state - 0.025 * state.pow(3)
    elif mechanism_id == 2:
        competition = adjacency.transpose(0, 1) @ state.abs()
        signed_feedback = signed.transpose(0, 1) @ state
        delta = (
            0.06 * state * (1.0 - competition)
            + 0.02 * torch.tanh(signed_feedback)
        )
    else:
        if relations.shape[-1] > 2:
            capacity = relations[..., 2].sum(dim=0).unsqueeze(-1).clamp_min(0.25)
        else:
            capacity = torch.ones(state.shape[0], 1)
        overflow = torch.relu(state.abs() - capacity) * torch.sign(state)
        delta = 0.10 * incoming - 0.08 * overflow
    drift = 1.0 + 0.015 * era
    return (state + drift * delta).clamp(-2.0, 2.0)


def _apply_intervention(
    state: Tensor,
    relations: Tensor,
    intervention_type: int,
    source: int,
    target: int,
    parameters: Tensor,
) -> tuple[Tensor, Tensor, float]:
    controls = torch.zeros(8, dtype=parameters.dtype, device=parameters.device)
    controls[: min(parameters.numel(), 8)] = parameters[:8]
    changed_state = state.clone()
    changed_relations = relations.clone()
    amplitude = float(0.05 + 0.20 * torch.sigmoid(controls[0]))
    feature = min(
        int(float(torch.sigmoid(controls[1])) * state.shape[1]),
        state.shape[1] - 1,
    )
    secondary_feature = min(
        int(float(torch.sigmoid(controls[2])) * state.shape[1]),
        state.shape[1] - 1,
    )
    scope = float(0.25 + 0.75 * torch.sigmoid(controls[3]))
    edge_gain = float(0.50 + torch.sigmoid(controls[4]))
    mixing = float(0.05 + 0.45 * torch.sigmoid(controls[5]))
    delay_strength = float(0.10 + 0.60 * torch.sigmoid(controls[6]))
    polarity_gain = float(0.50 + torch.sigmoid(controls[7]))
    delay_mix = 0.0
    if intervention_type == 0:
        changed_state[source, feature] += amplitude
        changed_state[target, secondary_feature] += 0.25 * amplitude * scope
    elif intervention_type == 1:
        changed_state[source, feature] *= 1.0 - amplitude * scope
        changed_state[target, secondary_feature] *= 1.0 - 0.25 * amplitude * scope
    elif intervention_type == 2:
        transfer = amplitude * scope * changed_state[source, feature]
        changed_state[source, feature] -= transfer
        changed_state[target, feature] += transfer
        secondary_transfer = 0.25 * amplitude * changed_state[source, secondary_feature]
        changed_state[source, secondary_feature] -= secondary_transfer
        changed_state[target, secondary_feature] += secondary_transfer
    elif intervention_type == 3:
        edge_delta = amplitude * edge_gain
        changed_relations[source, target, 0] += edge_delta
        if changed_relations.shape[-1] > 1:
            current = float(changed_relations[source, target, 1])
            direction = -1.0 if current < 0.0 else 1.0
            changed_relations[source, target, 1] += direction * edge_delta
        if changed_relations.shape[-1] > 2:
            changed_relations[source, target, 2] += mixing * edge_delta
    elif intervention_type == 4:
        weakening = min(amplitude * edge_gain, 0.90)
        changed_relations[source, target] *= 1.0 - weakening
    elif intervention_type == 5:
        blend = min(amplitude * scope + mixing, 0.90)
        mean_value = changed_state[:, feature].mean()
        changed_state[:, feature] = (
            (1.0 - blend) * changed_state[:, feature] + blend * mean_value
        )
        secondary_mean = changed_state[:, secondary_feature].mean()
        changed_state[:, secondary_feature] = (
            (1.0 - 0.5 * blend) * changed_state[:, secondary_feature]
            + 0.5 * blend * secondary_mean
        )
    elif intervention_type == 6:
        delay_mix = min(amplitude + delay_strength * scope, 0.85)
    else:
        if changed_relations.shape[-1] > 1:
            changed_relations[source, target, 1] *= -polarity_gain
        else:
            changed_relations[source, target, 0] *= -polarity_gain
    return changed_state.clamp(-2.0, 2.0), changed_relations, delay_mix


def intervention_parameter_sensitivity(config: MetaWorldModelConfig) -> tuple[bool, ...]:
    """Check that every registered C0 control can change its assigned intervention."""

    if config.intervention_parameters != 8:
        return (False,) * config.intervention_parameters
    active_slots = 4
    state = torch.linspace(
        -0.6,
        0.6,
        active_slots * config.observation_features,
    ).reshape(active_slots, config.observation_features)
    relations = torch.zeros(
        active_slots,
        active_slots,
        config.relation_features,
    )
    for source in range(active_slots):
        target = (source + 1) % active_slots
        relations[source, target, 0] = 0.5
        if config.relation_features > 1:
            relations[source, target, 1] = 0.5
        if config.relation_features > 2:
            relations[source, target, 2] = 0.5
    baseline = torch.zeros(config.intervention_parameters)
    assigned_interventions = (0, 0, 0, 0, 3, 5, 6, 7)
    sensitivity: list[bool] = []
    for parameter_index, intervention_type in enumerate(assigned_interventions):
        changed = baseline.clone()
        changed[parameter_index] = 2.0
        base_state, base_relations, base_delay = _apply_intervention(
            state,
            relations,
            intervention_type,
            0,
            1,
            baseline,
        )
        changed_state, changed_relations, changed_delay = _apply_intervention(
            state,
            relations,
            intervention_type,
            0,
            1,
            changed,
        )
        sensitivity.append(
            not torch.equal(base_state, changed_state)
            or not torch.equal(base_relations, changed_relations)
            or base_delay != changed_delay
        )
    return tuple(sensitivity)


def make_indexed_batch(
    config: MetaWorldModelConfig,
    record_seeds: Tensor,
    domain_ids: Tensor,
    mechanism_ids: Tensor,
    intervention_types: Tensor,
    eras: Tensor,
    active_slots: int,
    transform_seed: int,
    device: torch.device | str = "cpu",
) -> MetaWorldBatch:
    """Materialize order-independent trajectories from a compact numeric index."""

    if active_slots <= 1 or active_slots > config.max_slots:
        raise ValueError("active_slots must be within the model slot capacity")
    index_tensors = {
        "record_seeds": record_seeds,
        "domain_ids": domain_ids,
        "mechanism_ids": mechanism_ids,
        "intervention_types": intervention_types,
        "eras": eras,
    }
    if any(tensor.ndim != 1 for tensor in index_tensors.values()):
        raise ValueError("corpus index tensors must be one-dimensional")
    sizes = {int(tensor.numel()) for tensor in index_tensors.values()}
    if len(sizes) != 1 or not sizes or next(iter(sizes)) <= 0:
        raise ValueError("corpus index tensors must have one shared non-zero length")
    batch_size = next(iter(sizes))
    if torch.any(domain_ids < 0) or torch.any(domain_ids >= config.domain_count):
        raise ValueError("domain_ids are outside the configured range")
    if torch.any(mechanism_ids < 0) or torch.any(mechanism_ids >= config.mechanism_count):
        raise ValueError("mechanism_ids are outside the configured range")
    if torch.any(intervention_types < 0) or torch.any(
        intervention_types >= config.intervention_types
    ):
        raise ValueError("intervention_types are outside the configured range")
    if torch.any(eras < 0):
        raise ValueError("eras must be non-negative")
    shape = (
        batch_size,
        config.context_steps,
        config.max_slots,
        config.observation_features,
    )
    observations = torch.zeros(shape)
    observation_mask = torch.zeros(shape, dtype=torch.bool)
    slot_mask = torch.zeros(shape[:-1], dtype=torch.bool)
    relations = torch.zeros(
        batch_size,
        config.context_steps,
        config.max_slots,
        config.max_slots,
        config.relation_features,
    )
    time_mask = torch.ones(batch_size, config.context_steps, dtype=torch.bool)
    source_slots = torch.empty(batch_size, dtype=torch.long)
    target_slots = torch.empty(batch_size, dtype=torch.long)
    intervention_parameters = torch.empty(batch_size, config.intervention_parameters)
    next_observations = torch.zeros(
        batch_size, config.max_slots, config.observation_features
    )
    next_observation_mask = torch.zeros_like(next_observations, dtype=torch.bool)
    effect_targets = torch.zeros(batch_size, config.effect_dimensions)

    seed_values = record_seeds.detach().cpu().tolist()
    domain_values = domain_ids.detach().cpu().tolist()
    mechanism_values = mechanism_ids.detach().cpu().tolist()
    intervention_values = intervention_types.detach().cpu().tolist()
    era_values = eras.detach().cpu().tolist()
    for sample, record_seed in enumerate(seed_values):
        generator = torch.Generator(device="cpu").manual_seed(int(record_seed))
        mechanism_id = int(mechanism_values[sample])
        domain_id = int(domain_values[sample])
        intervention_type = int(intervention_values[sample])
        era = int(era_values[sample])
        source = int(record_seed) % active_slots
        target = (source + 1 + mechanism_id) % active_slots
        source_slots[sample] = source
        target_slots[sample] = target
        intervention_parameters[sample] = torch.randn(
            config.intervention_parameters, generator=generator
        )
        relation = _make_relations(active_slots, config.relation_features, generator)
        if intervention_type in {4, 7} and relation[source, target, 0] == 0:
            relation[source, target, 0] = 0.25
            if config.relation_features > 1:
                relation[source, target, 1] = 0.25
            if config.relation_features > 2:
                relation[source, target, 2] = 0.25
        state = torch.randn(
            active_slots, config.observation_features, generator=generator
        ) * 0.35
        transform, bias = _domain_transform(
            domain_id, config.observation_features, transform_seed
        )
        for step in range(config.context_steps):
            observed = _observe(state, transform, bias)
            observed_mask = torch.rand(
                active_slots, config.observation_features, generator=generator
            ) > 0.10
            observed_mask[:, 0] = True
            observations[sample, step, :active_slots] = observed * observed_mask
            observation_mask[sample, step, :active_slots] = observed_mask
            slot_mask[sample, step, :active_slots] = True
            relations[sample, step, :active_slots, :active_slots] = relation
            if step + 1 < config.context_steps:
                state = _evolve(state, relation, mechanism_id, era)

        counterfactual = _evolve(state, relation, mechanism_id, era)
        changed_state, changed_relations, delay_mix = _apply_intervention(
            state,
            relation,
            intervention_type,
            source,
            target,
            intervention_parameters[sample],
        )
        next_latent = _evolve(changed_state, changed_relations, mechanism_id, era)
        if delay_mix:
            next_latent = delay_mix * state + (1.0 - delay_mix) * next_latent
        effect = next_latent - counterfactual
        effect_summary = torch.stack(
            [
                effect.mean(),
                effect.abs().mean(),
                effect.square().mean().sqrt(),
                effect.abs().amax(),
            ]
        )
        effect_targets[sample, : min(config.effect_dimensions, 4)] = effect_summary[
            : config.effect_dimensions
        ]
        next_observed = _observe(next_latent, transform, bias)
        target_mask = torch.rand(
            active_slots, config.observation_features, generator=generator
        ) > 0.05
        target_mask[:, 0] = True
        next_observations[sample, :active_slots] = next_observed * target_mask
        next_observation_mask[sample, :active_slots] = target_mask

    batch = MetaWorldBatch(
        observations=observations,
        observation_mask=observation_mask,
        slot_mask=slot_mask,
        relations=relations,
        time_mask=time_mask,
        domain_ids=domain_ids.to(dtype=torch.long, device="cpu"),
        intervention_types=intervention_types.to(dtype=torch.long, device="cpu"),
        source_slots=source_slots,
        target_slots=target_slots,
        intervention_parameters=intervention_parameters,
        next_observations=next_observations,
        next_observation_mask=next_observation_mask,
        effect_targets=effect_targets,
        mechanism_ids=mechanism_ids.to(dtype=torch.long, device="cpu"),
    )
    batch.validate()
    return batch.to(device)


def make_mechanistic_batch(
    config: MetaWorldModelConfig,
    batch_size: int,
    active_slots: int,
    seed: int,
    device: torch.device | str = "cpu",
) -> MetaWorldBatch:
    """Build the fixed H000/H001 engineering batch from explicit record seeds."""

    if batch_size < config.mechanism_count * 2:
        raise ValueError("batch_size must contain at least two views of each mechanism")
    records = torch.arange(batch_size, dtype=torch.long)
    record_seeds = seed + records * 104_729
    domain_ids = (records // config.mechanism_count) % config.domain_count
    mechanism_ids = records % config.mechanism_count
    intervention_types = records % config.intervention_types
    eras = torch.zeros(batch_size, dtype=torch.long)
    return make_indexed_batch(
        config,
        record_seeds=record_seeds,
        domain_ids=domain_ids,
        mechanism_ids=mechanism_ids,
        intervention_types=intervention_types,
        eras=eras,
        active_slots=active_slots,
        transform_seed=seed,
        device=device,
    )
