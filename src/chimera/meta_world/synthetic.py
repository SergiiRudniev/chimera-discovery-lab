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


def _evolve(state: Tensor, relations: Tensor, mechanism_id: int) -> Tensor:
    adjacency = relations[..., 0]
    signed = relations[..., 1] if relations.shape[-1] > 1 else adjacency
    incoming = signed.transpose(0, 1) @ state
    if mechanism_id == 0:
        delta = 0.12 * (incoming - state)
    elif mechanism_id == 1:
        delta = 0.08 * torch.tanh(incoming) + 0.04 * state - 0.025 * state.pow(3)
    elif mechanism_id == 2:
        competition = adjacency.transpose(0, 1) @ state.abs()
        delta = 0.08 * state * (1.0 - competition)
    else:
        if relations.shape[-1] > 2:
            capacity = relations[..., 2].sum(dim=0).unsqueeze(-1).clamp_min(0.25)
        else:
            capacity = torch.ones(state.shape[0], 1)
        overflow = torch.relu(state.abs() - capacity) * torch.sign(state)
        delta = 0.10 * incoming - 0.08 * overflow
    return (state + delta).clamp(-2.0, 2.0)


def _apply_intervention(
    state: Tensor,
    relations: Tensor,
    intervention_type: int,
    source: int,
    target: int,
    parameters: Tensor,
) -> tuple[Tensor, Tensor, float]:
    changed_state = state.clone()
    changed_relations = relations.clone()
    amplitude = float(0.05 + 0.20 * torch.sigmoid(parameters[0]))
    feature = int(abs(float(parameters[1])) * 10_000) % state.shape[1]
    delay_mix = 0.0
    if intervention_type == 0:
        changed_state[source, feature] += amplitude
    elif intervention_type == 1:
        changed_state[source, feature] *= 1.0 - amplitude
    elif intervention_type == 2:
        transfer = amplitude * changed_state[source, feature]
        changed_state[source, feature] -= transfer
        changed_state[target, feature] += transfer
    elif intervention_type == 3:
        changed_relations[source, target, 0] += amplitude
        if changed_relations.shape[-1] > 1:
            changed_relations[source, target, 1] += amplitude
    elif intervention_type == 4:
        changed_relations[source, target] *= 1.0 - amplitude
    elif intervention_type == 5:
        mean_value = changed_state[:, feature].mean()
        changed_state[:, feature] = (
            (1.0 - amplitude) * changed_state[:, feature] + amplitude * mean_value
        )
    elif intervention_type == 6:
        delay_mix = amplitude
    else:
        if changed_relations.shape[-1] > 1:
            changed_relations[source, target, 1] *= -1.0
        else:
            changed_relations[source, target, 0] *= -1.0
    return changed_state.clamp(-2.0, 2.0), changed_relations, delay_mix


def make_mechanistic_batch(
    config: MetaWorldModelConfig,
    batch_size: int,
    active_slots: int,
    seed: int,
    device: torch.device | str = "cpu",
) -> MetaWorldBatch:
    """Build fixed histories with four known dynamics and eight interventions."""

    if batch_size < config.mechanism_count * 2:
        raise ValueError("batch_size must contain at least two views of each mechanism")
    if active_slots <= 1 or active_slots > config.max_slots:
        raise ValueError("active_slots must be within the model slot capacity")
    generator = torch.Generator(device="cpu").manual_seed(seed)
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
    domain_ids = torch.empty(batch_size, dtype=torch.long)
    mechanism_ids = torch.empty(batch_size, dtype=torch.long)
    intervention_types = torch.empty(batch_size, dtype=torch.long)
    source_slots = torch.empty(batch_size, dtype=torch.long)
    target_slots = torch.empty(batch_size, dtype=torch.long)
    intervention_parameters = torch.randn(
        batch_size, config.intervention_parameters, generator=generator
    )
    next_observations = torch.zeros(
        batch_size, config.max_slots, config.observation_features
    )
    next_observation_mask = torch.zeros_like(next_observations, dtype=torch.bool)
    effect_targets = torch.zeros(batch_size, config.effect_dimensions)

    for sample in range(batch_size):
        mechanism_id = sample % config.mechanism_count
        domain_id = (sample // config.mechanism_count) % config.domain_count
        mechanism_ids[sample] = mechanism_id
        domain_ids[sample] = domain_id
        intervention_type = sample % config.intervention_types
        intervention_types[sample] = intervention_type
        source = sample % active_slots
        target = (source + 1 + mechanism_id) % active_slots
        source_slots[sample] = source
        target_slots[sample] = target
        relation = _make_relations(active_slots, config.relation_features, generator)
        state = torch.randn(
            active_slots, config.observation_features, generator=generator
        ) * 0.35
        transform, bias = _domain_transform(domain_id, config.observation_features, seed)
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
                state = _evolve(state, relation, mechanism_id)

        counterfactual = _evolve(state, relation, mechanism_id)
        changed_state, changed_relations, delay_mix = _apply_intervention(
            state,
            relation,
            intervention_type,
            source,
            target,
            intervention_parameters[sample],
        )
        next_latent = _evolve(changed_state, changed_relations, mechanism_id)
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
        domain_ids=domain_ids,
        intervention_types=intervention_types,
        source_slots=source_slots,
        target_slots=target_slots,
        intervention_parameters=intervention_parameters,
        next_observations=next_observations,
        next_observation_mask=next_observation_mask,
        effect_targets=effect_targets,
        mechanism_ids=mechanism_ids,
    )
    batch.validate()
    return batch.to(device)
