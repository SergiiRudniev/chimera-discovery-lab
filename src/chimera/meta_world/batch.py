"""Tensor batch contract for dynamic Meta-World observations."""

from __future__ import annotations

from dataclasses import dataclass, fields

import torch
from torch import Tensor


@dataclass(frozen=True)
class MetaWorldBatch:
    """One batch of histories, interventions and observed consequences."""

    observations: Tensor
    observation_mask: Tensor
    slot_mask: Tensor
    relations: Tensor
    time_mask: Tensor
    domain_ids: Tensor
    intervention_types: Tensor
    source_slots: Tensor
    target_slots: Tensor
    intervention_parameters: Tensor
    next_observations: Tensor
    next_observation_mask: Tensor
    effect_targets: Tensor
    mechanism_ids: Tensor

    @property
    def batch_size(self) -> int:
        return int(self.observations.shape[0])

    def validate(self) -> None:
        if self.observations.ndim != 4:
            raise ValueError("observations must have shape [batch, time, slots, features]")
        batch, time, slots, features = self.observations.shape
        expected = {
            "observation_mask": (batch, time, slots, features),
            "slot_mask": (batch, time, slots),
            "time_mask": (batch, time),
            "domain_ids": (batch,),
            "intervention_types": (batch,),
            "source_slots": (batch,),
            "target_slots": (batch,),
            "next_observations": (batch, slots, features),
            "next_observation_mask": (batch, slots, features),
            "mechanism_ids": (batch,),
        }
        for name, shape in expected.items():
            if tuple(getattr(self, name).shape) != shape:
                raise ValueError(f"{name} must have shape {shape}")
        if self.relations.ndim != 5 or tuple(self.relations.shape[:4]) != (
            batch,
            time,
            slots,
            slots,
        ):
            raise ValueError("relations must have shape [batch, time, slots, slots, features]")
        if self.intervention_parameters.ndim != 2 or self.intervention_parameters.shape[0] != batch:
            raise ValueError("intervention_parameters must have shape [batch, parameters]")
        if self.effect_targets.ndim != 2 or self.effect_targets.shape[0] != batch:
            raise ValueError("effect_targets must have shape [batch, effects]")
        for name in ("observation_mask", "slot_mask", "time_mask", "next_observation_mask"):
            if getattr(self, name).dtype != torch.bool:
                raise TypeError(f"{name} must be boolean")
        if not torch.all(self.time_mask.any(dim=1)):
            raise ValueError("every sample must contain at least one context step")
        if torch.any(self.time_mask[:, 1:] & ~self.time_mask[:, :-1]):
            raise ValueError("time_mask must be a contiguous active prefix")
        if torch.any(self.slot_mask & ~self.time_mask.unsqueeze(-1)):
            raise ValueError("slots cannot be active outside the time mask")
        if torch.any(self.observation_mask & ~self.slot_mask.unsqueeze(-1)):
            raise ValueError("observations cannot be active outside the slot mask")
        if not torch.all(self.slot_mask.any(dim=2)[self.time_mask]):
            raise ValueError("every active context step must contain at least one slot")
        final_steps = self.time_mask.sum(dim=1) - 1
        final_masks = self.slot_mask[torch.arange(batch, device=final_steps.device), final_steps]
        if torch.any(self.next_observation_mask & ~final_masks.unsqueeze(-1)):
            raise ValueError("next observations must belong to final-step slots")
        for name in ("source_slots", "target_slots"):
            pointers = getattr(self, name)
            if torch.any(pointers < 0) or torch.any(pointers >= slots):
                raise ValueError(f"{name} contains an out-of-range pointer")
            if not torch.all(final_masks.gather(1, pointers[:, None]).squeeze(1)):
                raise ValueError(f"{name} must point to an active final-step slot")

    def to(self, device: torch.device | str) -> MetaWorldBatch:
        values = {item.name: getattr(self, item.name).to(device) for item in fields(self)}
        return MetaWorldBatch(**values)
