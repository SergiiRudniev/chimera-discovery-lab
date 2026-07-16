"""Relational world model with one shared aligned predictive mechanism state."""

from __future__ import annotations

from torch import Tensor

from chimera.meta_world.h002.model import RelationalSequenceWorldModel


class SharedBottleneckRelationalWorldModel(RelationalSequenceWorldModel):
    """Use the alignment projection as the mechanism input to prediction too."""

    def _mechanism_views(self, mechanism_state: Tensor) -> tuple[Tensor, Tensor]:
        shared = self.mechanism_projection(mechanism_state)
        return shared, shared
