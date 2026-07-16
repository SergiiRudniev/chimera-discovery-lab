"""Frozen numerical backbone plus trainable within-state ranking critic."""

from __future__ import annotations

from dataclasses import dataclass
from typing import cast

import torch
from torch import Tensor, nn

from chimera.meta_world.batch import MetaWorldBatch
from chimera.meta_world.h014.model import ResponseConditionedEffectWorldModel
from chimera.meta_world.model import MetaWorldOutput


@dataclass(frozen=True)
class H016RankingOutput:
    """Candidate rank logit and frozen pointwise-backbone output."""

    rank_logits: Tensor
    backbone: MetaWorldOutput


class WithinStateActionRanker(nn.Module):
    """Learn action ordering without updating the H015 numerical backbone."""

    def __init__(self, backbone: ResponseConditionedEffectWorldModel) -> None:
        super().__init__()
        self.backbone = backbone
        hidden = backbone.config.hidden_dim
        self.rank_head = nn.Sequential(
            nn.LayerNorm(hidden),
            nn.Linear(hidden, 256),
            nn.SiLU(),
            nn.Linear(256, 1),
        )
        for parameter in self.backbone.parameters():
            parameter.requires_grad_(False)
        self.backbone.eval()

    def train(self, mode: bool = True) -> WithinStateActionRanker:
        super().train(mode)
        self.backbone.eval()
        self.rank_head.train(mode)
        return self

    def forward(self, batch: MetaWorldBatch) -> H016RankingOutput:
        with torch.no_grad():
            backbone_output = cast(MetaWorldOutput, self.backbone(batch))
        logits = self.rank_head(backbone_output.transition_state.detach()).squeeze(-1)
        return H016RankingOutput(rank_logits=logits.float(), backbone=backbone_output)

    def trainable_parameter_count(self) -> int:
        return sum(
            parameter.numel() for parameter in self.parameters() if parameter.requires_grad
        )

    def total_parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())

    def frozen_backbone_parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.backbone.parameters())
