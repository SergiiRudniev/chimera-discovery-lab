"""Frozen-backbone ranking-head trainer for H016."""

from __future__ import annotations

import math
from collections.abc import Sequence
from contextlib import nullcontext

import torch
from torch import Tensor, nn

from chimera.meta_world.h015.evaluation import candidate_batch
from chimera.meta_world.h016.config import H016RankingTrainingConfig
from chimera.meta_world.h016.dataset import H016RankingGroup
from chimera.meta_world.h016.model import WithinStateActionRanker
from chimera.meta_world.h016.objectives import h016_ranking_loss


class H016RankingTrainer:
    """Optimize only the ranking head over exact generated action groups."""

    def __init__(
        self,
        model: WithinStateActionRanker,
        config: H016RankingTrainingConfig,
        *,
        device: torch.device,
        use_autocast: bool,
    ) -> None:
        self.model = model.to(device)
        self.config = config
        self.device = device
        self.use_autocast = use_autocast
        trainable = [
            parameter for parameter in self.model.parameters() if parameter.requires_grad
        ]
        if not trainable or any(
            parameter.requires_grad for parameter in self.model.backbone.parameters()
        ):
            raise ValueError("H016 requires a frozen backbone and trainable rank head")
        self.trainable_parameters = trainable
        self.optimizer = torch.optim.AdamW(
            trainable,
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)

    def _autocast(self) -> object:
        if self.use_autocast:
            return torch.autocast(
                device_type=self.device.type,
                dtype=torch.bfloat16,
            )
        return nullcontext()

    def train_step(self, groups: Sequence[H016RankingGroup]) -> dict[str, float]:
        if len(groups) != self.config.states_per_step:
            raise ValueError("H016 train step received the wrong state-group count")
        self.model.train()
        self.optimizer.zero_grad(set_to_none=True)
        group_losses: list[dict[str, Tensor]] = []
        for group in groups:
            batch = candidate_batch(group.window, group.candidates).to(self.device)
            targets = torch.from_numpy(group.realized_effects).to(self.device)
            with self._autocast():  # type: ignore[attr-defined]
                output = self.model(batch)
                losses = h016_ranking_loss(
                    output.rank_logits,
                    targets,
                    self.config,
                )
            group_losses.append(losses)
        loss = torch.stack([item["loss"] for item in group_losses]).mean()
        loss.backward()  # type: ignore[no-untyped-call]
        gradient_norm = nn.utils.clip_grad_norm_(
            self.trainable_parameters,
            self.config.max_grad_norm,
        )
        self.optimizer.step()
        names = tuple(group_losses[0])
        metrics = {
            name: float(
                torch.stack([item[name].detach().float() for item in group_losses])
                .mean()
                .cpu()
            )
            for name in names
        }
        metrics["gradient_norm"] = float(gradient_norm.detach().float().cpu())
        metrics["backbone_gradient_tensors"] = float(
            sum(parameter.grad is not None for parameter in self.model.backbone.parameters())
        )
        if not all(math.isfinite(value) for value in metrics.values()):
            raise FloatingPointError("non-finite H016 ranking training metric")
        return metrics

    def peak_memory_bytes(self) -> int:
        if self.device.type != "cuda":
            return 0
        return int(torch.cuda.max_memory_allocated(self.device))
