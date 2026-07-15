"""GPU/CPU trainer shared by H002 relational and temporal prediction arms."""

from __future__ import annotations

import math
from typing import cast

import torch
from torch import nn

from chimera.meta_world.batch import MetaWorldBatch
from chimera.meta_world.config import MetaWorldTrainingConfig
from chimera.meta_world.h002.objectives import h002_loss
from chimera.meta_world.model import MetaWorldOutput
from chimera.meta_world.trainer import resolve_device


class H002Trainer:
    """Train any H002 transition model without passing service metadata forward."""

    def __init__(self, model: nn.Module, config: MetaWorldTrainingConfig) -> None:
        self.config = config
        self.device = resolve_device(config.device)
        torch.manual_seed(config.seed)
        if self.device.type == "cuda":
            torch.cuda.manual_seed_all(config.seed)
            torch.set_float32_matmul_precision("high")
            torch.cuda.reset_peak_memory_stats(self.device)
        self.model = model.to(self.device)
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )
        self.use_autocast = config.precision == "bfloat16" and self.device.type == "cuda"

    def _autocast(self) -> torch.autocast:
        return torch.autocast(
            device_type=self.device.type,
            dtype=torch.bfloat16,
            enabled=self.use_autocast,
        )

    def train_step(self, batch: MetaWorldBatch) -> dict[str, float]:
        self.model.train()
        device_batch = batch.to(self.device)
        self.optimizer.zero_grad(set_to_none=True)
        with self._autocast():
            output = cast(MetaWorldOutput, self.model(device_batch))
            losses = h002_loss(output, device_batch, self.config)
        losses["loss"].backward()  # type: ignore[no-untyped-call]
        gradient_norm = nn.utils.clip_grad_norm_(
            self.model.parameters(),
            self.config.max_grad_norm,
        )
        self.optimizer.step()
        metrics = {
            name: float(value.detach().float().cpu()) for name, value in losses.items()
        }
        metrics["gradient_norm"] = float(gradient_norm.detach().float().cpu())
        if not all(math.isfinite(value) for value in metrics.values()):
            raise FloatingPointError("non-finite H002 training metric")
        return metrics

    @torch.no_grad()
    def predict(self, batch: MetaWorldBatch) -> MetaWorldOutput:
        self.model.eval()
        with self._autocast():
            return cast(MetaWorldOutput, self.model(batch.to(self.device)))

    def peak_memory_bytes(self) -> int:
        if self.device.type != "cuda":
            return 0
        return int(torch.cuda.max_memory_allocated(self.device))

