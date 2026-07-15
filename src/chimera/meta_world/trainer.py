"""Optimizer and inference wrapper for Chimera Meta-World W0."""

from __future__ import annotations

import math
from typing import cast

import torch
from torch import nn

from chimera.meta_world.batch import MetaWorldBatch
from chimera.meta_world.config import MetaWorldTrainingConfig
from chimera.meta_world.model import ChimeraMetaWorld, MetaWorldOutput
from chimera.meta_world.objectives import meta_world_loss


def resolve_device(requested: str) -> torch.device:
    if requested == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    return device


class MetaWorldTrainer:
    """Stateful W0 trainer with explicit precision and memory reporting."""

    def __init__(self, model: ChimeraMetaWorld, config: MetaWorldTrainingConfig) -> None:
        self.config = config
        self.device = resolve_device(config.device)
        torch.manual_seed(config.seed)
        if self.device.type == "cuda":
            torch.cuda.manual_seed_all(config.seed)
            torch.set_float32_matmul_precision("high")
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
            output = self.model(device_batch)
            losses = meta_world_loss(output, device_batch, self.config)
        losses["loss"].backward()  # type: ignore[no-untyped-call]
        gradient_norm = nn.utils.clip_grad_norm_(self.model.parameters(), self.config.max_grad_norm)
        self.optimizer.step()
        metrics = {name: float(value.detach().float().cpu()) for name, value in losses.items()}
        metrics["gradient_norm"] = float(gradient_norm.detach().float().cpu())
        if not all(math.isfinite(value) for value in metrics.values()):
            raise FloatingPointError("non-finite Meta-World training metric")
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
