"""GPU/CPU trainer shared by H002 relational and temporal prediction arms."""

from __future__ import annotations

import copy
import math
from collections.abc import Mapping
from typing import cast

import torch
from torch import Tensor, nn

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
        self.ema_model = (
            copy.deepcopy(self.model).requires_grad_(False)
            if config.ema_decay > 0.0
            else None
        )
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
        self._update_ema()
        metrics = {
            name: float(value.detach().float().cpu()) for name, value in losses.items()
        }
        metrics["gradient_norm"] = float(gradient_norm.detach().float().cpu())
        if not all(math.isfinite(value) for value in metrics.values()):
            raise FloatingPointError("non-finite H002 training metric")
        return metrics

    def _update_ema(self) -> None:
        """Update evaluation weights after one successful optimizer step."""

        if self.ema_model is not None:
            with torch.no_grad():
                for averaged, current in zip(
                    self.ema_model.parameters(),
                    self.model.parameters(),
                    strict=True,
                ):
                    averaged.lerp_(current, 1.0 - self.config.ema_decay)
                for averaged_buffer, current_buffer in zip(
                    self.ema_model.buffers(),
                    self.model.buffers(),
                    strict=True,
                ):
                    averaged_buffer.copy_(current_buffer)

    @torch.no_grad()
    def predict(self, batch: MetaWorldBatch) -> MetaWorldOutput:
        evaluation_model = self.ema_model if self.ema_model is not None else self.model
        evaluation_model.eval()
        with self._autocast():
            return cast(MetaWorldOutput, evaluation_model(batch.to(self.device)))

    def evaluation_state_dict(self) -> Mapping[str, Tensor]:
        evaluation_model = self.ema_model if self.ema_model is not None else self.model
        return evaluation_model.state_dict()

    @property
    def evaluation_weights_kind(self) -> str:
        return "ema" if self.ema_model is not None else "online"

    def peak_memory_bytes(self) -> int:
        if self.device.type != "cuda":
            return 0
        return int(torch.cuda.max_memory_allocated(self.device))
