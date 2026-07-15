"""Minimal trainer with an EMA target encoder for latent prediction."""

from __future__ import annotations

import copy
import random
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

from chimera.config import TrainingConfig
from chimera.data.contracts import TrainingBatch
from chimera.models.encoder import BusinessGraphEncoder
from chimera.models.venture import ChimeraVenture
from chimera.training.objectives import LossWeights, chimera_loss


def resolve_device(requested: str) -> torch.device:
    if requested != "auto":
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class ChimeraTrainer:
    """Train one model while updating a non-gradient target representation."""

    def __init__(
        self,
        model: ChimeraVenture,
        config: TrainingConfig,
        *,
        loss_weights: LossWeights | None = None,
    ) -> None:
        seed_everything(config.seed)
        self.config = config
        self.device = resolve_device(config.device)
        self.model = model.to(self.device)
        self.target_encoder: BusinessGraphEncoder = copy.deepcopy(model.encoder).to(self.device)
        self.target_encoder.requires_grad_(False)
        self.target_encoder.eval()
        self.loss_weights = loss_weights or LossWeights()
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )
        self.step = 0

    def train_step(self, batch: TrainingBatch) -> dict[str, float]:
        batch = batch.with_terminal_stop()
        batch.validate(
            feature_dim=self.model.config.node_numeric_features,
            score_dimensions=self.model.config.score_dimensions,
        )
        batch = batch.to(self.device)
        self.model.train()
        self.optimizer.zero_grad(set_to_none=True)
        output = self.model(batch.graph, batch.edits)
        with torch.no_grad():
            target_state = self.target_encoder(batch.next_graph).graph_state
        losses = chimera_loss(
            output,
            batch.edits,
            batch.scores,
            target_state,
            weights=self.loss_weights,
        )
        losses["loss"].backward()  # type: ignore[no-untyped-call]
        gradient_norm = nn.utils.clip_grad_norm_(self.model.parameters(), self.config.max_grad_norm)
        self.optimizer.step()
        self._update_target_encoder()
        self.step += 1
        metrics = {name: float(value.detach().cpu()) for name, value in losses.items()}
        metrics["gradient_norm"] = float(gradient_norm.detach().cpu())
        return metrics

    @torch.no_grad()
    def evaluate_step(self, batch: TrainingBatch) -> dict[str, float]:
        batch = batch.with_terminal_stop()
        batch.validate(
            feature_dim=self.model.config.node_numeric_features,
            score_dimensions=self.model.config.score_dimensions,
        )
        batch = batch.to(self.device)
        self.model.eval()
        output = self.model(batch.graph, batch.edits)
        target_state = self.target_encoder(batch.next_graph).graph_state
        losses = chimera_loss(
            output,
            batch.edits,
            batch.scores,
            target_state,
            weights=self.loss_weights,
        )
        return {name: float(value.detach().cpu()) for name, value in losses.items()}

    @torch.no_grad()
    def _update_target_encoder(self) -> None:
        decay = self.config.target_ema_decay
        for target, source in zip(
            self.target_encoder.parameters(), self.model.encoder.parameters(), strict=True
        ):
            target.mul_(decay).add_(source, alpha=1.0 - decay)

    def checkpoint(self) -> dict[str, Any]:
        return {
            "format_version": 1,
            "step": self.step,
            "model_config": asdict(self.model.config),
            "training_config": asdict(self.config),
            "model_state": self.model.state_dict(),
            "target_encoder_state": self.target_encoder.state_dict(),
            "optimizer_state": self.optimizer.state_dict(),
        }

    def save_checkpoint(self, path: str | Path, *, metadata: dict[str, Any] | None = None) -> Path:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        payload = self.checkpoint()
        payload["metadata"] = metadata or {}
        temporary = destination.with_suffix(destination.suffix + ".tmp")
        torch.save(payload, temporary)
        temporary.replace(destination)
        return destination
