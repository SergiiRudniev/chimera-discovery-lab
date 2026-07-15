"""Action-conditioned latent transition model."""

from __future__ import annotations

from typing import cast

from torch import Tensor, nn

from chimera.config import ModelConfig


class TransitionBlock(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        inner = config.hidden_dim * config.feedforward_multiplier
        self.norm = nn.LayerNorm(config.hidden_dim)
        self.network = nn.Sequential(
            nn.Linear(config.hidden_dim, inner),
            nn.SiLU(),
            nn.Dropout(config.dropout),
            nn.Linear(inner, config.hidden_dim),
            nn.Dropout(config.dropout),
        )

    def forward(self, values: Tensor) -> Tensor:
        return cast(Tensor, values + self.network(self.norm(values)))


class LatentWorldModel(nn.Module):
    """Predict the next business-state embedding from a proposed edit program."""

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.action_projection = nn.Linear(config.hidden_dim, config.hidden_dim)
        self.input_projection = nn.Linear(config.hidden_dim * 2, config.hidden_dim)
        self.blocks = nn.ModuleList(
            [TransitionBlock(config) for _ in range(config.transition_layers)]
        )
        self.output_norm = nn.LayerNorm(config.hidden_dim)

    def forward(self, graph_state: Tensor, action_state: Tensor) -> Tensor:
        values = self.input_projection(
            self._concat(graph_state, self.action_projection(action_state))
        )
        for block in self.blocks:
            values = block(values)
        return cast(Tensor, self.output_norm(values))

    @staticmethod
    def _concat(left: Tensor, right: Tensor) -> Tensor:
        from torch import cat

        return cat((left, right), dim=-1)
