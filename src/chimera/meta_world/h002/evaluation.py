"""Frozen numerical metrics for H002 validation and later test execution."""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace

import torch
import torch.nn.functional as F
from torch import Tensor

from chimera.meta_world.h002.trainer import H002Trainer
from chimera.meta_world.h002.windows import (
    GeneratedSequenceSample,
    make_transition_window,
)


@dataclass(frozen=True)
class H002EvaluationMetrics:
    """Primary and diagnostic H002 prediction metrics."""

    one_step_prediction_rmse: float
    intervention_effect_rmse: float
    intervention_effect_nrmse: float
    intervention_effect_nll: float
    intervention_effect_90_coverage: float
    four_step_rollout_nrmse: float
    mechanism_retrieval_accuracy: float

    def to_dict(self) -> dict[str, float]:
        return asdict(self)


def _global_nrmse(prediction: Tensor, target: Tensor, mask: Tensor) -> Tensor:
    predicted = prediction[mask].float()
    expected = target[mask].float()
    if expected.numel() <= 1:
        raise ValueError("rollout metric requires observed target features")
    rmse = (predicted - expected).square().mean().sqrt()
    scale = expected.std(unbiased=False).clamp_min(1e-6)
    return rmse / scale


def _retrieval_accuracy(embeddings: Tensor, mechanism_ids: Tensor) -> Tensor:
    normalized = F.normalize(embeddings.float(), dim=-1)
    similarity = normalized @ normalized.transpose(0, 1)
    similarity.fill_diagonal_(torch.finfo(similarity.dtype).min)
    nearest = similarity.argmax(dim=1)
    return (mechanism_ids[nearest] == mechanism_ids).float().mean()


@torch.no_grad()
def evaluate_h002_model(
    trainer: H002Trainer,
    sample: GeneratedSequenceSample,
    *,
    context_steps: int,
    rollout_horizon: int = 4,
) -> H002EvaluationMetrics:
    """Evaluate one-step effects and a fixed final-state autoregressive rollout."""

    generated = sample.batch
    _, time, _, _ = generated.observations.shape
    if context_steps <= 0 or rollout_horizon <= 0:
        raise ValueError("context_steps and rollout_horizon must be positive")
    rollout_start = context_steps - 1
    rollout_final = rollout_start + rollout_horizon
    if rollout_final >= time:
        raise ValueError("sequence is too short for the registered rollout")

    state_squared_error = 0.0
    state_values = 0
    effect_squared_error = 0.0
    effect_values = 0
    effect_targets: list[Tensor] = []
    effect_nll = 0.0
    effect_covered = 0
    retrieval_embedding: Tensor | None = None
    for prediction_step in range(time - 1):
        window = make_transition_window(
            sample,
            prediction_step=prediction_step,
            context_steps=context_steps,
        )
        output = trainer.predict(window)
        target = window.next_observations.to(trainer.device)
        mask = window.next_observation_mask.to(trainer.device)
        difference = output.next_state_mean.float() - target.float()
        state_squared_error += float(difference.square()[mask].sum().cpu())
        state_values += int(mask.sum())
        effect_target = window.effect_targets[:, 3].to(trainer.device).float()
        effect_targets.append(effect_target.detach())
        effect_mean = output.effect_mean[:, 3].float()
        log_variance = output.effect_log_variance[:, 3].float()
        residual = effect_mean - effect_target
        effect_squared_error += float(residual.square().sum().cpu())
        effect_values += int(effect_target.numel())
        effect_nll += float(
            (0.5 * (log_variance + residual.square() * torch.exp(-log_variance)))
            .sum()
            .cpu()
        )
        radius = 1.6448536269514722 * torch.exp(0.5 * log_variance)
        effect_covered += int((residual.abs() <= radius).sum().cpu())
        if prediction_step == rollout_start:
            retrieval_embedding = output.proposal_embedding.detach().float()

    working_observations = generated.observations.clone()
    final_prediction: Tensor | None = None
    for prediction_step in range(rollout_start, rollout_final):
        rolling_batch = replace(generated, observations=working_observations)
        rolling_sample = replace(sample, batch=rolling_batch)
        window = make_transition_window(
            rolling_sample,
            prediction_step=prediction_step,
            context_steps=context_steps,
        )
        output = trainer.predict(window)
        prediction = output.next_state_mean.detach().cpu()
        next_mask = generated.object_mask[:, prediction_step + 1].unsqueeze(-1)
        rolled_state = torch.zeros_like(prediction)
        rolled_state[:, :, : sample.state_features] = prediction[
            :, :, : sample.state_features
        ]
        working_observations[:, prediction_step + 1] = torch.where(
            next_mask,
            rolled_state,
            working_observations[:, prediction_step + 1],
        )
        final_prediction = prediction
    if final_prediction is None or retrieval_embedding is None:
        raise RuntimeError("evaluation did not produce registered predictions")
    final_target = generated.observations[:, rollout_final]
    final_mask = torch.zeros_like(final_target, dtype=torch.bool)
    final_mask[:, :, : sample.state_features] = generated.object_mask[
        :, rollout_final
    ].unsqueeze(-1)
    rollout_nrmse = _global_nrmse(final_prediction, final_target, final_mask)
    retrieval = _retrieval_accuracy(
        retrieval_embedding,
        sample.mechanism_ids.to(retrieval_embedding.device),
    )
    concatenated_effect_targets = torch.cat(effect_targets)
    effect_scale = concatenated_effect_targets.std(unbiased=False).clamp_min(1e-6)
    effect_rmse = (effect_squared_error / max(effect_values, 1)) ** 0.5
    return H002EvaluationMetrics(
        one_step_prediction_rmse=(state_squared_error / max(state_values, 1)) ** 0.5,
        intervention_effect_rmse=effect_rmse,
        intervention_effect_nrmse=effect_rmse / float(effect_scale.cpu()),
        intervention_effect_nll=effect_nll / max(effect_values, 1),
        intervention_effect_90_coverage=effect_covered / max(effect_values, 1),
        four_step_rollout_nrmse=float(rollout_nrmse.cpu()),
        mechanism_retrieval_accuracy=float(retrieval.cpu()),
    )
