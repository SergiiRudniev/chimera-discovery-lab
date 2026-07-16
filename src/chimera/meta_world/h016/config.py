"""Frozen backbone, ranking and evaluation configuration for CHM-W-H016."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import yaml

from chimera.meta_world.h013.config import H013Arm, H013RunConfig
from chimera.meta_world.h015.config import H015SearchConfig


def _mapping(value: object, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{name} must be a mapping")
    return cast(Mapping[str, Any], value)


@dataclass(frozen=True)
class H016BackboneConfig:
    """H015 factual-residual backbone retrained under the H016 seed."""

    response_source: str
    paired_runtime: H013RunConfig

    @classmethod
    def from_mapping(cls, values: Mapping[str, Any]) -> H016BackboneConfig:
        expected = {
            "run_id",
            "mode",
            "arm",
            "generator_config",
            "model",
            "training",
            "evaluation",
            "transition_semantics",
            "no_op_state_weight",
            "intervention_delta_weight",
            "response_source",
        }
        if set(values) != expected:
            raise ValueError("H016 backbone fields differ from the frozen schema")
        if values["arm"] != "factual_residual_conditioned_ranking_backbone":
            raise ValueError("H016 backbone arm is immutable")
        response_source = str(values["response_source"])
        if response_source != "predicted_factual_minus_final_observation":
            raise ValueError("H016 must retain the H015 factual-residual backbone")
        paired_values = dict(values)
        paired_values.pop("response_source")
        paired_values["arm"] = H013Arm.DIRECT.value
        paired = H013RunConfig.from_mapping(paired_values)
        training = paired.runtime.training
        if training.seed != 260954 or training.steps not in {2, 600}:
            raise ValueError("H016 backbone seed or engineering step count is invalid")
        return cls(response_source=response_source, paired_runtime=paired)

    @classmethod
    def from_yaml(cls, path: str | Path) -> H016BackboneConfig:
        with Path(path).open("r", encoding="utf-8") as handle:
            values = yaml.safe_load(handle)
        return cls.from_mapping(_mapping(values, "H016 backbone"))


@dataclass(frozen=True)
class H016RankingTrainingConfig:
    """Exact generated ranking-group and optimizer contract."""

    steps: int
    states_per_step: int
    candidates_per_state: int
    context_steps: int
    prediction_step: int
    state_view_rule: str
    candidate_seed_offset: int
    candidate_seed_stride: int
    candidate_sampler: str
    shared_state_event_and_renderer_noise: bool
    backbone_trainable: bool
    target_normalization: str
    listnet_target_temperature: float
    pairwise_weight: float
    pairwise_logit_temperature: float
    minimum_effect_separation: float
    learning_rate: float
    weight_decay: float
    max_grad_norm: float

    @classmethod
    def from_mapping(
        cls,
        values: Mapping[str, Any],
    ) -> H016RankingTrainingConfig:
        expected = {
            "steps",
            "states_per_step",
            "candidates_per_state",
            "context_steps",
            "prediction_step",
            "state_view_rule",
            "candidate_seed_offset",
            "candidate_seed_stride",
            "candidate_sampler",
            "shared_state_event_and_renderer_noise",
            "backbone_trainable",
            "target_normalization",
            "listnet_target_temperature",
            "pairwise_weight",
            "pairwise_logit_temperature",
            "minimum_effect_separation",
            "learning_rate",
            "weight_decay",
            "max_grad_norm",
        }
        if set(values) != expected:
            raise ValueError("H016 ranking fields differ from preregistration")
        result = cls(
            steps=int(values["steps"]),
            states_per_step=int(values["states_per_step"]),
            candidates_per_state=int(values["candidates_per_state"]),
            context_steps=int(values["context_steps"]),
            prediction_step=int(values["prediction_step"]),
            state_view_rule=str(values["state_view_rule"]),
            candidate_seed_offset=int(values["candidate_seed_offset"]),
            candidate_seed_stride=int(values["candidate_seed_stride"]),
            candidate_sampler=str(values["candidate_sampler"]),
            shared_state_event_and_renderer_noise=bool(
                values["shared_state_event_and_renderer_noise"]
            ),
            backbone_trainable=bool(values["backbone_trainable"]),
            target_normalization=str(values["target_normalization"]),
            listnet_target_temperature=float(values["listnet_target_temperature"]),
            pairwise_weight=float(values["pairwise_weight"]),
            pairwise_logit_temperature=float(
                values["pairwise_logit_temperature"]
            ),
            minimum_effect_separation=float(values["minimum_effect_separation"]),
            learning_rate=float(values["learning_rate"]),
            weight_decay=float(values["weight_decay"]),
            max_grad_norm=float(values["max_grad_norm"]),
        )
        if (
            result.steps != 600
            or result.states_per_step != 2
            or result.candidates_per_state != 16
            or result.context_steps != 4
            or result.prediction_step != 3
            or result.state_view_rule
            != "sequential_mechanism_groups_cyclic_renderer_view"
            or result.candidate_seed_offset != 101
            or result.candidate_seed_stride != 1_000_003
            or result.candidate_sampler != "deterministic_uniform_legal"
            or not result.shared_state_event_and_renderer_noise
            or result.backbone_trainable
            or result.target_normalization != "within_state_mean_standard_deviation"
            or result.listnet_target_temperature != 0.50
            or result.pairwise_weight != 0.50
            or result.pairwise_logit_temperature != 0.25
            or result.minimum_effect_separation != 0.00001
            or result.learning_rate != 0.0003
            or result.weight_decay != 0.01
            or result.max_grad_norm != 1.0
        ):
            raise ValueError("H016 ranking protocol is immutable")
        return result


@dataclass(frozen=True)
class H016SuiteConfig:
    """Complete development action-ranking protocol."""

    seed: int
    frozen_validation_seeds: tuple[int, ...]
    generator_config: Path
    integrity_report: Path
    backbone_config: Path
    ranking: H016RankingTrainingConfig
    evaluation_states: int
    oracle_pool_candidates_per_state: int
    ranking_diagnostic_candidates_per_state: int
    search: H015SearchConfig
    random_candidates_per_state: int
    random_regret_ratio_maximum: float
    pointwise_regret_ratio_maximum: float

    @classmethod
    def from_yaml(cls, path: str | Path) -> H016SuiteConfig:
        with Path(path).open("r", encoding="utf-8") as handle:
            values = _mapping(yaml.safe_load(handle), "H016 suite")
        expected = {
            "hypothesis_id",
            "trial_id",
            "mode",
            "development_seed",
            "frozen_validation_seeds",
            "test_access",
            "generator_config",
            "dataset_integrity_report",
            "backbone_config",
            "backbone_training_steps",
            "ranking_training",
            "evaluation_states",
            "oracle_pool_candidates_per_state",
            "ranking_diagnostic_candidates_per_state",
            "search",
            "arms",
            "primary_metric",
            "development_gate",
        }
        if set(values) != expected:
            raise ValueError("H016 suite fields differ from preregistration")
        if (
            values["hypothesis_id"] != "CHM-W-H016"
            or values["trial_id"] != "CHM-W-T016"
            or values["mode"] != "development"
            or values["test_access"] != "sealed"
            or values["primary_metric"] != "realized_best_candidate_regret"
            or int(values["backbone_training_steps"]) != 600
        ):
            raise ValueError("H016 suite identity or access boundary is invalid")
        seed = int(values["development_seed"])
        frozen = tuple(int(item) for item in values["frozen_validation_seeds"])
        if seed != 260954 or frozen != (260955, 260956, 260957):
            raise ValueError("H016 registered seeds are immutable")
        ranking = H016RankingTrainingConfig.from_mapping(
            _mapping(values["ranking_training"], "H016 ranking")
        )
        search_raw = _mapping(values["search"], "H016 search")
        search = H015SearchConfig(
            rounds=int(search_raw["rounds"]),
            candidates_per_round=int(search_raw["candidates_per_round"]),
            elite_candidates_per_round=int(search_raw["elite_candidates_per_round"]),
            simulator_executions_per_state=int(
                search_raw["simulator_executions_per_state"]
            ),
            archive_descriptor=tuple(
                str(item) for item in search_raw["archive_descriptor"]
            ),
        )
        if (
            search.rounds != 4
            or search.candidates_per_round != 64
            or search.elite_candidates_per_round != 8
            or search.simulator_executions_per_state != 8
            or search.archive_descriptor
            != ("source_slot", "target_slot", "magnitude_quartile")
            or search.model_scores_per_state != 256
        ):
            raise ValueError("H016 search differs from H015 preregistered search")
        arms = _mapping(values["arms"], "H016 arms")
        if set(arms) != {
            "listwise_ranking_critic_search",
            "H015_pointwise_mean_only_search",
            "legal_random_intervention_selection",
            "fixed_pool_oracle_evaluator",
        }:
            raise ValueError("H016 comparison arms differ from preregistration")
        rank_arm = _mapping(arms["listwise_ranking_critic_search"], "rank arm")
        point_arm = _mapping(arms["H015_pointwise_mean_only_search"], "point arm")
        random_arm = _mapping(
            arms["legal_random_intervention_selection"], "random arm"
        )
        oracle_arm = _mapping(arms["fixed_pool_oracle_evaluator"], "oracle arm")
        if (
            rank_arm != {"score": "learned_within_state_rank_logit"}
            or point_arm != {"score": "predicted_effect_mean"}
        ):
            raise ValueError("H016 learned-arm scores are immutable")
        gate = _mapping(values["development_gate"], "H016 gate")
        result = cls(
            seed=seed,
            frozen_validation_seeds=frozen,
            generator_config=Path(str(values["generator_config"])),
            integrity_report=Path(str(values["dataset_integrity_report"])),
            backbone_config=Path(str(values["backbone_config"])),
            ranking=ranking,
            evaluation_states=int(values["evaluation_states"]),
            oracle_pool_candidates_per_state=int(
                values["oracle_pool_candidates_per_state"]
            ),
            ranking_diagnostic_candidates_per_state=int(
                values["ranking_diagnostic_candidates_per_state"]
            ),
            search=search,
            random_candidates_per_state=int(random_arm["candidates_per_state"]),
            random_regret_ratio_maximum=float(
                gate["regret_ratio_vs_legal_random_maximum"]
            ),
            pointwise_regret_ratio_maximum=float(
                gate["regret_ratio_vs_H015_pointwise_maximum"]
            ),
        )
        if (
            result.evaluation_states != 32
            or result.oracle_pool_candidates_per_state != 256
            or result.ranking_diagnostic_candidates_per_state != 256
            or int(oracle_arm["candidates_per_state"]) != 256
            or result.random_candidates_per_state != 8
            or result.random_regret_ratio_maximum != 0.75
            or result.pointwise_regret_ratio_maximum != 0.85
            or float(gate["legal_action_rate"]) != 1.0
            or int(gate["simulator_executions_per_state_each_arm"]) != 8
            or int(gate["model_scores_per_state_each_search_arm"]) != 256
            or float(gate["deterministic_training_candidate_replay_rate"]) != 1.0
            or float(gate["deterministic_search_replay_rate"]) != 1.0
            or float(gate["deterministic_dataset_replay_rate"]) != 1.0
            or int(gate["split_leakage_findings"]) != 0
            or not bool(gate["all_metrics_finite"])
            or bool(gate["test_metrics_opened"])
        ):
            raise ValueError("H016 budgets or gates are immutable")
        backbone = H016BackboneConfig.from_yaml(result.backbone_config)
        if backbone.paired_runtime.runtime.training.steps != 600:
            raise ValueError("H016 suite requires the 600-step backbone")
        return result
