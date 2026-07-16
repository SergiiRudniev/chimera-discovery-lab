"""Frozen candidate-pool and comparison configuration for CHM-W-H017."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import yaml

from chimera.meta_world.h015.config import H015SearchConfig
from chimera.meta_world.h016.config import H016SuiteConfig


def _mapping(value: object, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{name} must be a mapping")
    return cast(Mapping[str, Any], value)


@dataclass(frozen=True)
class H017SupportPoolConfig:
    """Balanced discrete and Latin-hypercube continuous candidate support."""

    candidates_per_state: int
    discrete_design: str
    maximum_pair_count_discrepancy: int
    continuous_design: str
    magnitude_interval: str
    control_interval: str
    exact_continuous_boundary_rate: float
    unique_vector_rate: float
    seed_offset: int
    seed_stride: int

    @classmethod
    def from_mapping(cls, values: Mapping[str, Any]) -> H017SupportPoolConfig:
        expected = {
            "candidates_per_state",
            "discrete_design",
            "maximum_pair_count_discrepancy",
            "continuous_design",
            "magnitude_interval",
            "control_interval",
            "exact_continuous_boundary_rate",
            "unique_vector_rate",
            "seed_offset",
            "seed_stride",
        }
        if set(values) != expected:
            raise ValueError("H017 support-pool fields differ from preregistration")
        result = cls(
            candidates_per_state=int(values["candidates_per_state"]),
            discrete_design=str(values["discrete_design"]),
            maximum_pair_count_discrepancy=int(
                values["maximum_pair_count_discrepancy"]
            ),
            continuous_design=str(values["continuous_design"]),
            magnitude_interval=str(values["magnitude_interval"]),
            control_interval=str(values["control_interval"]),
            exact_continuous_boundary_rate=float(
                values["exact_continuous_boundary_rate"]
            ),
            unique_vector_rate=float(values["unique_vector_rate"]),
            seed_offset=int(values["seed_offset"]),
            seed_stride=int(values["seed_stride"]),
        )
        if (
            result.candidates_per_state != 256
            or result.discrete_design != "balanced_ordered_source_target_pairs"
            or result.maximum_pair_count_discrepancy != 1
            or result.continuous_design != "seeded_independent_Latin_hypercube"
            or result.magnitude_interval != "open_0_1"
            or result.control_interval != "open_minus1_1"
            or result.exact_continuous_boundary_rate != 0.0
            or result.unique_vector_rate != 1.0
            or result.seed_offset != 41
            or result.seed_stride != 1_000_003
        ):
            raise ValueError("H017 support-pool protocol is immutable")
        return result


@dataclass(frozen=True)
class H017PoolRerankingConfig:
    """One-pass quality-diversity selection contract."""

    simulator_executions_per_state: int
    archive_descriptor: tuple[str, ...]
    archive_retention: str


@dataclass(frozen=True)
class H017SuiteConfig:
    """Complete H017 development protocol."""

    seed: int
    frozen_validation_seeds: tuple[int, ...]
    generator_config: Path
    integrity_report: Path
    critic_suite_config: Path
    support_pool: H017SupportPoolConfig
    pool_reranking: H017PoolRerankingConfig
    adaptive_search: H015SearchConfig
    adaptive_seed_offset: int
    oracle_pool_candidates_per_state: int
    oracle_seed_offset: int
    random_candidates_per_state: int
    evaluation_states: int
    random_regret_ratio_maximum: float
    adaptive_regret_ratio_maximum: float

    @classmethod
    def from_yaml(cls, path: str | Path) -> H017SuiteConfig:
        with Path(path).open("r", encoding="utf-8") as handle:
            values = _mapping(yaml.safe_load(handle), "H017 suite")
        expected = {
            "hypothesis_id",
            "trial_id",
            "mode",
            "development_seed",
            "frozen_validation_seeds",
            "test_access",
            "generator_config",
            "dataset_integrity_report",
            "critic_suite_config",
            "critic_training_seed",
            "evaluation_states",
            "support_pool",
            "pool_reranking",
            "adaptive_control",
            "oracle_pool",
            "legal_random",
            "primary_metric",
            "development_gate",
        }
        if set(values) != expected:
            raise ValueError("H017 suite fields differ from preregistration")
        if (
            values["hypothesis_id"] != "CHM-W-H017"
            or values["trial_id"] != "CHM-W-T017"
            or values["mode"] != "development"
            or values["test_access"] != "sealed"
            or values["primary_metric"] != "realized_best_candidate_regret"
            or int(values["critic_training_seed"]) != 260954
        ):
            raise ValueError("H017 suite identity or access boundary is invalid")
        seed = int(values["development_seed"])
        frozen = tuple(int(item) for item in values["frozen_validation_seeds"])
        if seed != 260958 or frozen != (260959, 260960, 260961):
            raise ValueError("H017 registered seeds are immutable")
        support_pool = H017SupportPoolConfig.from_mapping(
            _mapping(values["support_pool"], "H017 support pool")
        )
        reranking_raw = _mapping(values["pool_reranking"], "H017 reranking")
        if set(reranking_raw) != {
            "simulator_executions_per_state",
            "archive_descriptor",
            "archive_retention",
        }:
            raise ValueError("H017 reranking fields differ from preregistration")
        pool_reranking = H017PoolRerankingConfig(
            simulator_executions_per_state=int(
                reranking_raw["simulator_executions_per_state"]
            ),
            archive_descriptor=tuple(
                str(item) for item in reranking_raw["archive_descriptor"]
            ),
            archive_retention=str(reranking_raw["archive_retention"]),
        )
        adaptive_raw = _mapping(values["adaptive_control"], "H017 adaptive")
        adaptive_search = H015SearchConfig(
            rounds=int(adaptive_raw["rounds"]),
            candidates_per_round=int(adaptive_raw["candidates_per_round"]),
            elite_candidates_per_round=int(
                adaptive_raw["elite_candidates_per_round"]
            ),
            simulator_executions_per_state=int(
                adaptive_raw["simulator_executions_per_state"]
            ),
            archive_descriptor=tuple(
                str(item) for item in adaptive_raw["archive_descriptor"]
            ),
        )
        oracle = _mapping(values["oracle_pool"], "H017 oracle")
        random = _mapping(values["legal_random"], "H017 random")
        gate = _mapping(values["development_gate"], "H017 gate")
        result = cls(
            seed=seed,
            frozen_validation_seeds=frozen,
            generator_config=Path(str(values["generator_config"])),
            integrity_report=Path(str(values["dataset_integrity_report"])),
            critic_suite_config=Path(str(values["critic_suite_config"])),
            support_pool=support_pool,
            pool_reranking=pool_reranking,
            adaptive_search=adaptive_search,
            adaptive_seed_offset=int(adaptive_raw["seed_offset"]),
            oracle_pool_candidates_per_state=int(oracle["candidates_per_state"]),
            oracle_seed_offset=int(oracle["seed_offset"]),
            random_candidates_per_state=int(random["candidates_per_state"]),
            evaluation_states=int(values["evaluation_states"]),
            random_regret_ratio_maximum=float(
                gate["regret_ratio_vs_legal_random_maximum"]
            ),
            adaptive_regret_ratio_maximum=float(
                gate["regret_ratio_vs_H016_adaptive_CEM_maximum"]
            ),
        )
        descriptor = ("source_slot", "target_slot", "magnitude_quartile")
        if (
            pool_reranking.simulator_executions_per_state != 8
            or pool_reranking.archive_descriptor != descriptor
            or pool_reranking.archive_retention != "maximum_rank_logit_per_cell"
            or adaptive_search.rounds != 4
            or adaptive_search.candidates_per_round != 64
            or adaptive_search.elite_candidates_per_round != 8
            or adaptive_search.simulator_executions_per_state != 8
            or adaptive_search.archive_descriptor != descriptor
            or adaptive_search.model_scores_per_state != 256
            or result.adaptive_seed_offset != 29
            or result.oracle_pool_candidates_per_state != 256
            or result.oracle_seed_offset != 11
            or random != {"source": "first_from_support_pool", "candidates_per_state": 8}
            or result.evaluation_states != 32
            or result.random_regret_ratio_maximum != 0.75
            or result.adaptive_regret_ratio_maximum != 0.85
        ):
            raise ValueError("H017 comparison protocol is immutable")
        if (
            float(gate["legal_action_rate"]) != 1.0
            or float(gate["support_pool_replay_rate"]) != 1.0
            or float(gate["support_pool_exact_boundary_rate"]) != 0.0
            or float(gate["support_pool_unique_vector_rate"]) != 1.0
            or int(gate["support_pool_pair_count_discrepancy_maximum"]) != 1
            or int(gate["simulator_executions_per_state_each_arm"]) != 8
            or int(gate["model_scores_per_state_each_learned_arm"]) != 256
            or float(gate["deterministic_training_candidate_replay_rate"]) != 1.0
            or float(gate["deterministic_search_replay_rate"]) != 1.0
            or float(gate["deterministic_dataset_replay_rate"]) != 1.0
            or int(gate["split_leakage_findings"]) != 0
            or not bool(gate["all_metrics_finite"])
            or bool(gate["test_metrics_opened"])
        ):
            raise ValueError("H017 hard gates are immutable")
        critic = H016SuiteConfig.from_yaml(result.critic_suite_config)
        if critic.seed != 260954 or critic.ranking.steps != 600:
            raise ValueError("H017 must retrain the exact H016 critic")
        return result
