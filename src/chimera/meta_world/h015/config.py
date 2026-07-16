"""Frozen backbone and search configuration for CHM-W-H015."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import yaml

from chimera.meta_world.h013.config import H013Arm, H013RunConfig


def _mapping(value: object, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{name} must be a mapping")
    return cast(Mapping[str, Any], value)


@dataclass(frozen=True)
class H015BackboneConfig:
    """H014 factual-residual control retrained under the H015 seed."""

    response_source: str
    paired_runtime: H013RunConfig

    @classmethod
    def from_mapping(cls, values: Mapping[str, Any]) -> H015BackboneConfig:
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
            raise ValueError("H015 backbone fields differ from the frozen schema")
        if values["arm"] != "factual_residual_conditioned_search_backbone":
            raise ValueError("H015 backbone arm is immutable")
        response_source = str(values["response_source"])
        if response_source != "predicted_factual_minus_final_observation":
            raise ValueError("H015 must retain the H014 factual-residual control")
        paired_values = dict(values)
        paired_values.pop("response_source")
        paired_values["arm"] = H013Arm.DIRECT.value
        paired = H013RunConfig.from_mapping(paired_values)
        if paired.runtime.training.seed != 260950:
            raise ValueError("H015 backbone seed is immutable")
        return cls(response_source=response_source, paired_runtime=paired)

    @classmethod
    def from_yaml(cls, path: str | Path) -> H015BackboneConfig:
        with Path(path).open("r", encoding="utf-8") as handle:
            values = yaml.safe_load(handle)
        return cls.from_mapping(_mapping(values, "H015 backbone"))


@dataclass(frozen=True)
class H015SearchConfig:
    """Exact model and simulator budgets for one state."""

    rounds: int
    candidates_per_round: int
    elite_candidates_per_round: int
    simulator_executions_per_state: int
    archive_descriptor: tuple[str, ...]

    @property
    def model_scores_per_state(self) -> int:
        return self.rounds * self.candidates_per_round


@dataclass(frozen=True)
class H015SuiteConfig:
    """Complete development candidate-generation protocol."""

    seed: int
    frozen_validation_seeds: tuple[int, ...]
    generator_config: Path
    integrity_report: Path
    backbone_config: Path
    evaluation_states: int
    oracle_pool_candidates_per_state: int
    search: H015SearchConfig
    uncertainty_beta: float
    mean_only_beta: float
    random_candidates_per_state: int
    random_regret_ratio_maximum: float
    mean_regret_ratio_maximum: float

    @classmethod
    def from_yaml(cls, path: str | Path) -> H015SuiteConfig:
        with Path(path).open("r", encoding="utf-8") as handle:
            values = _mapping(yaml.safe_load(handle), "H015 suite")
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
            "evaluation_states",
            "oracle_pool_candidates_per_state",
            "search",
            "arms",
            "primary_metric",
            "development_gate",
        }
        if set(values) != expected:
            raise ValueError("H015 suite fields differ from preregistration")
        if (
            values["hypothesis_id"] != "CHM-W-H015"
            or values["trial_id"] != "CHM-W-T015"
            or values["mode"] != "development"
            or values["test_access"] != "sealed"
            or values["primary_metric"] != "realized_best_candidate_regret"
        ):
            raise ValueError("H015 suite identity or access boundary is invalid")
        seed = int(values["development_seed"])
        frozen = tuple(int(item) for item in values["frozen_validation_seeds"])
        if seed != 260950 or frozen != (260951, 260952, 260953):
            raise ValueError("H015 registered seeds are immutable")
        search_raw = _mapping(values["search"], "H015 search")
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
        ):
            raise ValueError("H015 search algorithm differs from preregistration")
        arms = _mapping(values["arms"], "H015 arms")
        if set(arms) != {
            "uncertainty_aware_quality_diversity_search",
            "mean_only_quality_diversity_search",
            "legal_random_intervention_selection",
            "fixed_pool_oracle_evaluator",
        }:
            raise ValueError("H015 comparison arms differ from preregistration")
        response = _mapping(
            arms["uncertainty_aware_quality_diversity_search"],
            "uncertainty arm",
        )
        mean = _mapping(
            arms["mean_only_quality_diversity_search"],
            "mean arm",
        )
        random = _mapping(
            arms["legal_random_intervention_selection"],
            "random arm",
        )
        oracle = _mapping(arms["fixed_pool_oracle_evaluator"], "oracle")
        gate = _mapping(values["development_gate"], "H015 gate")
        result = cls(
            seed=seed,
            frozen_validation_seeds=frozen,
            generator_config=Path(str(values["generator_config"])),
            integrity_report=Path(str(values["dataset_integrity_report"])),
            backbone_config=Path(str(values["backbone_config"])),
            evaluation_states=int(values["evaluation_states"]),
            oracle_pool_candidates_per_state=int(
                values["oracle_pool_candidates_per_state"]
            ),
            search=search,
            uncertainty_beta=float(response["uncertainty_beta"]),
            mean_only_beta=float(mean["uncertainty_beta"]),
            random_candidates_per_state=int(random["candidates_per_state"]),
            random_regret_ratio_maximum=float(
                gate["regret_ratio_vs_legal_random_maximum"]
            ),
            mean_regret_ratio_maximum=float(
                gate["regret_ratio_vs_mean_only_search_maximum"]
            ),
        )
        if (
            result.evaluation_states != 32
            or result.oracle_pool_candidates_per_state != 256
            or int(oracle["candidates_per_state"]) != 256
            or result.uncertainty_beta != 1.0
            or result.mean_only_beta != 0.0
            or result.random_candidates_per_state != 8
            or result.random_regret_ratio_maximum != 0.75
            or result.mean_regret_ratio_maximum != 0.90
            or search.model_scores_per_state != 256
        ):
            raise ValueError("H015 comparison budgets or gates are immutable")
        if (
            float(gate["legal_action_rate"]) != 1.0
            or int(gate["simulator_executions_per_state_each_arm"]) != 8
            or int(gate["model_scores_per_state_each_search_arm"]) != 256
            or float(gate["deterministic_search_replay_rate"]) != 1.0
            or float(gate["deterministic_dataset_replay_rate"]) != 1.0
            or int(gate["split_leakage_findings"]) != 0
            or not bool(gate["all_metrics_finite"])
            or bool(gate["test_metrics_opened"])
        ):
            raise ValueError("H015 invariant gates are invalid")
        H015BackboneConfig.from_yaml(result.backbone_config)
        return result
