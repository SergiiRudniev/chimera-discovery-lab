from __future__ import annotations

import json
from pathlib import Path

import pytest

from chimera.research import load_research_registry


def test_research_registry_is_valid() -> None:
    records = load_research_registry("research/registry.yaml")
    assert [record["id"] for record in records] == [
        "CHM-V-H000",
        "CHM-V-H001",
        "CHM-V-H002",
        "CHM-V-H003",
        "CHM-W-H000",
        "CHM-W-H001",
        "CHM-W-H002",
        "CHM-W-H003",
        "CHM-W-H004",
        "CHM-W-H005",
        "CHM-W-H006",
        "CHM-W-H007",
        "CHM-W-H008",
        "CHM-W-H009",
        "CHM-W-H010",
        "CHM-W-H011",
    ]


def test_duplicate_hypothesis_is_rejected(tmp_path: Path) -> None:
    registry = tmp_path / "registry.yaml"
    entry = (
        "  - id: CHM-V-H000\n"
        "    title: x\n"
        "    status: registered\n"
        "    registered_at: 2026-07-15\n"
        "    config: x.yaml\n"
        "    result: x.json\n"
    )
    registry.write_text("hypotheses:\n" + entry + entry, encoding="utf-8")
    with pytest.raises(ValueError, match="duplicate"):
        load_research_registry(registry)


def test_meta_world_hypothesis_namespace_is_valid(tmp_path: Path) -> None:
    registry = tmp_path / "registry.yaml"
    registry.write_text(
        "hypotheses:\n"
        "  - id: CHM-W-H000\n"
        "    title: Meta-World representation qualification\n"
        "    status: registered\n"
        "    registered_at: 2026-07-15\n"
        "    config: configs/meta_world/meta_world_w0.yaml\n"
        "    result: research/results/CHM-W-H000.json\n",
        encoding="utf-8",
    )

    records = load_research_registry(registry)

    assert records[0]["id"] == "CHM-W-H000"


def test_h002_preflight_cannot_be_mistaken_for_scientific_result() -> None:
    preflight = json.loads(
        Path("research/preflights/CHM-W-H002-validation.json").read_text(encoding="utf-8")
    )
    result = json.loads(Path("research/results/CHM-W-H002.json").read_text(encoding="utf-8"))

    assert preflight["status"] == "completed_validation_preflight"
    assert preflight["scientific_result"] is False
    assert preflight["registered_trial_executed"] is False
    assert preflight["opened_splits"] == ["train", "validation"]
    assert preflight["test_metrics_opened"] is False
    assert preflight["decision"] == "do_not_freeze_T002"
    assert all(not arm["checkpoint"]["promoted"] for arm in preflight["arms"].values())
    assert result["status"] == "not_run"
    assert result["decision"] == "not_run"
    assert result["metrics"] is None


def test_h003_is_registered_before_metrics_are_opened() -> None:
    result = json.loads(Path("research/results/CHM-W-H003.json").read_text(encoding="utf-8"))

    assert result == {
        "claim_boundary": (
            "No result has been run. A future passing result would support transfer only "
            "within the frozen simulator distribution, not real-world causal discovery, "
            "profitable business ideas, language-independent thought or production readiness."
        ),
        "decision": "not_run",
        "id": "CHM-W-H003",
        "metrics": None,
        "status": "not_run",
        "trial_id": "CHM-W-T003",
    }


def test_h003_exploratory_preflight_does_not_open_registered_trial() -> None:
    preflight = json.loads(
        Path("research/preflights/CHM-W-H003-validation.json").read_text(encoding="utf-8")
    )
    result = json.loads(Path("research/results/CHM-W-H003.json").read_text(encoding="utf-8"))

    assert preflight["scientific_result"] is False
    assert preflight["registered_trial_executed"] is False
    assert preflight["remaining_registered_seeds_run"] is False
    assert preflight["test_metrics_opened"] is False
    assert preflight["opened_splits"] == ["train", "validation"]
    assert preflight["decision"] == "do_not_freeze_T003"
    assert all(not arm["checkpoint"]["promoted"] for arm in preflight["arms"].values())
    assert result["status"] == "not_run"
    assert result["metrics"] is None


def test_h004_is_registered_before_dataset_or_metrics_exist() -> None:
    result = json.loads(Path("research/results/CHM-W-H004.json").read_text(encoding="utf-8"))

    assert result["id"] == "CHM-W-H004"
    assert result["trial_id"] == "CHM-W-T004"
    assert result["status"] == "not_run"
    assert result["decision"] == "not_run"
    assert result["metrics"] is None


def test_h004_development_preflight_does_not_open_frozen_validation_or_test() -> None:
    preflight = json.loads(
        Path("research/preflights/CHM-W-H004-development.json").read_text(encoding="utf-8")
    )
    result = json.loads(Path("research/results/CHM-W-H004.json").read_text(encoding="utf-8"))

    assert preflight["scientific_result"] is False
    assert preflight["registered_trial_executed"] is False
    assert preflight["frozen_validation_seeds_opened"] is False
    assert preflight["test_metrics_opened"] is False
    assert preflight["opened_splits"] == ["train", "validation"]
    assert preflight["decision"] == "do_not_freeze_T004"
    assert all(not arm["checkpoint"]["promoted"] for arm in preflight["arms"].values())
    assert result["status"] == "not_run"
    assert result["metrics"] is None


def test_h005_is_registered_before_metrics_are_opened() -> None:
    result = json.loads(Path("research/results/CHM-W-H005.json").read_text(encoding="utf-8"))

    assert result["id"] == "CHM-W-H005"
    assert result["trial_id"] == "CHM-W-T005"
    assert result["status"] == "not_run"
    assert result["decision"] == "not_run"
    assert result["metrics"] is None


def test_h005_failed_validation_gate_keeps_test_sealed() -> None:
    validation = json.loads(
        Path("research/preflights/CHM-W-H005-validation.json").read_text(
            encoding="utf-8"
        )
    )
    result = json.loads(Path("research/results/CHM-W-H005.json").read_text(encoding="utf-8"))

    assert validation["validation_gate"]["passed"] is False
    assert validation["decision"] == "do_not_open_T005_test"
    assert validation["frozen_validation_seeds_opened"] is True
    assert validation["test_metrics_opened"] is False
    assert validation["checkpoint_promoted"] is False
    assert result["status"] == "not_run"
    assert result["metrics"] is None


def test_h006_is_registered_before_metrics_are_opened() -> None:
    result = json.loads(Path("research/results/CHM-W-H006.json").read_text(encoding="utf-8"))

    assert result["id"] == "CHM-W-H006"
    assert result["trial_id"] == "CHM-W-T006"
    assert result["status"] == "not_run"
    assert result["decision"] == "not_run"
    assert result["metrics"] is None


def test_h006_development_failure_keeps_validation_and_test_sealed() -> None:
    preflight = json.loads(
        Path("research/preflights/CHM-W-H006-development.json").read_text(
            encoding="utf-8"
        )
    )
    result = json.loads(Path("research/results/CHM-W-H006.json").read_text(encoding="utf-8"))

    assert preflight["development_gate"]["passed"] is False
    assert preflight["decision"] == "do_not_open_H006_frozen_validation"
    assert preflight["frozen_validation_seeds_opened"] is False
    assert preflight["test_metrics_opened"] is False
    assert preflight["checkpoint_promoted"] is False
    assert result["status"] == "not_run"
    assert result["metrics"] is None


def test_h007_is_registered_before_metrics_are_opened() -> None:
    result = json.loads(Path("research/results/CHM-W-H007.json").read_text(encoding="utf-8"))

    assert result["id"] == "CHM-W-H007"
    assert result["trial_id"] == "CHM-W-T007"
    assert result["status"] == "not_run"
    assert result["decision"] == "not_run"
    assert result["metrics"] is None


def test_h007_development_failure_keeps_validation_and_test_sealed() -> None:
    preflight = json.loads(
        Path("research/preflights/CHM-W-H007-development.json").read_text(
            encoding="utf-8"
        )
    )
    result = json.loads(Path("research/results/CHM-W-H007.json").read_text(encoding="utf-8"))

    assert preflight["development_gate"]["passed"] is False
    assert preflight["decision"] == "do_not_open_H007_frozen_validation"
    assert preflight["frozen_validation_seeds_opened"] is False
    assert preflight["test_metrics_opened"] is False
    assert preflight["checkpoint_promoted"] is False
    assert result["status"] == "not_run"
    assert result["metrics"] is None


def test_h008_is_registered_before_metrics_are_opened() -> None:
    result = json.loads(Path("research/results/CHM-W-H008.json").read_text(encoding="utf-8"))

    assert result["id"] == "CHM-W-H008"
    assert result["trial_id"] == "CHM-W-T008"
    assert result["status"] == "not_run"
    assert result["decision"] == "not_run"
    assert result["metrics"] is None


def test_h009_is_registered_before_metrics_are_opened() -> None:
    result = json.loads(Path("research/results/CHM-W-H009.json").read_text(encoding="utf-8"))

    assert result["id"] == "CHM-W-H009"
    assert result["trial_id"] == "CHM-W-T009"
    assert result["status"] == "not_run"
    assert result["decision"] == "not_run"
    assert result["metrics"] is None


def test_h009_development_failure_keeps_validation_and_test_sealed() -> None:
    preflight = json.loads(
        Path("research/preflights/CHM-W-H009-development.json").read_text(
            encoding="utf-8"
        )
    )
    result = json.loads(Path("research/results/CHM-W-H009.json").read_text(encoding="utf-8"))

    assert preflight["development_gate"]["passed"] is False
    assert preflight["decision"] == "do_not_open_H009_frozen_validation"
    assert preflight["frozen_validation_seeds_opened"] is False
    assert preflight["test_metrics_opened"] is False
    assert preflight["checkpoint_promoted"] is False
    assert result["status"] == "not_run"
    assert result["metrics"] is None


def test_h010_is_registered_before_metrics_are_opened() -> None:
    result = json.loads(Path("research/results/CHM-W-H010.json").read_text(encoding="utf-8"))

    assert result["id"] == "CHM-W-H010"
    assert result["trial_id"] == "CHM-W-T010"
    assert result["status"] == "not_run"
    assert result["decision"] == "not_run"
    assert result["metrics"] is None


def test_h010_development_failure_keeps_validation_and_test_sealed() -> None:
    preflight = json.loads(
        Path("research/preflights/CHM-W-H010-development.json").read_text(
            encoding="utf-8"
        )
    )
    result = json.loads(Path("research/results/CHM-W-H010.json").read_text(encoding="utf-8"))

    assert preflight["development_gate"]["passed"] is False
    assert preflight["decision"] == "do_not_open_H010_frozen_validation"
    assert preflight["frozen_validation_seeds_opened"] is False
    assert preflight["test_metrics_opened"] is False
    assert preflight["checkpoint_promoted"] is False
    assert result["status"] == "not_run"
    assert result["metrics"] is None


def test_h011_is_registered_before_metrics_are_opened() -> None:
    result = json.loads(Path("research/results/CHM-W-H011.json").read_text(encoding="utf-8"))

    assert result["id"] == "CHM-W-H011"
    assert result["trial_id"] == "CHM-W-T011"
    assert result["status"] == "not_run"
    assert result["decision"] == "not_run"
    assert result["metrics"] is None
