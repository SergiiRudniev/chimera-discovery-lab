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
