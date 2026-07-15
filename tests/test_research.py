from __future__ import annotations

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
