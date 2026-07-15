from __future__ import annotations

import json

from chimera.cli import main


def test_inspect_cli_reports_registered_model(capsys: object) -> None:
    assert main(
        ["inspect", "--config", "configs/venture/venture_m0_20m.yaml"]
    ) == 0
    captured = capsys.readouterr()  # type: ignore[attr-defined]
    payload = json.loads(captured.out)
    assert payload["trainable_parameters"] == 20_647_992


def test_research_cli_validates_registry(capsys: object) -> None:
    assert main(["validate-research", "--registry", "research/registry.yaml"]) == 0
    captured = capsys.readouterr()  # type: ignore[attr-defined]
    assert json.loads(captured.out) == {"validated_hypotheses": 4}


def test_smoke_cli_runs_one_step(capsys: object) -> None:
    assert main(
        ["smoke", "--config", "configs/venture/venture_smoke.yaml", "--steps", "1"]
    ) == 0
    captured = capsys.readouterr()  # type: ignore[attr-defined]
    lines = captured.out.strip().splitlines()
    summary = json.loads(lines[-1])["summary"]
    assert summary["steps"] == 1
    assert summary["finite"] is True
