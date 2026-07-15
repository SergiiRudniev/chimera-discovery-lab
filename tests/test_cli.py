from __future__ import annotations

import json
from pathlib import Path

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
    assert json.loads(captured.out) == {"validated_hypotheses": 9}


def test_corpus_cli_validates_dataset(capsys: object) -> None:
    assert main(
        ["validate-corpus", "--manifest", "datasets/venture_corpus_c0/manifest.json"]
    ) == 0
    captured = capsys.readouterr()  # type: ignore[attr-defined]
    assert json.loads(captured.out) == {"canonical_graphs": 10, "transitions": 640}


def test_evaluation_corpus_cli_validates_dataset(capsys: object) -> None:
    assert main(
        [
            "validate-evaluation-corpus",
            "--manifest",
            "datasets/venture_corpus_c1/manifest.json",
        ]
    ) == 0
    captured = capsys.readouterr()  # type: ignore[attr-defined]
    assert json.loads(captured.out) == {"calibration": 2, "cases": 10, "evaluation": 8}


def test_smoke_cli_runs_one_step(capsys: object) -> None:
    assert main(
        ["smoke", "--config", "configs/venture/venture_smoke.yaml", "--steps", "1"]
    ) == 0
    captured = capsys.readouterr()  # type: ignore[attr-defined]
    lines = captured.out.strip().splitlines()
    summary = json.loads(lines[-1])["summary"]
    assert summary["steps"] == 1
    assert summary["finite"] is True


def test_meta_world_inspect_reports_w0_contract(capsys: object) -> None:
    assert main(
        ["meta-world-inspect", "--config", "configs/meta_world/meta_world_w0.yaml"]
    ) == 0
    captured = capsys.readouterr()  # type: ignore[attr-defined]
    payload = json.loads(captured.out)
    assert payload["model"] == "Chimera Meta-World W0"
    assert 50_000_000 <= payload["trainable_parameters"] <= 80_000_000
    assert payload["language_inputs"] is False
def test_world_generator_smoke_cli_reports_numeric_contract(capsys: object) -> None:
    assert main(["world-generator-smoke", "--batch-size", "4"]) == 0
    captured = capsys.readouterr()  # type: ignore[attr-defined]
    payload = json.loads(captured.out)
    assert payload["dataset_id"] == "CHM-W-WG0"
    assert payload["observations"] == [4, 8, 10, 8]
    assert payload["actions"] == [4, 8, 2]
    assert payload["outcomes"] == [4, 8, 4]
    assert payload["all_finite"] is True
    assert payload["language_inputs"] is False


def test_world_generator_fixed_dataset_cli(tmp_path: Path, capsys: object) -> None:
    assert main(
        [
            "build-world-generator-dataset",
            "--output",
            str(tmp_path),
            "--trajectories-per-split",
            "8",
        ]
    ) == 0
    build_payload = json.loads(capsys.readouterr().out)  # type: ignore[attr-defined]
    assert build_payload["total"] == 40
    assert main(
        [
            "validate-world-generator-dataset",
            "--manifest",
            str(tmp_path / "manifest.json"),
        ]
    ) == 0
    validate_payload = json.loads(capsys.readouterr().out)  # type: ignore[attr-defined]
    assert validate_payload["status"] == "passed"
