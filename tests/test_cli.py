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
    assert json.loads(captured.out) == {"validated_hypotheses": 20}


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


def test_h009_smoke_dataset_cli(tmp_path: Path, capsys: object) -> None:
    assert main(
        [
            "meta-world-h009-smoke-dataset",
            "--output",
            str(tmp_path),
            "--trajectories-per-split",
            "16",
        ]
    ) == 0
    payload = json.loads(capsys.readouterr().out)  # type: ignore[attr-defined]

    assert payload["dataset_id"] == "CHM-W-WG2"
    assert payload["hypothesis_id"] == "CHM-W-H009"
    assert payload["status"] == "passed"
    assert payload["scientific_result"] is False
    assert payload["checks"]["paired_renderer_trajectory_consistency"] is True


def test_h012_smoke_dataset_cli(tmp_path: Path, capsys: object) -> None:
    assert main(
        [
            "meta-world-h012-smoke-dataset",
            "--output",
            str(tmp_path),
            "--trajectories-per-split",
            "16",
        ]
    ) == 0
    payload = json.loads(capsys.readouterr().out)  # type: ignore[attr-defined]

    assert payload["dataset_id"] == "CHM-W-WG3"
    assert payload["hypothesis_id"] == "CHM-W-H012"
    assert payload["status"] == "passed"
    assert payload["scientific_result"] is False
    assert payload["checks"]["service_metadata_excluded_from_model_batch"] is True


def test_h010_preflight_cli_reports_shared_path(tmp_path: Path, capsys: object) -> None:
    assert main(
        [
            "meta-world-h010-preflight",
            "--config",
            "configs/meta_world/world_h010_development_smoke.yaml",
            "--output",
            str(tmp_path),
        ]
    ) == 0
    payload = json.loads(capsys.readouterr().out)  # type: ignore[attr-defined]

    assert payload["hypothesis_id"] == "CHM-W-H010"
    assert payload["model_variant"] == "shared_aligned_bottleneck"
    assert payload["projection_prediction_delta"] > 1e-6
    assert payload["test_metrics_opened"] is False


def test_h011_preflight_cli_reports_response_consistency(
    tmp_path: Path,
    capsys: object,
) -> None:
    assert main(
        [
            "meta-world-h011-preflight",
            "--config",
            "configs/meta_world/world_h011_development_smoke.yaml",
            "--output",
            str(tmp_path),
        ]
    ) == 0
    payload = json.loads(capsys.readouterr().out)  # type: ignore[attr-defined]

    assert payload["hypothesis_id"] == "CHM-W-H011"
    assert payload["response_consistency_weight"] == 1.0
    assert payload["paired_effect_mean_disagreement"] >= 0.0
    assert payload["test_metrics_opened"] is False


def test_world_probe_fixed_dataset_cli(tmp_path: Path, capsys: object) -> None:
    assert main(
        [
            "build-world-probe-dataset",
            "--output",
            str(tmp_path),
            "--trajectories-per-split",
            "8",
        ]
    ) == 0
    build_payload = json.loads(capsys.readouterr().out)  # type: ignore[attr-defined]
    assert build_payload["dataset_id"] == "CHM-W-WG1"
    assert main(
        [
            "validate-world-probe-dataset",
            "--manifest",
            str(tmp_path / "manifest.json"),
        ]
    ) == 0
    validation = json.loads(capsys.readouterr().out)  # type: ignore[attr-defined]
    assert validation["status"] == "passed"
    assert validation["probe_response_separation"] > 0.0
