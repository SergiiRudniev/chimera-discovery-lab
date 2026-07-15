from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import yaml

from chimera.data.evaluation import (
    EvaluationCorpus,
    build_evaluation_corpus,
    validate_evaluation_corpus,
)


def test_committed_evaluation_corpus_validates() -> None:
    assert validate_evaluation_corpus("datasets/venture_corpus_c1/manifest.json") == {
        "cases": 10,
        "calibration": 2,
        "evaluation": 8,
    }


def test_evaluation_build_is_deterministic(tmp_path: Path) -> None:
    first = tmp_path / "first"
    second = tmp_path / "second"
    first_manifest = build_evaluation_corpus(
        "datasets/venture_corpus_c1/source_cases.yaml",
        first,
        pretraining_manifest_path="datasets/venture_corpus_c0/manifest.json",
    )
    second_manifest = build_evaluation_corpus(
        "datasets/venture_corpus_c1/source_cases.yaml",
        second,
        pretraining_manifest_path="datasets/venture_corpus_c0/manifest.json",
    )
    assert first_manifest == second_manifest
    assert b"\r\n" not in (first / "manifest.json").read_bytes()
    assert (first / "manifest.json").read_bytes() == (second / "manifest.json").read_bytes()


def test_numeric_archive_contains_no_text_or_object_arrays() -> None:
    corpus = EvaluationCorpus("datasets/venture_corpus_c1/manifest.json")
    assert len(corpus) == 10
    with np.load("datasets/venture_corpus_c1/graphs.npz", allow_pickle=False) as archive:
        assert all(archive[name].dtype.kind not in {"O", "S", "U"} for name in archive.files)
    assert corpus.graph(0).node_features.shape == (1, 64, 8)


@pytest.mark.parametrize(
    "edge",
    (
        ["value_need", "DEPENDS_ON", "customer_value"],
        ["delivery_cost", "REDUCES", "customer_frequency"],
        ["customer", "TRANSFERS_TO", "store_network"],
    ),
)
def test_known_causal_edge_antipatterns_are_rejected(
    tmp_path: Path, edge: list[str]
) -> None:
    source = yaml.safe_load(
        Path("datasets/venture_corpus_c1/source_cases.yaml").read_text(encoding="utf-8")
    )
    source["cases"][0]["edges"].append(edge)
    source_path = tmp_path / "source_cases.yaml"
    source_path.write_text(yaml.safe_dump(source), encoding="utf-8")
    with pytest.raises(ValueError, match="known causal edge anti-pattern"):
        build_evaluation_corpus(
            source_path,
            tmp_path / "output",
            pretraining_manifest_path="datasets/venture_corpus_c0/manifest.json",
        )


def test_enables_target_role_is_enforced(tmp_path: Path) -> None:
    source = yaml.safe_load(
        Path("datasets/venture_corpus_c1/source_cases.yaml").read_text(encoding="utf-8")
    )
    source["cases"][0]["edges"].append(
        ["inventory_data", "ENABLES", "omnichannel"]
    )
    source_path = tmp_path / "source_cases.yaml"
    source_path.write_text(yaml.safe_dump(source), encoding="utf-8")
    with pytest.raises(ValueError, match="ENABLES target must be an action or resource"):
        build_evaluation_corpus(
            source_path,
            tmp_path / "output",
            pretraining_manifest_path="datasets/venture_corpus_c0/manifest.json",
        )
