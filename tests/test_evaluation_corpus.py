from __future__ import annotations

from pathlib import Path

import numpy as np

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


def test_numeric_archive_contains_no_text_or_object_arrays() -> None:
    corpus = EvaluationCorpus("datasets/venture_corpus_c1/manifest.json")
    assert len(corpus) == 10
    with np.load("datasets/venture_corpus_c1/graphs.npz", allow_pickle=False) as archive:
        assert all(archive[name].dtype.kind not in {"O", "S", "U"} for name in archive.files)
    assert corpus.graph(0).node_features.shape == (1, 64, 8)
