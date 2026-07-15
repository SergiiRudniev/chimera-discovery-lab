from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


def test_target_quality_report_is_current() -> None:
    completed = subprocess.run(
        [sys.executable, "scripts/profile_venture_targets.py"],
        check=True,
        capture_output=True,
        text=True,
    )
    generated = json.loads(completed.stdout)
    committed = json.loads(
        Path("datasets/venture_corpus_c0/target_quality_report.json").read_text(encoding="utf-8")
    )
    assert generated == committed
    assert generated["splits"]["train"]["target_graph_majority_upper_bound"] == 1.0
    assert generated["splits"]["train"]["registered_program_majority_upper_bound"] < 1.0
    assert not any(generated["integrity"]["cross_split_input_overlap"].values())
