"""Build and execute the reproducible Venture Corpus C1 quality notebook."""

# mypy: disable-error-code=no-untyped-call

from __future__ import annotations

import argparse
from pathlib import Path

import nbformat
from nbclient import NotebookClient
from nbformat.v4 import new_code_cell, new_markdown_cell, new_notebook


def build_notebook(output: Path, repository: Path) -> None:
    cells = [
        new_markdown_cell(
            """# Venture Corpus C1 — quality audit

## TL;DR

C1 contains 2 calibration and 8 evaluation cases. Numeric graph inputs contain no
text, all sources are later than C0, and organization/CIK/accession overlap is zero.
Candidate generation remains blocked until an independent reviewer verifies the
source-to-graph annotations."""
        ),
        new_markdown_cell(
            """## Context and methods

The intended grain is one organization filing, one registered business challenge
and one typed numeric graph. This notebook reads the committed manifest, safe NPZ
archive, audit sidecar and generated quality report. It does not contact external
services and does not create experiment outcomes."""
        ),
        new_code_cell(
            """from pathlib import Path
import hashlib
import json
import numpy as np

ROOT = Path.cwd()
DATA = ROOT / "datasets" / "venture_corpus_c1"
manifest = json.loads((DATA / "manifest.json").read_text(encoding="utf-8"))
quality = json.loads((DATA / "quality_report.json").read_text(encoding="utf-8"))
case_lines = (DATA / "cases.jsonl").read_text(encoding="utf-8").splitlines()
cases = [json.loads(line) for line in case_lines]
with np.load(DATA / "graphs.npz", allow_pickle=False) as archive:
    arrays = {name: archive[name].copy() for name in archive.files}

print({
    "corpus_id": manifest["corpus_id"],
    "cases": manifest["counts"]["cases"],
    "calibration": manifest["counts"]["calibration"],
    "evaluation": manifest["counts"]["evaluation"],
    "release_status": manifest["release_status"],
})"""
        ),
        new_markdown_cell("## Data"),
        new_code_cell(
            """node_counts = arrays["graph_node_mask"].sum(axis=1)
edge_counts = (arrays["graph_edge_types"] > 0).sum(axis=(1, 2))
print(f"{'case_id':28} {'partition':12} {'period':10} {'nodes':>5} {'edges':>5}")
for record, nodes, edges in zip(cases, node_counts, edge_counts, strict=True):
    print(
        f"{record['case_id']:28} {record['partition']:12} "
        f"{record['source']['period_end']:10} {int(nodes):5d} {int(edges):5d}"
    )"""
        ),
        new_markdown_cell("## Results"),
        new_code_cell(
            """dimensions = quality["dimensions"]
summary = {
    "unique_case_ids": dimensions["uniqueness"]["unique_case_ids"],
    "unique_numeric_graphs": dimensions["uniqueness"]["unique_numeric_graphs"],
    "unique_topologies": dimensions["uniqueness"]["unique_topologies"],
    "feature_range": [
        dimensions["validity"]["feature_min"],
        dimensions["validity"]["feature_max"],
    ],
    "finite_features": dimensions["validity"]["finite_features"],
    "numeric_archive_has_text_or_objects": dimensions["validity"][
        "numeric_archive_has_text_or_objects"
    ],
    "case_aligned_briefs": dimensions["consistency"]["case_aligned_briefs"],
}
print(json.dumps(summary, indent=2))"""
        ),
        new_code_cell(
            """boundary = manifest["temporal_boundary"]
leakage = quality["dimensions"]["leakage"]
print(json.dumps({
    "pretraining_max_period_end": boundary["pretraining_max_period_end"],
    "evaluation_min_period_end": boundary["evaluation_min_period_end"],
    "organization_overlap_count": leakage["organization_overlap_count"],
    "cik_overlap_count": leakage["cik_overlap_count"],
    "accession_overlap_count": leakage["accession_overlap_count"],
    "outcome_labels_present": leakage["outcome_labels_present"],
}, indent=2))"""
        ),
        new_code_cell(
            """for finding in quality["findings"]:
    print(
        f"[{finding['severity'].upper()}] {finding['id']} — "
        f"{finding['status']}: {finding['finding']}"
    )
print()
print("Fitness:")
print(json.dumps(quality["fitness_for_use"], indent=2))"""
        ),
        new_markdown_cell(
            """## Takeaways

- The numeric contract, temporal isolation and matched baseline alignment pass.
- C1 is fit for preregistration, not yet for candidate generation.
- A second reviewer must close `C1-Q001` before the frozen H001 run.
- H001 remains `not_run`; this notebook contains no creativity result."""
        ),
    ]
    notebook = new_notebook(
        cells=cells,
        metadata={
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.12"},
        },
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    nbformat.write(notebook, output)
    client = NotebookClient(notebook, timeout=120, kernel_name="python3")
    client.execute(cwd=str(repository))
    nbformat.write(notebook, output)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("notebooks/venture_corpus_c1_quality.ipynb"),
    )
    arguments = parser.parse_args()
    repository = Path(__file__).resolve().parents[1]
    build_notebook(arguments.output, repository)
    print(arguments.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
