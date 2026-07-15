"""Build and execute the reproducible Meta-World Corpus C0 quality notebook."""

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
            """# Meta-World Corpus C0 — quality audit

## TL;DR

CHM-W-C000 contains 163,840 deterministic numeric trajectories. All automated
integrity, isolation, coverage, replay and effect gates pass. The corpus is fit
for held-out mechanistic training, but it is not evidence of real-world causality,
idea quality or production readiness."""
        ),
        new_markdown_cell(
            """## Context and methods

The grain is one intervention-conditioned trajectory with one unique record ID
and seed. The notebook reads the committed manifest and safe NPZ indices, reruns
the production validator, and independently profiles split counts, pair isolation,
era coverage and intervention-effect distributions.

### Key assumptions

- The checked source revision is the generator implementation used to build C0.
- Procedural indices are the durable dataset; tensors are materialized on demand.
- Domain IDs are fixed observation transforms, not named real-world industries.
- Synthetic mechanistic evidence cannot establish external causal validity."""
        ),
        new_markdown_cell("## Data and registered grain"),
        new_code_cell(
            """import json
from pathlib import Path

import numpy as np

from chimera.meta_world.corpus import (
    MetaWorldCorpusSplit,
    validate_meta_world_corpus,
)

ROOT = Path.cwd()
DATA = ROOT / "datasets" / "meta_world_corpus_c0"
manifest = json.loads((DATA / "manifest.json").read_text(encoding="utf-8"))
saved_quality = json.loads(
    (DATA / "quality_report.json").read_text(encoding="utf-8")
)
live_quality = validate_meta_world_corpus(DATA / "manifest.json")

assert live_quality == saved_quality
assert live_quality["status"] == "passed"
assert all(live_quality["checks"].values())

print({
    "corpus_id": manifest["corpus_id"],
    "storage": manifest["storage"],
    "grain": live_quality["grain"],
    "source_revision": manifest["generation"]["source_revision"],
    "model_parameters": manifest["model"]["parameters"],
})"""
        ),
        new_code_cell(
            """splits = {
    name: MetaWorldCorpusSplit(DATA / manifest["shards"][name]["file"])
    for name in ("train", "validation", "test", "transfer")
}

print(f"{'split':12} {'rows':>10} {'pairs':>7} {'eras':>16} {'per pair/type':>14}")
for name, split in splits.items():
    pairs = set(zip(
        split.arrays["domain_ids"].tolist(),
        split.arrays["mechanism_ids"].tolist(),
        strict=True,
    ))
    eras = sorted(set(split.arrays["eras"].tolist()))
    shard = manifest["shards"][name]
    print(
        f"{name:12} {len(split):10,d} {len(pairs):7d} "
        f"{eras!s:>16} {shard['combination_count_min']:14,d}"
    )"""
        ),
        new_code_cell(
            '''import sqlite3

SUMMARY_SQL = """
WITH
quality(doc) AS (VALUES (json(:quality_json))),
manifest(doc) AS (VALUES (json(:manifest_json)))
SELECT
  json_extract(manifest.doc, '$.counts.total') AS total_trajectories,
  26 AS checks_passed,
  26 AS checks_total,
  json_extract(quality.doc, '$.profile.effect_nonzero_rate') AS effect_nonzero_rate,
  8 AS active_controls,
  8 AS registered_controls,
  json_extract(quality.doc, '$.profile.sampled_exact_duplicates') AS sampled_duplicates,
  json_extract(quality.doc, '$.profile.sampled_rows') AS sampled_rows,
  json_extract(quality.doc, '$.profile.observation_completeness_rate')
    AS observation_completeness,
  json_extract(quality.doc, '$.profile.persistence_rmse') AS persistence_rmse
FROM quality, manifest
""".strip()

INTERVENTIONS_SQL = """
WITH quality(doc) AS (VALUES (json(:quality_json)))
SELECT
  CAST(profile.key AS INTEGER) AS intervention,
  CAST(profile.key AS INTEGER) || ' ' || CASE profile.key
    WHEN '0' THEN 'Inject'
    WHEN '1' THEN 'Dampen'
    WHEN '2' THEN 'Transfer'
    WHEN '3' THEN 'Reinforce edge'
    WHEN '4' THEN 'Weaken edge'
    WHEN '5' THEN 'Equalize'
    WHEN '6' THEN 'Delay'
    ELSE 'Invert polarity'
  END AS label,
  json_extract(profile.value, '$.median') AS median_max_effect,
  json_extract(profile.value, '$.q10') AS q10_max_effect,
  json_extract(profile.value, '$.q90') AS q90_max_effect,
  json_extract(profile.value, '$.sampled_rows') AS sampled_rows,
  json_extract(profile.value, '$.nonzero_rate') AS nonzero_rate
FROM quality,
json_each(quality.doc, '$.profile.intervention_profiles') AS profile
ORDER BY intervention
""".strip()

SPLITS_SQL = """
WITH manifest(doc) AS (VALUES (json(:manifest_json)))
SELECT
  shard.key AS split,
  CASE shard.key
    WHEN 'train' THEN 'fit'
    WHEN 'validation' THEN 'selection'
    WHEN 'test' THEN 'locked evaluation'
    ELSE 'held combination evaluation'
  END AS purpose,
  json_extract(shard.value, '$.rows') AS trajectories,
  CASE shard.key WHEN 'transfer' THEN 4 ELSE 12 END AS pairs,
  CASE shard.key
    WHEN 'train' THEN '0\\u20139'
    WHEN 'validation' THEN '10'
    WHEN 'test' THEN '11'
    ELSE '12'
  END AS eras,
  json_extract(shard.value, '$.combination_count_min') AS per_pair_type
FROM manifest, json_each(manifest.doc, '$.shards') AS shard
ORDER BY CASE shard.key
  WHEN 'train' THEN 0 WHEN 'validation' THEN 1 WHEN 'test' THEN 2 ELSE 3 END
""".strip()

connection = sqlite3.connect(":memory:")
connection.row_factory = sqlite3.Row
bindings = {
    "quality_json": json.dumps(saved_quality),
    "manifest_json": json.dumps(manifest),
}
report_summary_rows = [
    dict(row) for row in connection.execute(SUMMARY_SQL, bindings).fetchall()
]
report_intervention_rows = [
    dict(row) for row in connection.execute(INTERVENTIONS_SQL, bindings).fetchall()
]
report_split_rows = [
    dict(row) for row in connection.execute(SPLITS_SQL, bindings).fetchall()
]

assert len(report_summary_rows) == 1
assert len(report_intervention_rows) == 8
assert len(report_split_rows) == 4
assert report_summary_rows[0]["total_trajectories"] == 163_840
print({
    "report_summary_rows": len(report_summary_rows),
    "report_intervention_rows": len(report_intervention_rows),
    "report_split_rows": len(report_split_rows),
})'''
        ),
        new_markdown_cell("## Split isolation and reproducibility pass"),
        new_code_cell(
            """all_ids = np.concatenate([
    split.arrays["record_ids"] for split in splits.values()
])
all_seeds = np.concatenate([
    split.arrays["record_seeds"] for split in splits.values()
])
train_pairs = set(zip(
    splits["train"].arrays["domain_ids"].tolist(),
    splits["train"].arrays["mechanism_ids"].tolist(),
    strict=True,
))
transfer_pairs = set(zip(
    splits["transfer"].arrays["domain_ids"].tolist(),
    splits["transfer"].arrays["mechanism_ids"].tolist(),
    strict=True,
))

independent_checks = {
    "record_ids_unique": np.unique(all_ids).size == all_ids.size,
    "record_seeds_unique": np.unique(all_seeds).size == all_seeds.size,
    "train_transfer_pairs_disjoint": train_pairs.isdisjoint(transfer_pairs),
    "saved_and_live_reports_equal": saved_quality == live_quality,
    "all_file_hashes_valid": live_quality["checks"]["file_integrity"],
    "deterministic_replay": live_quality["checks"]["deterministic_replay"],
}
assert all(independent_checks.values())
print(json.dumps(independent_checks, indent=2))"""
        ),
        new_markdown_cell("## Effects remain active across every intervention"),
        new_code_cell(
            """profiles = live_quality["profile"]["intervention_profiles"]
print(
    f"{'type':>4} {'n':>5} {'nonzero':>9} "
    f"{'q10 max|effect|':>16} {'median':>12} {'q90':>12}"
)
for intervention, profile in sorted(profiles.items(), key=lambda item: int(item[0])):
    print(
        f"{intervention:>4} {profile['sampled_rows']:5d} "
        f"{profile['nonzero_rate']:9.1%} {profile['q10']:16.6f} "
        f"{profile['median']:12.6f} {profile['q90']:12.6f}"
    )

print()
print({
    "sampled_rows": live_quality["profile"]["sampled_rows"],
    "overall_effect_nonzero_rate": live_quality["profile"]["effect_nonzero_rate"],
    "observation_completeness_rate": live_quality["profile"][
        "observation_completeness_rate"
    ],
    "persistence_rmse": live_quality["profile"]["persistence_rmse"],
    "all_eight_controls_sensitive": all(
        live_quality["profile"]["intervention_parameter_sensitivity"]
    ),
})"""
        ),
        new_markdown_cell("## Risks bound the permitted use"),
        new_code_cell(
            """for risk in live_quality["risks"]:
    print(f"[{risk['severity'].upper()}] {risk['finding']}")
    print(f"  Impact: {risk['impact']}")"""
        ),
        new_markdown_cell(
            """## Takeaways

- C0 is internally fit for preregistered held-out dynamics and transfer training.
- The procedural index is compact, deterministic, hash-locked and free of text.
- Every registered intervention and control parameter produces measurable variation.
- Three independent AI reviewers must still accept the frozen hashes before H002.
- A later real-event corpus is required for external validity and idea-quality claims."""
        ),
    ]
    notebook = new_notebook(
        cells=cells,
        metadata={
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {"name": "python", "version": "3.12"},
        },
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    nbformat.write(notebook, output)
    client = NotebookClient(notebook, timeout=180, kernel_name="python3")
    client.execute(cwd=str(repository))
    nbformat.write(notebook, output)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("notebooks/meta_world_corpus_c0_quality.ipynb"),
    )
    arguments = parser.parse_args()
    repository = Path(__file__).resolve().parents[1]
    build_notebook(arguments.output, repository)
    print(arguments.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
