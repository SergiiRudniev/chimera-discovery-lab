# Research Protocol

## Registration

Before target metrics are opened, add an immutable hypothesis file, config and
`not_run` result record. The registry entry must specify the primary metric,
guardrails, data boundary and decision rule.

## Evaluation

The primary CHM-V-H001 comparison holds source information and evaluation budget
constant between:

1. a text-generating baseline;
2. Chimera graph-edit generation with language introduced only after freezing.

Reviewers see independently rendered outputs in randomized order. Novelty is
reported together with feasibility, utility, within-batch diversity and nearest
training-case distance. Structural distance is evaluated separately from text
embedding distance to avoid making language the only novelty judge.

For `CHM-V-H001`, Corpus C1 fixes two calibration cases and eight evaluation
cases. Each arm generates eight candidates per case from the same typed graph,
objective and constraint. The text baseline model revision, generation seeds,
rating dimensions, invalidity rules and analysis are frozen in the C1 dataset
card before candidate generation.

The primary effect is the paired case-level novelty difference after the frozen
feasibility threshold and matched-count rule. Acceptance additionally requires
the one-sided 90% case-cluster bootstrap lower bound for feasibility to remain
above the registered -0.5 non-inferiority margin. Calibration cases never enter
the primary estimate.

## Decisions

- `accepted`: preregistered primary rule passed.
- `rejected`: primary rule failed.
- `inconclusive`: execution completed but evidence cannot decide the claim.
- `not_run`: no target result was opened.

Engineering smoke tests cannot promote a research claim.

## Integrity

All attempted hypotheses remain visible. Test labels cannot select checkpoints,
thresholds, archive dimensions or interpreter prompts. Multiple comparisons and
repeated human ratings must be reported.
