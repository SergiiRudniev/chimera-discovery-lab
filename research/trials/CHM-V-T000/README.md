# Venture Trial T0

Venture Trial T0 is the first full engineering qualification of Chimera Venture
M0 on Venture Corpus C0. The frozen protocol ran for 300 optimization steps on
an NVIDIA GeForce RTX 5070 and selected step 175 by validation loss.

## Result

| Check | Result |
| --- | --- |
| Finite training | Passed |
| Training loss reduced | Passed: 7.9839 to 3.3122 |
| Train exact-graph reconstruction | Failed: 0.0% versus 95% threshold |
| Generated-program validity | Passed: 100% |
| Deterministic replay | Passed |

The trial status is `completed_with_gaps`. M0 did not qualify for exact
structural reconstruction and this checkpoint is not an accepted model release.

## Evaluation

| Split | Loss | Operation accuracy | Exact graph | Score MAE |
| --- | ---: | ---: | ---: | ---: |
| Train | 3.3177 | 49.21% | 0.00% | 0.0251 |
| Validation | 3.5473 | 39.01% | 0.00% | 0.0581 |
| Test | 3.8310 | 37.96% | 0.00% | 0.0762 |

Checkpoint selection used validation loss only. The test split was opened after
the step-175 checkpoint had been frozen.

## Structured generation

- 160 candidates from 10 canonical source graphs;
- 0 invalid programs;
- 160 candidates changed their source graph;
- 148 unique resulting graphs;
- 3 observed non-terminal operations: `CONNECT`, `INVERT_RELATION`, `SUBSTITUTE`;
- 7 final MAP-Elites cells across a 4 x 4 archive.

`archive_retained` in `candidates.jsonl` records acceptance at insertion time.
Later candidates can replace an accepted candidate in the same cell; the final
archive contains seven entries.

## Artifacts

- `result.json`: qualification decision and final metrics;
- `metrics.jsonl`: step and validation history;
- `candidates.jsonl`: frozen graph-edit outputs for external interpretation;
- `checkpoint_manifest.json`: expected checkpoint name, size and SHA-256;
- `environment.json`: execution environment;
- `protocol.yaml`: protocol frozen before execution.

The checkpoint is structural-pretraining evidence only. T0 does not evaluate
creativity, commercial utility or the CHM-V-H001 language hypothesis.
