# Venture Trial T1

Venture Trial T1 is the corrective structural-reconstruction qualification of
Chimera Venture M0 on Venture Corpus C0. The frozen protocol used operation-
conditioned argument supervision, cosine learning-rate decay and 3,000 GPU
steps. Checkpoint step 2700 was selected by validation exact-graph rate.

## Result

| Check | Result |
| --- | --- |
| Finite training | Passed |
| Training loss reduced | Passed: 7.9580 to 0.1474 |
| Train exact-graph reconstruction | Passed: 99.22% versus 95% threshold |
| Generated-program validity | Passed: 100% |
| Deterministic replay | Passed |

The trial status is `passed`. This qualifies structural reconstruction and
constrained generation only; it does not qualify creativity or commercial use.

## Evaluation

| Split | Loss | Operation accuracy | Exact graph | Exact program | Score MAE |
| --- | ---: | ---: | ---: | ---: | ---: |
| Train | 0.1519 | 100.00% | 99.22% | 99.22% | 0.0014 |
| Validation | 6.6403 | 64.14% | 30.47% | 26.56% | 0.0489 |
| Test | 7.0026 | 59.95% | 14.84% | 14.06% | 0.0559 |

Checkpoint selection used validation exact-graph rate and validation loss only
as a tie-breaker. The test split was opened after the checkpoint was frozen.

## Structured generation

- 160 candidates from 10 canonical source graphs;
- 0 invalid programs;
- 144 candidates changed their source graph;
- 53 unique resulting graphs;
- 3 observed non-terminal operations: `CONNECT`, `INVERT_RELATION`, `SUBSTITUTE`;
- 3 final MAP-Elites cells across a 4 x 4 archive.

T1 improved reconstruction but reduced generation diversity relative to T0:
unique-graph rate fell from 92.50% to 33.13%. This trade-off is a measured
regression and must be addressed before creativity evaluation.

## Artifacts

- `result.json`: qualification decision and final metrics;
- `metrics.jsonl`: step and validation history;
- `candidates.jsonl`: frozen graph-edit outputs for external interpretation;
- `checkpoint_manifest.json`: expected checkpoint name, size and SHA-256;
- `environment.json`: execution environment;
- `protocol.yaml`: protocol frozen before execution.

The checkpoint is an engineering prerelease. T1 does not evaluate the
CHM-V-H001 language hypothesis.
