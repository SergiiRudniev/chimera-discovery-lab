# Meta-World Corpus C0

`CHM-W-C000` is the first procedural trajectory corpus for Chimera Meta-World W0.

## Grain

One row is one numeric, intervention-conditioned trajectory identified by a unique
`record_id` and `record_seed`. The model receives no text, names or language-model
embeddings.

## Size

| Split | Trajectories | Eras |
|---|---:|---|
| train | 122,880 | 0, 1, 2, 3, 4, 5, 6, 7, 8, 9 |
| validation | 12,288 | 10 |
| test | 12,288 | 11 |
| transfer | 16,384 | 12 |

Total: **163,840 trajectories**.

Train, validation and test use twelve domain x mechanism pairs. Transfer uses four
disjoint pairs while retaining domains and mechanisms seen in other combinations.

## Files

- `manifest.json`: hashes, split policy, model tensor contract and claim boundary.
- `generator_contract.json`: exact numerical dynamics and intervention semantics.
- `*.npz`: compact `int64` indices; tensors are materialized deterministically.
- `quality_report.json`: automated integrity, leakage and distribution gates.

## Boundary

C0 qualifies mechanistic dynamics learning and combinatorial transfer only. It is
not evidence of real-world causality or production idea quality.

## Commands

```console
chimera validate-meta-world-corpus
```
