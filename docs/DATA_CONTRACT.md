# Data Contract

## Core Input

| Tensor | Shape | Type | Meaning |
| --- | --- | --- | --- |
| `node_types` | `[B, N]` | `int64` | Typed entity IDs; zero is padding |
| `node_features` | `[B, N, 8]` | float32 | Fixed shared axes in `[0, 1]` |
| `edge_types` | `[B, N, N]` | `int64` | Directed relation IDs; zero is absent |
| `node_mask` | `[B, N]` | bool | Active node indicator |

Free-form strings and language-model embeddings are prohibited in model-core
inputs. Human-readable labels belong in optional sidecar metadata and must not
be loaded by the training pipeline.

The feature order is `salience`, `evidence`, `control`, `immediacy`,
`recurrence`, `scalability`, `value_proximity`, `risk`. Exact anchors and
derivations are frozen in the [business graph semantics](BUSINESS_GRAPH_SEMANTICS.md).

## Edit Target

Every edit step contains an operation, source node, target node, node type and
edge type. `step_mask` identifies registered steps. `STOP` terminates execution.

## Dataset Splits

Real experiments must use chronological or source-isolated splits. Cases,
organizations and derived variants may not cross boundaries. Any schema,
normalizer or candidate archive must be fitted without validation or test data.

## Provenance

Each dataset release must record source, acquisition timestamp, immutable
accession or content hash, transformation version, split hashes and exclusions.
Private or personal business data must not be committed to the public repository.
