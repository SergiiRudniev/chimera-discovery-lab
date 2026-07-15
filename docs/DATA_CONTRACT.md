# Data Contract

## Core Input

| Tensor | Shape | Type | Meaning |
| --- | --- | --- | --- |
| `node_types` | `[B, N]` | `int64` | Typed entity IDs; zero is padding |
| `node_features` | `[B, N, 8]` | float | Normalized numeric attributes |
| `edge_types` | `[B, N, N]` | `int64` | Directed relation IDs; zero is absent |
| `node_mask` | `[B, N]` | bool | Active node indicator |

Free-form strings and language-model embeddings are prohibited in model-core
inputs. Human-readable labels belong in optional sidecar metadata and must not
be loaded by the training pipeline.

## Edit Target

Every edit step contains an operation, source node, target node, node type and
edge type. `step_mask` identifies registered steps. `STOP` terminates execution.

## Dataset Splits

Real experiments must use chronological or source-isolated splits. Cases,
organizations and derived variants may not cross boundaries. Any schema,
normalizer or candidate archive must be fitted without validation or test data.

## Provenance

Each dataset release must record source, acquisition timestamp, license,
content hash, transformation version, split hashes and exclusions. Private or
personal business data must not be committed to the public repository.
