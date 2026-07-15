# Business Graph Semantics

## Scope

Corpus C0 represents one disclosed business mechanism at one reporting date.
The graph records structure, not prose. Company names, node labels and evidence
notes are provenance sidecars and are never loaded into the model batch.

"Non-linguistic" here means that no text tokens or language embeddings enter
the network. The ontology and annotations remain human-designed; Corpus C0 does
not claim independence from human concepts.

## Node Types

| ID | Type | Exact role |
| ---: | --- | --- |
| 0 | `PAD` | Inactive capacity; all features and relations are zero |
| 1 | `ACTOR` | Participant that decides, supplies, consumes or pays |
| 2 | `NEED` | Unresolved job, demand or required state |
| 3 | `RESOURCE` | Controlled or accessible capability, asset, data or supply |
| 4 | `ACTION` | Repeatable transformation performed by the business mechanism |
| 5 | `CONSTRAINT` | Condition that can block or materially reduce an outcome |
| 6 | `CHANNEL` | Route through which an actor is reached or served |
| 7 | `VALUE` | Direct benefit delivered to an actor |
| 8 | `REVENUE` | Mechanism that transfers economic value to the focal business |
| 9 | `COST` | Material resource consumption or economic burden |
| 10 | `OUTCOME` | Observable result produced by delivered value |
| 11 | `FEEDBACK` | Signal that changes a resource, action or channel in a later cycle |

## Relation Types

Every relation is directed from `source` to `target`.

| ID | Relation | Meaning |
| ---: | --- | --- |
| 0 | `NONE` | No relation |
| 1 | `HAS_NEED` | Actor experiences the target need |
| 2 | `USES` | Source consumes or invokes the target |
| 3 | `ENABLES` | Source makes the target action or resource possible |
| 4 | `BLOCKS` | Source constraint prevents or reduces the target |
| 5 | `REACHES` | Source channel reaches the target actor |
| 6 | `DELIVERS` | Source action delivers the target value |
| 7 | `PAYS` | Source actor transfers value through the target revenue mechanism |
| 8 | `COSTS` | Source action creates the target cost |
| 9 | `PRODUCES` | Source value or action creates the target outcome |
| 10 | `FEEDS_BACK` | Source outcome creates the target feedback signal |
| 11 | `DEPENDS_ON` | Source cannot operate without the target |
| 12 | `SUBSTITUTE_FOR` | Source can replace the target |
| 13 | `TRANSFERS_TO` | Source provides a resource or role to the target |
| 14 | `AMPLIFIES` | Source increases the strength or capacity of the target |
| 15 | `REDUCES` | Source decreases the target state or outcome |

## Numeric Features

The feature order is immutable for Corpus C0:

```text
[salience, evidence, control, immediacy,
 recurrence, scalability, value_proximity, risk]
```

Human-assigned features use exactly `0`, `0.25`, `0.5`, `0.75` or `1`. Values
between anchors are prohibited. `value_proximity` is derived from topology.

| Feature | 0 | 0.25 | 0.5 | 0.75 | 1 |
| --- | --- | --- | --- | --- | --- |
| `salience` | Incidental | Supporting | Material | Core | Mechanism fails without it |
| `evidence` | Unsupported assumption | Indirect primary-source inference | Explicit qualitative disclosure | Repeated or quantified primary disclosure | Audited, contractual or directly measured fact |
| `control` | External | Influence only | Shared or contractual control | Operational control | Owned and directly governed |
| `immediacy` | More than 3 years | 1–3 years | 91–365 days | 2–90 days | Same day or continuous |
| `recurrence` | One-off or less than annual | Annual | Monthly or quarterly | Weekly | Daily, per transaction or continuous |
| `scalability` | Hard cap or superlinear input | Physical linear scaling | Partial operating leverage | Strong leverage with capacity limits | Near-zero marginal replication or network scaling |
| `value_proximity` | No path or 4+ hops | 3 hops | 2 hops | 1 hop | Node is `VALUE`, `REVENUE` or `OUTCOME` |
| `risk` | Negligible | Local and reversible | Material but recoverable | Severe cross-system exposure | Existential, safety-critical or regulatory-critical exposure |

Higher is favorable for all axes except `risk`. `evidence` describes confidence
in the graph assertion, not business quality. After every structural edit,
`value_proximity` is recomputed by reverse breadth-first traversal over directed
relations.

## Proxy Scores

Corpus C0 does not contain outcome, human-preference or creativity labels. It
contains transparent structural proxies solely to exercise the three score
heads.

```text
utility_proxy = mean over VALUE / REVENUE / OUTCOME nodes of
  0.30*salience + 0.20*evidence + 0.15*immediacy
  + 0.15*recurrence + 0.10*scalability + 0.10*value_proximity

feasibility_proxy = mean over RESOURCE / ACTION / CHANNEL nodes of
  0.30*control + 0.25*evidence + 0.15*immediacy
  + 0.15*scalability + 0.15*(1-risk)
  - 0.20*mean(constraint.salience * constraint.risk)

coherence_proxy =
  0.35*required_type_coverage + 0.35*value_reachability
  + 0.20*largest_weak_component_ratio + 0.10*feedback_presence
```

All scores are clipped to `[0, 1]`. They must be replaced or calibrated before
any claim about commercial utility, feasibility or creativity.

`required_type_coverage` is the fraction of six present groups: `ACTOR`, `NEED`,
`RESOURCE or ACTION`, `VALUE`, `REVENUE`, `OUTCOME`. `value_reachability` is the
fraction of active nodes with a directed path of at most three hops to a value
target. `feedback_presence` is one only when an active `FEEDBACK` node has an
incoming or outgoing relation.

## Annotation Rules

- Use only facts available by the source reporting date.
- Create one node per dominant business role; do not encode prose fragments.
- Record every interpretive step in the evidence sidecar.
- Do not infer causal success from revenue size or company survival.
- Keep all variants of one organization in a single split.
- Never load sidecar labels, evidence notes or source text into model tensors.
