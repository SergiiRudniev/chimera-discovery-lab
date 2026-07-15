# Chimera Venture Architecture

## Boundary

The model consumes categorical IDs, graph topology and numeric attributes. It
does not consume names, descriptions, embeddings produced by language models or
tokenized text. A separate interpreter may describe a frozen candidate, but its
output cannot modify the candidate or its research score.

This boundary tests a narrower claim than “the model does not think in
language”: whether structured latent search produces more useful novelty than a
matched text-generation system.

## Representation

Each business state is a bounded directed multirelation graph represented as a
dense relation matrix. M0 uses 12 node types, 16 relation types and eight
normalized numeric features per node. The numeric fields are domain contracts,
not text embeddings. Their fixed order, anchors and derived topology feature are
defined in [Business Graph Semantics](BUSINESS_GRAPH_SEMANTICS.md).

The encoder adds a learned global state token and applies five pre-normalized
graph transformer blocks. Relation IDs contribute a separate learned bias to
each attention head. Padded nodes cannot be attended to and are zeroed after
every block.

## Edit Program

The decoder predicts a bounded sequence with five categorical fields per step:

```text
(operation, source node, target node, node type, edge type)
```

Supported operations are `STOP`, `ADD_NODE`, `CONNECT`, `REWIRE`,
`TRANSFER_ROLE`, `REMOVE_CONSTRAINT`, `INVERT_RELATION`, `SUBSTITUTE` and
`MERGE`. Source and target fields use pointer logits over encoded nodes.

The deterministic executor applies a program to the source graph. Invalid or
out-of-capacity operations are no-ops and are counted during evaluation.

## Latent World Model

The edit decoder state is pooled and combined with the source graph state. Three
residual transition blocks predict the representation of the post-edit graph.
During training, the target is produced by an exponential-moving-average copy
of the graph encoder. Gradients never enter the target encoder.

The transition objective encourages an edit to represent a coherent state
change instead of merely matching edit labels. CHM-V-H003 is reserved for the
required ablation against zero transition weight.

## Candidate Scores

The model predicts normalized utility, feasibility and coherence from its
predicted next-state representation, so scores are conditioned on the proposed
edit program. These heads are learned estimates, not ground truth. Deterministic
schema validity is kept outside the network.

Novelty is calculated against retained candidate embeddings. MAP-Elites keeps
the highest-quality candidate in each descriptor cell rather than collapsing
the search into one global optimum.

## Proposal Policy

Reconstruction and exploration use separate inference policies. The frozen T1
weights provide graph and edit logits. T2 mixes each masked model distribution
with a uniform distribution over currently legal symbols:

```text
p_proposal = (1 - r) * p_model + r * uniform(legal symbols)
```

The validity masks apply before sampling to operations and their arguments. T2
selected `r = 0.50`; the model weights and reconstruction path are unchanged.

## Parameter Accounting

`chimera inspect` instantiates the registered configuration and counts trainable
parameters directly. The EMA target encoder is training-only state and is not
included in the inference model count. Optimizer state, archive entries and an
external language interpreter are also excluded.

The registered Venture M0 inference model contains **20,647,992 trainable
parameters**.

## Deferred Work

- Learn the numeric business-state schema from real event data.
- Define time-isolated business-case splits.
- Calibrate learned score heads against blinded human and observed outcomes.
- Test graph and latent representations against matched text baselines.
- Audit interpreter consistency with multiple independent decoders.
