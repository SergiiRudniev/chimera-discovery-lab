# Support-Preserving Candidate Pool H017

`CHM-W-H017` keeps the H016 critic fixed and changes only candidate generation.

## Failure addressed

H016 learned better fixed-pool ordering, but iterative CEM optimized rank logits
outside their reliable action distribution. It collapsed toward magnitude zero:
32.42% of selected actions landed exactly on that boundary, and median best
realized effect became zero.

## Candidate generator

For each state, H017 creates 256 legal numeric interventions. Ordered
source/target pairs are balanced so pair counts differ by at most one.
Magnitude and control use independently shuffled seeded Latin-hypercube strata:

```text
magnitude = (permuted_stratum + uniform_jitter) / 256
control   = -1 + 2 * (permuted_stratum + uniform_jitter) / 256
```

Every continuous value lies strictly inside its legal interval. Candidate
vectors are unique and deterministic for a fixed seed. No effect label or world
metadata guides generation.

## One-pass reranking

The unchanged H016 ranking critic scores all 256 candidates once. A
source/target/magnitude-quartile archive retains the maximum-logit candidate in
each occupied cell, then executes the eight highest-scoring cells. There is no
iterative resampling and therefore no optimizer-created action distribution.

## Controlled comparison

H017 retrains the exact H016 backbone and ranking head under their original seed
`260954`. The same weights score both learned arms:

- support-preserving one-pass pool reranking;
- H016 adaptive CEM plus quality diversity.

Each learned arm receives 256 scores and eight simulator executions. Legal
random executes the first eight candidates from the same seeded support pool.
An independent 256-action pool is evaluator-only and supplies regret.

Pool reranking must reduce regret to at most `0.75` of legal random and `0.85`
of adaptive CEM. Exact pool replay, pair balance, zero continuous boundaries,
unique vectors, legality, budgets, train/search/dataset replay, finite metrics
and zero leakage are hard guards. Frozen validation seeds and model tests remain
sealed until the development gate passes.

## Claim boundary

A passing result demonstrates numerical candidate generation only inside
registered generated worlds. It does not establish real-world causality,
business value, creativity, language-independent thought or production
readiness.
