# AI Dataset Validation

Chimera uses AI subagents as the dataset release authority during early research.
Human review is optional and does not block experiments.

## C1 decision

`CHM-VENTURE-C1` is already validated by the complete multi-lens ledger
`CHM-V-C1-AI-MULTI-LENS-004`: 1,191/1,191 items verified and 10/10 cases accepted.
It is not reviewed again during the policy migration.

## Default for later datasets

Every new dataset requires three independent full-coverage subagent passes over the
same immutable source, manifest and reviewer-packet hashes:

1. `source_diligence` emphasizes source identity, locators and evidence.
2. `semantic_integrity` emphasizes nodes, ratings and relation contracts.
3. `commercial_challenge` emphasizes objectives, constraints, payers and revenue logic.

Each reviewer still covers the complete packet. The different roles reduce correlated
blind spots; they do not partition responsibility.

The gate passes only when all required roles return `accept`, every scoped item is
covered, no unresolved decision remains and every artifact matches the same snapshot.
Candidate and model outputs remain hidden from dataset reviewers.

## Failure behavior

- Missing review: `blocked`.
- `needs_change` or `cannot_verify`: `failed`.
- Stale or mismatched hash: validation error.
- Complete unanimous acceptance: `passed`; downstream generation may start.

The machine-readable default policy is `datasets/ai_review_policy.yaml`. A dataset
may register an explicit exception, as C1 does, but the exception must be recorded
before the next experiment run.
