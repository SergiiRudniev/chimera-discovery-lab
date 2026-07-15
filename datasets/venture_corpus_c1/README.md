# Venture Corpus C1

Preregistered, source-isolated evaluation corpus for `CHM-V-H001`.

| Property | Value |
|---|---:|
| Cases | 10 |
| Calibration | 2 |
| Evaluation | 8 |
| Numeric features | 8 |
| Maximum nodes | 64 |
| C0 organization / CIK / accession overlap | 0 / 0 / 0 |
| C0 latest period | 2024-12-31 |
| C1 earliest period | 2025-01-26 |

`graphs.npz` is the only Chimera model input. It contains typed graph tensors and
numeric objective/constraint masks; it contains no strings or object arrays.
`cases.jsonl` is the audit sidecar. `matched_briefs.jsonl` is the deterministic
language rendering of the same registered structure for the text baseline.

The corpus is validated for H001 input construction. It contains no creativity
scores or experiment results.

An internal source pass verifies filing identity and primary-source support for
10/10 cases. The review packet exposes every evidence note, graph element and
human-assigned numeric rating; `value_proximity` remains derived from topology.

The complete [multi-lens AI review](ai_reviews/multi_lens_ai_review.json) accepts
1,191/1,191 item decisions after correction. The registered
[AI review protocol](ai_review_protocol.yaml) accepts this existing ledger without
a repeat pass and enables generation. Human review remains optional.

```powershell
chimera build-evaluation-corpus
chimera validate-evaluation-corpus
chimera build-review-packet
chimera validate-review-gate
```

Sources are public SEC Form 10-K filings registered in `source_cases.yaml`.
