# Research Journal

## 2026-07-15 — Venture Corpus C0

- **Sources:** Ten public SEC Form 10-K filings spanning subscription, SaaS,
  marketplace, retail, payments, cloud, franchising and industrial mechanisms.
- **Semantics:** Eight shared numeric axes; seven use a frozen five-level rubric
  and `value_proximity` is derived from graph topology.
- **Dataset:** 10 canonical graphs and 640 deterministic denoising transitions;
  company-isolated splits contain 384 train, 128 validation and 128 test records.
- **Validation:** Every edit program reconstructs its registered target; all
  core files and shards are SHA-256 locked in the manifest; there are no exact
  numeric duplicates or company overlaps between splits.
- **Engineering smoke:** Full Venture M0 loss decreased from 7.3673 to 1.1501
  over five steps on one fixed two-record training batch; all metrics were finite.
- **Claim boundary:** This is post-build engineering evidence, not a
  preregistered creativity, generalization or commercial-utility result.

## 2026-07-15 — Venture M0 foundation

### CHM-V-H000

- **Question:** Does the full graph-to-edit training path run with finite
  gradients and reduce loss on a fixed synthetic transition batch?
- **Registration:** Complete before execution.
- **Result:** Loss decreased from 7.1843 to 1.0263 in 20 fixed-batch steps;
  every recorded loss and gradient norm was finite.
- **Decision:** Accepted as engineering validation.
- **Claim boundary:** A passing result validates wiring only, not creativity.

### CHM-V-H001

- **Question:** Does non-linguistic graph generation improve blind-rated novelty
  at matched feasibility versus a text baseline?
- **Registration:** Complete.
- **Result:** `not_run`; Corpus C0 is structural pretraining data, and no
  evidence-bearing evaluation corpus or trained checkpoint exists.
- **Decision:** Pending.

### CHM-V-H002

- **Question:** Does MAP-Elites improve structural diversity without reducing
  median feasibility?
- **Registration:** Complete.
- **Result:** `not_run`.
- **Decision:** Deferred until a frozen CHM-V-H001 policy exists.

### CHM-V-H003

- **Question:** Does latent next-state prediction improve held-out coherence?
- **Registration:** Complete.
- **Result:** `not_run`.
- **Decision:** Deferred until real transition data is split and frozen.
