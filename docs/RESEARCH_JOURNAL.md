# Research Journal

## 2026-07-15 — Venture Trial T1 registration

- **Protocol:** `CHM-V-T001`; corrective engineering qualification after T0.
- **Target audit:** Target-graph reconstruction is fully identifiable in C0;
  registered edit-program reconstruction has a 99.74% train majority upper
  bound because one duplicated input has alternative valid programs.
- **Loss audit:** 41.27% of raw train argument slots are irrelevant placeholders
  for their selected operation.
- **Correction:** Operation-conditioned argument loss, cosine learning-rate
  schedule, 3,000 steps and exact-graph checkpoint selection.
- **Preflight:** A 500-step probe on the first 16 training records reached 100%
  exact graph and program reconstruction. Validation and test were not opened.
- **Status:** Registered; target metrics not opened.

## 2026-07-15 — Venture Trial T0

- **Protocol:** `CHM-V-T000`, frozen at commit `72fcec8` before metrics were opened.
- **Runtime:** Full 20,647,992-parameter Venture M0; 300 steps on an NVIDIA
  GeForce RTX 5070 with PyTorch 2.13.0 and CUDA 13.2.
- **Selection:** Step 175 minimized validation loss at 3.5133; the test split was
  opened only after checkpoint selection.
- **Optimization:** Batch loss decreased from 7.9839 to 3.3122 and remained finite.
- **Reconstruction:** Train exact-graph reconstruction was 0.0% against the
  registered 95% threshold; the trial status is `completed_with_gaps`.
- **Generation:** 160/160 programs were valid and changed their source graph;
  148 result graphs were unique and fixed-seed replay was exact.
- **Coverage:** Generated non-terminal operations were limited to `CONNECT`,
  `INVERT_RELATION` and `SUBSTITUTE`, matching the operations supervised by C0.
- **Decision:** Preserve the checkpoint for audit, but do not promote it as an
  accepted model or evidence for CHM-V-H001.
- **Next action:** Register a separate corrective trial; do not alter T0 thresholds.

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
- **Result:** `not_run`; T0 produced an unqualified structural-pretraining
  checkpoint, but no evidence-bearing evaluation corpus or matched baseline exists.
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
