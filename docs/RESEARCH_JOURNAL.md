# Research Journal

## 2026-07-15 — C1 correction and AI re-review

- **Correction:** Removed unsupported causal and feedback edges, aligned payer
  roles with revenue mechanisms, corrected four SEC filing dates, separated
  NIKE direct and wholesale revenue, and recalibrated overstated numeric anchors.
- **Protection:** Review packet schema v2 requires an independent decision for
  every one of the seven human-assigned ratings on every node. The corpus builder
  rejects fixed relation-role violations and the discovered causal anti-patterns.
- **AI verification:** The four-lens subagent accepted 1,191/1,191 item decisions
  across 10 filing identities, 32 evidence notes, 126 nodes, 882 ratings, 100
  edges, 20 objectives and 11 constraints; all 10 cases were accepted.
- **Integrity:** The AI ledger is SHA-256 bound to the source, corpus manifest and
  reviewer packet. Local validation, deterministic rebuild and semantic checks pass.
- **Independence:** AI acceptance is support evidence only. `C1-Q001` remains open,
  the human gate remains blocked at 0/1 and H001 generation remains disabled.

## 2026-07-15 — C1 multi-lens AI review

- **Reviewer:** Codex subagent with source-diligence, commercial-logic,
  problem-alignment and semantic-overreach lenses.
- **Coverage:** 10/10 SEC filings and all registered graph elements were
  screened; the committed support artifact records material findings, not a
  complete item-level decision ledger.
- **Decision:** `needs_change`. Repeated issues include invalid
  `NEED→DEPENDS_ON→VALUE`, unsupported `COST→REDUCES→OUTCOME`, invalid actor to
  resource transfers, payer mismatches and overstated numeric anchors.
- **Verification:** The three systemic relation counts reproduce directly from
  `source_cases.yaml`: 10, 10 and 7 affected cases respectively.
- **Independence:** This is an AI-assisted internal audit, not an independent
  human review. `C1-Q001` remains open and H001 generation remains blocked.

## 2026-07-15 — Venture Corpus C1 review gate

- **Internal audit:** Filing identity and primary-source support verified for
  10/10 registered cases.
- **Independence:** The internal auditor is the annotation author, so this work
  does not close `C1-Q001`.
- **Gate:** Coverage and decisions are checked for every evidence note, node,
  edge, objective and constraint, together with manifest and packet hashes.
- **Protection:** Self-review, partial review, changed hashes, missing
  attestations and `needs_change` all block generation.
- **Status:** Blocked; 0/1 independent reviews accepted.
- **Claim boundary:** No candidate, model output, rating or H001 result was
  produced.

## 2026-07-15 — Venture Corpus C1 preregistration

- **Corpus:** `CHM-VENTURE-C1`; 2 calibration and 8 evaluation cases from
  source-registered FY2025 SEC filings.
- **Isolation:** Zero C0 overlap by organization, CIK and accession; the C1
  period boundary begins after the maximum C0 period end.
- **Model input:** Numeric typed graphs plus objective and constraint masks;
  no strings or object arrays in `graphs.npz`.
- **Baseline:** `Qwen/Qwen2.5-0.5B-Instruct` frozen at revision
  `7ae557604adf67be50417f59c2c2f167def9a775`; 8 candidates per case.
- **Rating plan:** 128 evaluation candidates, 3 blind raters and 384 planned
  ratings. Novelty uses a paired exact sign-flip test with a feasibility
  non-inferiority guardrail.
- **Quality:** File integrity, range, alignment, topology, time boundary and
  leakage checks passed. All ten graphs remain pending independent annotation
  review.
- **Decision:** Freeze C1 and the H001 protocol. Do not generate candidates
  until the independent review is complete.
- **Claim boundary:** No H001 candidate, human rating or creativity result has
  been produced.

## 2026-07-15 — Venture Trial T2 result

- **Protocol:** `CHM-V-T002`, frozen at commit `df49c79` before validation and
  test policy metrics were opened.
- **Runtime:** 36.5 seconds on the NVIDIA GeForce RTX 5070; T1 weights remained
  unchanged and matched the registered SHA-256 before and after execution.
- **Selection:** Validation selected `explore-50` at 88.02% mean unique graphs;
  all three exploratory policies satisfied the registered eligibility rules.
- **Test:** `explore-50` reached 94.01% mean unique graphs versus 27.08% for
  `model-only`; the per-seed minimum was 92.19%.
- **Guardrails:** 100% changed, 0% invalid, deterministic replay and median
  feasibility 0.5875 versus 0.5719 baseline.
- **Coverage:** All eight non-terminal operations were observed and mean
  MAP-Elites coverage reached 52.08% versus 31.25% baseline.
- **Reconstruction:** Exact-graph reconstruction remained 99.22% on train.
- **Decision:** Accept `explore-50` as the frozen T2 proposal policy; keep the T1
  checkpoint as the only model-weight artifact.
- **Claim boundary:** This is an engineering policy qualification, not evidence
  for semantic novelty, commercial utility or CHM-V-H001.

## 2026-07-15 — Venture Trial T2 registration

- **Protocol:** `CHM-V-T002`; frozen proposal-policy qualification using the
  unchanged T1 step-2700 checkpoint.
- **Diagnosis:** T1 unique-graph rate fell by 59.38 percentage points versus T0;
  `CONNECT>STOP` accounted for 45.63% of its candidate programs.
- **Train-only probe:** Legal-uniform exploration at rate 0.50 produced 86.55%
  mean unique graphs, 99.22% changed candidates, all eight non-terminal
  operations and zero invalid programs across three seeds.
- **Separation:** Reconstruction remains bound to the immutable T1 checkpoint;
  T2 selects only an inference-time proposal policy.
- **Data boundary:** Train was used for diagnosis; validation will select the
  policy and test is reserved for one final report.
- **Status:** Registered; validation and test policy metrics remain unopened.
- **Claim boundary:** Engineering proposal-policy evidence only.

## 2026-07-15 — Venture Trial T1 result

- **Protocol:** `CHM-V-T001`, frozen at commit `7b69794` before target metrics
  were opened.
- **Runtime:** Full 20,647,992-parameter Venture M0; 3,000 steps in 134.9 seconds
  on an NVIDIA GeForce RTX 5070.
- **Selection:** Step 2700 maximized validation exact-graph reconstruction at
  30.47%; validation loss was only a tie-breaker.
- **Optimization:** Batch loss decreased from 7.9580 to 0.1474 and remained finite.
- **Reconstruction:** Exact-graph rates were 99.22% train, 30.47% validation and
  14.84% test. The registered 95% train criterion passed.
- **Generation:** 160/160 programs were valid; 144 changed their source graph and
  53 result graphs were unique. Fixed-seed replay was exact.
- **Trade-off:** Unique-graph rate decreased from 92.50% in T0 to 33.13% in T1;
  final MAP-Elites coverage decreased from 43.75% to 18.75%.
- **Decision:** Accept T1 as an engineering structural-reconstruction checkpoint,
  not as creativity or CHM-V-H001 evidence.
- **Next action:** Correct generation collapse before building the matched
  language-baseline evaluation.

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
