# Research Journal

## 2026-07-16 - CHM-W-H013 implementation and WG4 integrity

- **Implementation:** WG4 now emits factual and no-op next observations from
  identical simulator state, event and renderer-noise state. Only the factual
  branch advances the trajectory.
- **Leakage guard:** The no-op tensor is a target field. A model-invariance test
  replaces it with arbitrary values and confirms unchanged forward outputs.
- **Matched architecture:** Factorized and direct dual-transition models each
  contain `65,484,814` trainable parameters. Both retain H008 factual/no-op
  utility semantics and receive identical factual, no-op and delta losses.
- **WG4 integrity:** All fixed-shard hashes, tensor shapes, finite checks,
  deterministic replay, paired-renderer consistency, mechanism/world/seed and
  exact-configuration isolation checks passed across 80 trajectories. No model
  test metric was opened and no human or LLM judgment was used.
- **GPU smoke:** RTX 5070 BF16 training completed cleanly at about 1.54 GiB peak
  allocated memory. The additive identity residual was `2.98e-8`, below `1e-6`.
- **Status:** Implementation-qualified only. The registered development suite
  has not run; `research/results/CHM-W-H013.json` remains `not_run`.
- **Claim boundary:** Engineering evidence only; no transfer, causal, business,
  language-independence or production claim.

## 2026-07-16 - CHM-W-H013 registration

- **Question:** Does exact additive factorization of state dynamics improve
  prediction of intervention-induced state change without harming factual
  rollout, no-op-state or intervention-effect prediction?
- **Data change:** WG4 adds a simulator-derived no-op next-state target paired
  with every factual transition under identical state, external event and
  renderer noise. The target never enters the model as a feature.
- **Controlled comparison:** Factorized and direct dual-transition models have
  identical parameter counts, encoder, outcome semantics, data, actions,
  optimizer, seed and training budget. Only state-transition arithmetic differs.
- **Primary gate:** Delta-state NRMSE must improve by at least 10% versus the
  matched direct control while factual rollout, no-op-state and effect NRMSE do
  not increase. The additive identity residual must be at most `1e-6`.
- **Isolation:** Development opens train and validation only. Frozen validation
  seeds and every test split stay sealed until the development gate passes.
- **Status:** `not_run`; no H013 metric or checkpoint exists.
- **Claim boundary:** Simulator representation evidence only; no real-world
  causal, business-profitability, language-independence or production claim.

## 2026-07-16 — CHM-W-H012 development preflight

- **Scope:** Development seed `260938`, four trainable arms, 1000 optimizer
  steps each, plus deterministic legal-random regret. Only `train` and
  `validation` were opened; frozen validation seeds and all test splits stayed
  sealed.
- **Aligned cross-world:** Effect NRMSE `0.749912`, rollout NRMSE `0.453476`,
  retrieval `0.484375`.
- **No-alignment cross-world:** Effect NRMSE `0.742323`, rollout NRMSE
  `0.451357`, retrieval `0.265625`.
- **Target-family-only:** Effect NRMSE `1.402974`, rollout NRMSE `0.471859`.
  The arm used actual held-family train sampling and selected step `400`.
- **Temporal baseline:** Effect NRMSE `0.948956`, rollout NRMSE `0.457700`.
- **Random baseline:** All actions were legal; mean sampled intervention regret
  was `0.025495` across four states and eight candidates per state.
- **Controlled result:** Alignment increased retrieval by `0.218750`, but its
  effect and rollout ratios versus the matched no-alignment arm were `1.010223`
  and `1.004695`. The preregistered `0.90`/`0.90` development gate failed.
- **Diagnostic:** Cross-world relational pretraining without alignment beat the
  temporal baseline on effect (`0.782253x`) and slightly on rollout
  (`0.986142x`). Generator diversity is useful in validation; the global
  mechanism-alignment objective remains the conflicting component.
- **Decision:** Do not open H012 frozen validation or test, do not promote any
  checkpoint and leave `research/results/CHM-W-H012.json` as `not_run`.
- **Next action:** Complete the already preregistered H011 direct paired-response
  consistency experiment, which supervises predictions rather than global
  embedding similarity.
- **Claim boundary:** Development simulator evidence only; no real-world
  transfer, causal discovery, business utility, language independence or
  production claim.

## 2026-07-16 — CHM-W-H012 registration

- **Question:** Does generator-first cross-world pretraining improve numerical
  intervention and rollout prediction on held world-family mappings?
- **Foundation:** Reuse the H002/H009 `MechanismGenerator`, `WorldGenerator`
  and `ObservationRenderer` contracts instead of creating a parallel simulator
  API. Training data remain online and evaluation data remain fixed by SHA-256.
- **Controlled comparison:** Aligned cross-world, unaligned cross-world,
  target-family-only and non-relational temporal models use matched data and
  optimization budgets. Legal random intervention is a regret-only baseline.
- **Primary gate:** On `test_world_transfer`, aligned cross-world effect and
  four-step rollout NRMSE must each be at most `0.90` of the strongest eligible
  predictive baseline, with paired 90% bootstrap ratio bounds below `1.00`.
- **Isolation:** All generator provenance remains evaluator-only. Test stays
  sealed until all arms, checkpoints and validation decisions are frozen.
- **Status:** `not_run`; no H012 metric, checkpoint or transfer claim exists.
- **Claim boundary:** Numerical simulator evidence only; no real-world causal,
  business-profitability, language-independence or production claim.

## 2026-07-15 — CHM-W-H005 registration

- **Question:** Can a 50:50 probe/random curriculum retain H004's
  intervention-effect gain without increasing four-step rollout error?
- **Diagnosis inherited from H004:** Probe training improved effect prediction
  materially. Probe-only distribution shift and instance discrimination damaged
  rollout, while removing discrimination almost restored the random baseline.
- **Change:** Pair the same generated mechanisms/worlds under probe and random
  action policies in each train batch. Disable mechanism-discrimination loss.
- **Controlled comparison:** Mixed, random-only and probe-only arms share model,
  optimizer, seed, number of trajectories, closed-loop objective and hybrid
  evaluator. Only train policy mixture differs.
- **Development gate:** On seed `260910`, require effect ratio at most `0.90`
  and rollout ratio at most `1.00` versus random-only before freezing any
  validation configuration.
- **Validation discipline:** Hyperparameters freeze before seeds
  `260911..260913`; test remains sealed until their aggregate gate passes.
- **Status:** `not_run`; no H005 model metric or checkpoint exists.
- **Claim boundary:** Simulator curriculum evidence only; no real-world
  causality, probe safety, business utility or production claim.

## 2026-07-15 — CHM-W-H004 development preflight

- **Scope:** Development seed `260906`, 300 steps, `train` and `validation`
  only. Frozen validation seeds and every test split remained unopened.
- **Matched probe arm:** Effect NRMSE `0.883479`, rollout NRMSE `0.501839`,
  coverage `0.991667`, retrieval `0.0`.
- **Matched random arm:** Effect NRMSE `1.000466`, rollout NRMSE `0.474423`,
  coverage `0.983333`, retrieval `0.0`.
- **Controlled policy effect:** Probe/random ratios were `0.883068` for
  intervention effect and `1.057790` for rollout. Probes improved effect by
  `11.69%` but worsened rollout by `5.78%`.
- **No-discrimination diagnostic:** Probe closed-loop training without the
  mechanism loss reached effect `0.857156` and rollout `0.477443`. This suggests
  that probes are useful while instance discrimination conflicts with dynamics.
- **Decision:** Do not run seeds `260907..260909`, do not freeze `CHM-W-T004`,
  do not open test and do not promote checkpoints. The full arm failed rollout
  and retrieval gates despite its effect improvement.
- **Next action:** Register a mixed probe/random curriculum without instance
  discrimination, preserving active-identification effect gains while enforcing
  rollout non-inferiority against a matched random-only arm.
- **Claim boundary:** Development validation engineering evidence only. H004
  remains `not_run`; no test-world transfer or production claim exists.

## 2026-07-15 — CHM-W-H004 WG1 implementation

- **Dataset:** Implemented `CHM-W-WG1` with 16-step trajectories, deterministic
  system-identification train probes and a shared four-probe-plus-random
  evaluator policy.
- **Boundary:** Policy IDs remain manifest metadata. Model batches contain only
  numeric observations, masks, relations, actions, time and outcomes.
- **Fixed smoke:** 16 trajectories per split, 80 total; manifest SHA-256
  `cc0305bd99f05cf5d528f045dd494652d377be2c5836d5922d589eb6d3b96461`.
- **Integrity:** 20/20 gates passed, including exact replay, source/shard hashes,
  tensor shape and finite checks, all split-isolation policies and registered
  probe-prefix coverage.
- **Excitation diagnostic:** Between/within paired response separation was
  `1.235036`. This confirms non-zero numeric response diversity only.
- **Status:** Data pipeline ready for a validation-only model preflight. H004
  remains `not_run`; no model metric or checkpoint was produced.

## 2026-07-15 — CHM-W-H004 registration

- **Question:** Do controlled numeric system-identification probes improve
  mechanism retrieval and unseen-world prediction over random-action training?
- **Diagnosis inherited from H003:** Hard-negative pressure made embeddings
  more similar across views but could not rank a true pair above the closest
  distinct mechanism. The short passive trajectory was not sufficiently
  identifying.
- **Change:** Expand trajectories to 16 steps. Train on deterministic zero,
  impulse, magnitude, control-polarity, reversal and recovery probes. Evaluate
  every arm on the same four-probe prefix followed by seeded random actions.
- **Controlled comparison:** The full and random-curriculum arms share model,
  optimizer, loss, mechanism queue, splits and evaluator; only the train action
  policy differs.
- **Isolation:** Probe type and all generator provenance remain evaluator-only.
  Model inputs contain only observations, masks, relations, time and actions.
- **Validation discipline:** Seed `260906` is development-only. Hyperparameters
  must freeze before seeds `260907..260909`; test remains sealed until the
  registered validation gate passes.
- **Status:** `not_run`; WG1, model metrics and checkpoints do not exist yet.
- **Claim boundary:** Simulator-only active identification; no real-world
  causality, experiment safety, business utility or production claim.

## 2026-07-15 — CHM-W-H003 exploratory validation preflight

- **Scope:** Single-seed (`260903`), 300-step exploratory comparison on
  `train` and `validation`; all test splits remained sealed.
- **Full arm:** With alignment weight `1.0`, intervention-effect NRMSE reached
  `0.829864`, four-step rollout NRMSE `0.432069`, effect coverage `0.988839`
  and mechanism retrieval `0.0`.
- **Matched one-step baseline:** Effect NRMSE `0.857798` and rollout NRMSE
  `0.445780`. Full-arm ratios were `0.967435` and `0.969244` respectively.
- **Ablations:** Closed-loop without discrimination reached effect `0.836194`
  and rollout `0.441474`. Alignment weight `0.2` reached effect `0.852021`
  and rollout `0.432761`.
- **Embedding diagnosis:** Same-mechanism validation cosine reached `0.984971`,
  but the mean hardest distinct-mechanism cosine was still `0.994591`; nearest
  neighbour retrieval therefore remained zero.
- **Decision:** Do not run the remaining registered seeds, do not freeze
  `CHM-W-T003`, do not open test and do not promote a checkpoint. The full arm
  missed both `0.95` primary-error ratios and the `0.10` retrieval gate.
- **Next action:** Change the data-identification signal. Add controlled numeric
  system-identification probes instead of applying more weight or search to the
  same instance-discrimination loss.
- **Claim boundary:** Exploratory validation engineering evidence only. H003
  remains `not_run`; no test-world transfer or production claim exists.

## 2026-07-15 — CHM-W-H003 registration

- **Question:** Does four-step closed-loop training plus a cross-batch
  hard-negative queue improve unseen-world intervention-effect and rollout
  prediction over every matched baseline?
- **Diagnosis inherited from H002:** Relational state improved intervention
  prediction over the temporal baseline, but in-batch mechanism alignment hurt
  both primary validation metrics relative to the same unaligned architecture.
- **Change:** Train autoregressively for four steps and contrast mechanisms
  against `256..2048` detached embeddings retained across batches.
- **Isolation:** Hidden IDs may pair examples for the training loss, but never
  enter the model forward contract. H002 split isolation and generator SHA-256
  remain frozen.
- **Validation gate:** Across seeds `260903..260905`, both primary median errors
  must be no more than `0.95` times the strongest learned baseline, mechanism
  retrieval must reach `0.10`, and effect coverage must remain at least `0.85`.
- **Status:** `not_run`; test metrics remain sealed and no checkpoint exists.
- **Claim boundary:** Simulator-distribution transfer only; no real-world
  causality, business utility, language-independent thought or production claim.

## 2026-07-15 — CHM-W-H002 validation-only model preflight

- **Scope:** Matched single-seed engineering preflight on `train` and
  `validation` only. Every H002 test split remained sealed.
- **Runtime:** NVIDIA GeForce RTX 5070, CUDA BF16, EMA weights, 1,000 optimizer
  steps per arm, seed `260902`, Git commit `7cf8931`.
- **Metric correction:** State and rollout targets now contain only the four
  dynamic signal channels. Independently resampled renderer nuisance channels
  remain visible inputs but are not treated as predictable future state.
- **Aligned relational:** Intervention-effect NRMSE `0.693132`, four-step
  rollout NRMSE `0.416371`, mechanism retrieval `0.03125`.
- **Unaligned relational:** Intervention-effect NRMSE `0.663543`, four-step
  rollout NRMSE `0.413200`, mechanism retrieval `0.0`.
- **Temporal baseline:** Intervention-effect NRMSE `0.955519`, four-step
  rollout NRMSE `0.417827`, mechanism retrieval `0.0`.
- **Comparison:** The aligned relational arm reduced effect error by `27.46%`
  versus the temporal baseline, but was `4.46%` worse than the same relational
  architecture without alignment. Its rollout error was `0.77%` worse than the
  unaligned relational arm.
- **Decision:** Do not freeze `CHM-W-T002`; do not open test metrics and do not
  promote any preflight checkpoint. The H002 result remains `not_run` because
  the aligned arm did not plausibly satisfy the registered requirement to beat
  the strongest baseline on both primary metrics.
- **Next action:** Register a new immutable hypothesis for multi-step
  closed-loop training and stronger cross-batch mechanism discrimination;
  preserve H002 as an append-only negative validation diagnosis.
- **Claim boundary:** Validation-only simulator engineering evidence. It does
  not establish test-world transfer, causal discovery, business utility or
  production readiness.

## 2026-07-15 — CHM-W-H002 generator implementation

- **Architecture:** Implemented independent `MechanismGenerator`,
  `WorldGenerator` and `ObservationRenderer` layers with `FlowWorld`,
  `CompetitionWorld` and `FunnelWorld`.
- **Modes:** CPU online generation and fixed object-free NPZ evaluation shards
  use the same seed-addressable trajectory pipeline.
- **Smoke artifact:** Five splits with 12 trajectories each; manifest SHA-256 is
  `eda799f1f499078491269724e0ac58839e0b14f23676dafe80d522a52c75d657`.
- **Integrity:** 15/15 gates passed, including exact replay, source and shard
  hashes, tensor shapes, finite values, concrete mechanism/world/seed/config
  isolation and held mechanism/family/renderer policies.
- **Software validation:** Ruff and strict mypy passed; 62 tests passed with
  82.40% branch coverage.
- **Decision:** Engineering pipeline ready for an evidence-bearing trial.
  `CHM-W-H002` remains `not_run`; no target metrics or checkpoint were created.

## 2026-07-15 — CHM-W-H002 registration

- **Question:** Does cross-world pretraining with mechanism alignment improve
  intervention-effect prediction and four-step rollout prediction in held-out
  world-family mappings?
- **Worlds:** Programmatic numerical `FlowWorld`, `CompetitionWorld` and
  `FunnelWorld`; no language is accepted by the model input contract.
- **Representation:** Hidden mechanisms are rendered through independently
  sampled object, feature, unit, time, noise and visibility transforms.
- **Isolation:** Concrete mechanisms, world instances, seeds, world configs and
  renderer configs cannot cross dataset splits. Mechanism templates may repeat
  where the registered transfer question requires a known law in a new world
  family or rendering.
- **Primary gate:** On `test_world_transfer`, both intervention-effect RMSE and
  four-step rollout NRMSE must be at most 0.90 times the strongest baseline,
  and each paired 90% bootstrap ratio upper bound must be below 1.00.
- **Comparators:** No-alignment, target-family-only, temporal-without-relations
  and legal-random-intervention baselines.
- **Status:** `not_run`; no target metrics or checkpoints have been opened.
- **Claim boundary:** Simulator-distribution mechanism transfer only; no claim
  of real-world causal discovery, profitable ideas or production readiness.

## 2026-07-15 — Meta-World failure artifact policy

- **Finding:** T000 failed before the trial runner created its normal result
  files, so the rejection had to be recorded manually.
- **Change:** The Meta-World runner now writes an `execution_failed` result with
  exception type, environment, Git commit and config hash before re-raising.
- **Protection:** A regression test injects an optimizer-step failure and verifies
  both the trial-local and public research result artifacts.
- **Boundary:** The policy records failures; it does not convert them into valid
  model evidence or permit reuse of a failed trial ID.

## 2026-07-15 — Meta-World H001 result

- **Protocol:** `CHM-W-T001`, frozen at commit `a437d3e`; architecture,
  fixed batch, seed, optimizer and gates were unchanged from rejected T000.
- **Correction:** Replaced FP32 indexed assignment with dtype-preserving
  selection across domain-adapter outputs and added a BF16 autocast regression test.
- **Runtime:** 61,854,120 parameters; 20 BF16 CUDA steps in 1.37 seconds on the
  RTX 5070; peak allocated VRAM was 1,902,361,600 bytes.
- **Optimization:** Fixed-batch loss moved from 0.110891 to -1.892130; all
  recorded metrics and gradients were finite.
- **Determinism:** Maximum evaluation replay delta was exactly 0 before and
  after training.
- **Decision:** Accept H001 as W0 core engineering qualification. Do not publish
  a checkpoint from this overfit smoke trial.
- **Next action:** Build time-isolated mechanistic trajectories and evaluate
  held-out next-state accuracy, uncertainty calibration and cross-domain transfer.
- **Claim boundary:** No real-world causality, semantic grounding, idea quality
  or production-readiness evidence exists.

## 2026-07-15 — Meta-World H000 result

- **Protocol:** `CHM-W-T000`, frozen at commit `42f016c` before CUDA execution.
- **Architecture:** Full 61,854,120-parameter Meta-World W0; 16 fixed systems,
  four mechanism families, four domain transforms and eight interventions.
- **Failure:** The first BF16 forward pass stopped in domain-adapter selection:
  indexed assignment attempted to write a BF16 adapter result into an FP32 buffer.
- **Boundary:** No complete forward pass, optimizer step, target loss metric or
  checkpoint was produced.
- **Decision:** Reject H000. Correct dtype propagation under a new immutable
  hypothesis and trial ID; do not rerun T000.

## 2026-07-15 — Chimera Meta-World W0 registration

- **Decision:** Name the new causal world-model family `Chimera Meta-World` and
  its first generation `Chimera Meta-World W0`.
- **Namespace:** Reserve `chimera-meta-world`, `CHM-W-H###`, `CHM-W-T###` and
  `CHM-W-C###` for the family, experiments and corpora.
- **Scope:** W0 will learn numerical cross-domain state dynamics and propose a
  frozen intervention before language grounding.
- **Hardware envelope:** Target approximately 64M trainable parameters for local
  mixed-precision training on the registered RTX 5070 with 12,227 MiB VRAM.
- **Status:** Design registered; no W0 implementation, dataset, checkpoint or
  empirical result exists yet.
- **Claim boundary:** Registration does not establish language independence,
  causal discovery, cross-domain transfer or idea quality.

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
## 2026-07-15 — Meta-World H005 development gate

- **Hypothesis:** `CHM-W-H005`; a paired 50:50 probe/random curriculum without
  mechanism-discrimination loss versus a matched paired random-only curriculum.
- **Fairness correction:** Both primary arms saw the same mechanisms and number
  of unique worlds; each world contributed two action-policy trajectories.
- **Runtime:** Three 300-step, 65,213,950-parameter BF16 runs on the NVIDIA
  GeForce RTX 5070; peak allocated memory was 2,165,645,312 bytes.
- **Primary development comparison:** Mixed/random intervention-effect NRMSE
  ratio was 0.89223 and four-step rollout NRMSE ratio was 0.94626.
- **Diagnostic:** Probe-only improved rollout but worsened intervention-effect
  error, supporting a mixture rather than deterministic probes alone.
- **Decision:** Development gate passed. Freeze architecture, optimizer,
  checkpoint step 300 and seeds 260911–260913 before opening validation.
- **Boundary:** Development evidence only. Frozen validation and every test split
  remained unopened; no checkpoint was promoted.
## 2026-07-15 — Meta-World H005 frozen validation

- **Protocol:** `CHM-W-H005`, frozen at commit `5b6fa44`; checkpoint step 300
  and seeds 260911–260913 were fixed before metrics were opened.
- **Result:** Median mixed/random intervention-effect NRMSE ratio was 1.03230
  versus the registered maximum 0.90. The primary gate failed.
- **Rollout:** Median four-step ratio was 0.98968 and the worst seed was 1.00936,
  both within their guardrails. Mixed training helped dynamics but did not
  improve intervention-effect transfer.
- **Calibration:** Minimum mixed-arm 90% interval coverage was 0.97813; all
  metrics were finite and WG1 replay/leakage checks remained clean.
- **Decision:** Do not open `CHM-W-T005` test and do not promote a checkpoint.
- **Next diagnosis:** Separate effect learning from passive rollout learning;
  the shared objective appears to convert probe coverage into state prediction
  without a seed-stable intervention-effect advantage.
- **Boundary:** Frozen-validation evidence only; no test, real-world or
  production claim.

## 2026-07-15 — CHM-W-H006 registration

- **Question:** Can probes improve latent dynamics while continuous random
  interventions alone supervise the intervention-effect likelihood?
- **Diagnosis:** In a 256-trajectory train-only audit, probe actions had three
  magnitudes and 39.4% exact-zero effects; random actions had 3,840 magnitudes
  and 16.2% exact-zero effects. H005's shared loss improved rollout but failed
  its effect-transfer validation gate.
- **Change:** Retain paired 50:50 trajectories. Apply state and variance losses
  to both halves, but apply effect NLL only to the random half. The binary route
  stays in the trainer and never enters model forward.
- **Controls:** Routed mixed, shared-loss mixed and matched random-only arms use
  the same model, worlds, trajectory count, optimizer and evaluator.
- **Development gate:** Seed `260914`; require effect ratio at most `0.90` and
  rollout ratio at most `1.00` versus random-only before freezing validation.
- **Validation discipline:** Seeds `260915..260917` and test remain sealed.
- **Status:** `not_run`; no H006 metric or checkpoint exists.
- **Boundary:** Simulator objective-routing evidence only; no real-world,
  business-utility or production claim.

## 2026-07-15 — CHM-W-H006 development preflight

- **Protocol:** Seed `260914`, 300 steps, paired WG1 worlds and the frozen
  hybrid evaluator. Frozen validation and test remained sealed.
- **Routed arm:** Effect NRMSE `1.07551`, rollout NRMSE `0.47489` and effect
  coverage `0.98229`.
- **Matched random:** Effect NRMSE `1.01771` and rollout NRMSE `0.48781`.
  Routed/random ratios were `1.05679` and `0.97350`; the primary effect gate
  failed even though rollout improved.
- **Shared-loss control:** Effect NRMSE `0.81285`, rollout `0.46926`; ratios
  versus random were `0.79871` and `0.96196`. Removing probe effect supervision
  caused a 32.31% effect-error increase versus shared mixed training.
- **Decision:** Reject H006 routing, do not open seeds `260915..260917`, do not
  open test and do not promote any checkpoint.
- **Diagnosis:** Probe outcomes contain useful effect supervision; the remaining
  issue is seed instability of shared multi-objective optimization, not simply
  a zero-heavy probe label distribution.
- **Boundary:** Development evidence only; no transfer or production claim.

## 2026-07-15 — CHM-W-H007 registration

- **Question:** Can conflict-projected state/effect gradients stabilize the
  shared mixed curriculum without discarding probe effect supervision?
- **Gradient audit:** Across 32 train-only batches, cosine was negative in
  62.5%, with median `-0.04242` and effect/state norm ratio `0.96544`.
- **Change:** When the two weighted task gradients conflict, symmetrically
  remove each component opposing the other. Add auxiliary gradients afterward,
  then apply the existing global norm clip.
- **Controls:** PCGrad mixed, standard mixed and matched random-only use the same
  model, data, actions, optimizer settings, seed and evaluator.
- **Development gate:** Seed `260918`; require effect ratio at most `0.90`,
  rollout ratio at most `1.00` and observed conflict fraction at least `0.10`.
- **Validation discipline:** Seeds `260919..260921` and every test split remain
  sealed until the development gate passes.
- **Status:** `not_run`; no H007 metric or checkpoint exists.
- **Boundary:** Simulator optimizer evidence only; no real-world, business or
  production claim.

## 2026-07-15 — CHM-W-H007 development preflight

- **Protocol:** Seed `260918`, 300 steps, paired WG1 trajectories and identical
  hybrid evaluator; frozen validation and test remained sealed.
- **PCGrad behavior:** Conflict projection activated on 54.33% of steps. Mean
  cosine was `0.02828`, with range `[-0.84141, 0.87488]`.
- **Failure:** The selected checkpoint stayed at untrained step 0. By step 300,
  rollout reached `0.49750`, but effect NRMSE diverged to `12.60578` and 90%
  coverage collapsed to `0.0`.
- **Controls:** Standard mixed trained normally to effect `0.81995`, rollout
  `0.47587`; matched random reached `0.84268`, `0.48917`.
- **Decision:** Reject symmetric PCGrad, do not open seeds `260919..260921`, do
  not open test and do not promote any checkpoint.
- **Diagnosis:** Conflict is real, but global symmetric projection overcorrects
  the heteroscedastic effect task. The standard mixed model remains seed
  sensitive; stability should be addressed at prediction aggregation rather
  than by altering shared training gradients.
- **Boundary:** Development optimizer evidence only; no transfer or production
  claim.

## 2026-07-15 — CHM-W-H008 registration

- **Ensemble audit:** Four matched 3-member windows and one 6-member ensemble
  were evaluated only on already-opened validation. Six-member effect NRMSE was
  `0.82647` mixed versus `0.82491` random; ratio `1.00190`.
- **Decision:** Do not spend production inference compute on ensembling a policy
  advantage that disappears after aggregation.
- **Question:** Does enforcing effect as factual utility minus no-op utility
  improve intervention prediction over an unconstrained direct-effect head?
- **Change:** Reinterpret the fourth head channel as no-op utility. Emit effect
  mean by exact subtraction and effect variance as the sum of factual/no-op
  variances. Head width and parameter count remain matched.
- **Controls:** Counterfactual/direct heads under both mixed and random policies;
  all other architecture, optimizer, data and evaluator settings are identical.
- **Development gate:** Seed `260922`; require effect ratio at most `0.90`,
  rollout ratio at most `1.00` versus direct mixed, and exact identity residual
  below `1e-6`.
- **Validation discipline:** Seeds `260923..260925` and every test split remain
  sealed until the development gate passes.
- **Status:** `not_run`; no H008 metric or checkpoint exists.
- **Boundary:** Simulator outcome-head evidence only; no real-world, business or
  production claim.

## 2026-07-16 — CHM-W-H008 implementation

- **Head:** Implemented a parameter-matched relational wrapper whose fourth raw
  channel predicts no-op utility and whose public intervention effect is the
  exact factual-minus-no-op difference.
- **Uncertainty:** Effect variance is the sum of factual and no-op variances
  under the preregistered zero-covariance assumption.
- **Controls:** Added matched mixed-policy and random-policy direct heads,
  one-step relational and temporal baselines, plus legal random intervention
  regret.
- **Integrity:** Reuse the already validated WG1 replay/leakage evidence only
  after matching its generator-config SHA-256; the dataset is not revalidated.
- **Execution:** Added a deterministic six-arm suite that writes one development
  gate report while keeping frozen validation seeds and all test splits sealed.
- **Status:** Engineering implementation pending smoke, GPU and full-suite
  verification; no H008 model-quality result yet.
- **Boundary:** Implementation evidence only; no transfer, causal, business or
  production claim.

## 2026-07-16 — CHM-W-H008 development preflight

- **Protocol:** Ran all six trainable arms for 300 steps with seed `260922` on
  RTX 5070 BF16, plus 64 legal-random intervention samples. Reused the existing
  WG1 replay/leakage evidence by matching generator SHA-256; did not revalidate
  the dataset.
- **Primary effect:** Counterfactual mixed NRMSE was `1.046694` versus
  `1.081250` for the direct mixed head, ratio `0.968041`. This is a 3.2%
  reduction, below the preregistered 10% requirement.
- **Rollout:** Four-step NRMSE was `0.509629` versus `0.509794`, ratio
  `0.999677`; the non-regression gate passed.
- **Structure:** Identity residual was exactly `0.0`, coverage was `1.0`, and
  matched relational arms each had `65,213,950` trainable parameters. Implied
  no-op utility NRMSE improved by 11.9%, ratio `0.881272`.
- **Robustness:** Under paired-random training, counterfactual/direct effect
  ratio was `0.935644` and no-op ratio was `0.910206`. The gain is directionally
  consistent but still below the primary effect threshold.
- **Baselines:** Counterfactual mixed improved effect over one-step relational
  by 17.4% but worsened rollout by 7.7%; it improved both effect and rollout
  over the temporal baseline by 5.6% and 16.1%, respectively.
- **Decision:** Development gate failed only on effect ratio. Do not open seeds
  `260923..260925`, do not open any test split, and do not promote a checkpoint.
- **Next action:** Preserve explicit factual/no-op semantics, but move the
  constraint upstream into learned transition factors so the decomposition can
  affect state rollout rather than only the terminal outcome head.
- **Boundary:** Development simulator evidence only; no real-world causal,
  business-utility, language-independence or production claim.

## 2026-07-15 — CHM-W-H009 registration

- **Question:** Does cross-world pretraining transfer mechanisms when alignment
  positives include two renderer views of the exact same hidden trajectory?
- **Predecessor:** H002 generic alignment was worse than no alignment on
  validation: effect ratio `1.04459`, rollout ratio `1.00767`; its test stayed
  sealed.
- **Change:** For each mechanism, sample two world-family realizations and two
  renderer views per world. Renderer and dynamics RNG streams are independent;
  paired views share latent actions, exogenous events and numerical outcomes.
- **Controls:** Matched no-alignment, target-family-only, non-relational temporal
  and legal-random baselines remain required.
- **Primary test:** Intervention-effect NRMSE and four-step rollout NRMSE in
  `test_world_transfer`, opened only after validation selection is frozen.
- **Acceptance:** Both errors must be at most `0.90` of the strongest baseline,
  with paired 90% bootstrap ratio upper bounds below `1.00`.
- **Status:** `not_run`; no H009 metric, checkpoint or transfer claim exists.
- **Boundary:** Simulator transfer evidence only; no causal-discovery,
  business-utility, language-independence or production claim.

## 2026-07-16 — CHM-W-H009 development preflight

- **Protocol:** Seed `260926`, 1,000 BF16 steps per arm on RTX 5070. Only
  generated `train` and checkpoint-selection `validation` were opened.
- **Aligned:** Effect NRMSE `0.89387`, rollout NRMSE `0.44715`, retrieval
  accuracy `0.45312` and effect coverage `0.97321`.
- **No alignment:** Effect `0.89338`, rollout `0.44775`, retrieval `0.29688`.
  Aligned/no-alignment ratios were `1.00056` and `0.99866`.
- **Temporal control:** Effect `0.99010`, rollout `0.46294`. Relational state
  remains useful, but the paired alignment objective does not improve its
  prediction heads over the matched relational control.
- **Diagnosis:** Exact renderer pairs make mechanism identity easier to
  retrieve (`+0.15625`) without improving intervention or rollout prediction.
  The alignment embedding and predictive state can still specialize into
  weakly coupled representations.
- **Decision:** Fail the H009 development gate. Do not open seeds
  `260927..260929`, do not open any test split and do not promote checkpoints.
- **Status:** H009 remains `not_run`; these are development diagnostics, not a
  registered transfer result.
- **Boundary:** Simulator-only evidence; no causal-discovery, business-utility,
  language-independence or production claim.

## 2026-07-16 — CHM-W-H010 registration

- **Diagnosis:** H009 alignment increased mechanism retrieval by `0.15625`, but
  aligned/no-alignment effect and rollout ratios stayed at `1.00056` and
  `0.99866`.
- **Mechanism:** Prediction consumes raw `mechanism_state`, while alignment can
  be satisfied inside a separate projection. Retrieval can improve without
  changing the state used by transition and effect heads.
- **Change:** Use `mechanism_projection(mechanism_state)` as one shared
  bottleneck for both predictive conditioning and normalized alignment.
- **Controls:** Cross alignment and bottleneck presence in a matched 2×2 design:
  shared/separate projection × alignment on/off. Architecture size, data,
  optimizer and evaluator remain fixed.
- **Development gate:** Seed `260930`; require effect ratio ≤ `0.90`, rollout
  ratio ≤ `1.00`, effect coverage ≥ `0.85` and structural path audits.
- **Validation discipline:** Seeds `260931..260933` and all test splits remain
  sealed until the development gate passes.
- **Status:** `not_run`; no H010 metric or checkpoint exists.
- **Boundary:** Simulator representation evidence only; no real-world,
  business-utility, language-independence or production claim.

## 2026-07-16 — CHM-W-H010 development preflight

- **Protocol:** Four matched 1,000-step BF16 arms on seed `260930`; train and
  checkpoint-selection validation only.
- **Shared + alignment:** Effect NRMSE `0.89336`, rollout `0.44687`, retrieval
  `0.56250`; projection prediction delta `4.44e-4`.
- **Separate + alignment:** Effect `0.91784`, rollout `0.44753`, retrieval
  `0.39062`; structural delta exactly `0`.
- **Shared without alignment:** Effect `0.90643`, rollout `0.44759`.
- **Separate without alignment:** Best effect `0.88980`, rollout `0.44818`.
- **Factor diagnosis:** Sharing the path made alignment directionally useful
  versus both aligned and shared-path controls, but shared+aligned remained
  `1.00400×` the strongest effect baseline. Retrieval improvement still does
  not imply superior intervention prediction.
- **Decision:** Fail H010; do not open seeds `260931..260933`, do not open test
  and do not promote any checkpoint.
- **Status:** H010 remains `not_run`; no registered transfer result exists.
- **Boundary:** Simulator-only evidence; no real-world, business-utility,
  language-independence or production claim.

## 2026-07-16 — CHM-W-H011 registration

- **Diagnosis:** H010 made alignment structurally predictive, yet its best
  effect ratio was `1.00400`. Global embedding similarity remains an indirect
  objective.
- **Change:** For two renderer views of the exact same hidden trajectory,
  directly match predicted primary effect mean and log variance. Pair keys are
  evaluator-only and are never passed to model forward.
- **Controls:** Identical relational model with consistency weight on/off; the
  mechanism-alignment objective is disabled in both arms.
- **Development gate:** Seed `260934`; require effect ratio ≤ `0.90`, rollout
  ratio ≤ `1.00`, coverage ≥ `0.85` and pair disagreement ratio ≤ `0.80`.
- **Validation discipline:** Seeds `260935..260937` and every test split remain
  sealed until development passes.
- **Status:** `not_run`; no H011 metric or checkpoint exists.
- **Boundary:** Simulator response-function evidence only; no real-world,
  business-utility, language-independence or production claim.

## 2026-07-16 — CHM-W-H011 implementation

- **Pairing:** Added stable evaluator-only world-instance keys. Each key groups
  two renderer views with identical latent dynamics, interventions and outcomes.
- **Objective:** Added direct Smooth L1 consistency for primary intervention
  effect mean and log variance. Global mechanism alignment stays disabled.
- **Isolation:** Pair keys are created after the language-free generated batch,
  are consumed only by the trainer/evaluator and are not read by model forward.
- **Controls:** Treatment and control share model parameters, generator, seed,
  optimizer, train budget and validation checkpoint selector.
- **Status:** Engineering implementation only; development metrics are not yet
  recorded and `research/results/CHM-W-H011.json` remains `not_run`.

## 2026-07-16 — CHM-W-H011 development preflight

- **Scope:** Two matched 1,000-step BF16 arms on development seed `260934`;
  only `train` and `validation` were opened.
- **Response consistency:** Effect NRMSE `0.907379`, rollout NRMSE `0.447798`,
  coverage `0.973214`, paired mean disagreement `0.002203`.
- **Matched control:** Effect NRMSE `0.903804`, rollout NRMSE `0.447938`,
  coverage `0.973214`, paired mean disagreement `0.002498`.
- **Controlled result:** Consistency reduced paired mean disagreement by
  `11.81%` and kept rollout non-inferior (`0.999689x`), but effect ratio was
  `1.003955` and disagreement ratio `0.881922`. The registered `0.90` effect
  and `0.80` disagreement gates both failed.
- **Decision:** Do not open seeds `260935..260937`, do not open test and do not
  promote either checkpoint. H011 remains `not_run`.
- **Diagnosis:** Making two renderer views agree is not sufficient to make the
  shared response numerically correct. The next structural experiment should
  constrain the counterfactual quantity itself against factual/no-op utility.
- **Next action:** Implement the already preregistered H008 counterfactual
  outcome-head decomposition before registering another alignment variant.
- **Boundary:** Development simulator evidence only; no real-world, causal,
  business-utility, language-independence or production claim.
