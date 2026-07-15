# Closed-Loop Training H003

`CHM-W-H003` tests whether multi-step closed-loop learning and a larger
mechanism-negative pool improve transfer inside the frozen generated-world
distribution. It does not change H002 data or open any test split.

## Method

For each train batch, the relational model receives four observed context
steps. It predicts the next signal state, inserts that prediction into its own
history and repeats for four actions:

```text
observed context
  -> predict state t+1
  -> reuse predicted state
  -> predict state t+2
  -> reuse predicted state
  -> predict state t+3
  -> reuse predicted state
  -> predict state t+4
```

Loss is the equal-horizon mean of Gaussian state and outcome losses. The four
dynamic renderer channels are targets. Independently resampled nuisance
channels remain visible in the original context and are zeroed when a predicted
state is fed back.

## Mechanism Queue

Each generated mechanism has two renderer/world views in the current batch.
Their stable numeric fingerprint is available only to the loss and evaluator.
The model never receives that fingerprint.

After an optimizer step, normalized mechanism embeddings are detached and added
to a FIFO queue. Once the queue contains 256 entries, its distinct mechanisms
join the current batch as hard negatives. The queue holds at most 2,048 entries
and never propagates gradients into earlier batches.

```text
current views -> positive pairs
current distinct mechanisms + detached queue -> hard negatives
```

## Comparison Arms

1. closed-loop relational training with cross-batch mechanism discrimination;
2. the same closed-loop training without mechanism discrimination;
3. H002 one-step relational training without alignment;
4. temporal prediction without relational state;
5. legal random interventions in the frozen trial.

## Validation Gate

Across seeds `260903`, `260904` and `260905`, the full arm may freeze
`CHM-W-T003` only if:

- intervention-effect NRMSE and four-step rollout NRMSE are each at most `0.95`
  times the strongest learned baseline;
- median mechanism retrieval is at least `0.10`;
- intervention-effect 90% coverage is at least `0.85`;
- deterministic replay is exact, every metric is finite and leakage findings
  remain zero.

Only `train` and `validation` may be materialized during preflight. Failure of
this gate leaves `CHM-W-H003` as `not_run` and forbids test access.

## Commands

```powershell
chimera meta-world-h003-preflight `
  --config configs/meta_world/world_h003_preflight_smoke.yaml `
  --output runs/h003_preflight_smoke

chimera meta-world-h003-preflight `
  --config configs/meta_world/world_h003_preflight_closed_loop.yaml `
  --output runs/h003_preflight_closed_loop_a
```

## Claim Boundary

Even a passing result would establish transfer only inside the frozen simulator
distribution. It would not establish real-world causal discovery, profitable
business ideas, language-independent thought or production readiness.
