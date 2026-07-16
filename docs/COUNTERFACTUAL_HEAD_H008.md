# Counterfactual Outcome Head H008

CHM-W-H008 tests whether an exact numerical decomposition improves prediction
of intervention effects in generated worlds.

## Numerical contract

The relational model emits four raw outcome means:

```text
[factual utility, throughput, constraint, no-op utility]
```

The public effect vector remains compatible with the WG1 evaluator:

```text
[factual utility, throughput, constraint, factual utility - no-op utility]
```

The intervention-effect variance is the sum of factual and no-op variances.
The first version assumes zero covariance. The following identity is enforced by
construction rather than by a penalty:

```text
predicted factual utility - predicted intervention effect
    == predicted no-op utility
```

No derived no-op target, action-policy ID, world-family ID, mechanism ID,
renderer ID or generation seed is passed to the model forward method.

## Matched comparison

The suite runs six trainable arms and one evaluator-only baseline:

1. counterfactual head, mixed probe/random closed-loop training;
2. direct head, mixed probe/random closed-loop training;
3. counterfactual head, paired-random closed-loop training;
4. direct head, paired-random closed-loop training;
5. direct one-step relational model;
6. direct temporal model without relational state;
7. legal random intervention regret.

Counterfactual and direct relational heads have identical trainable parameter
counts. The only controlled change in the primary comparison is the semantics
of the fourth raw outcome channel. Both perform outcome-head arithmetic in FP32
under BF16 autocast so the registered algebraic residual is not dominated by
reduced-precision subtraction. Every arm reseeds model initialization before
parameters are created, so matched arms begin from identical underlying weights.

On compute capability 12.x GPUs, GRU execution uses PyTorch's native CUDA path.
The installed cuDNN 9/CUDA 13 stack completes the forward pass but aborts the
Windows process during teardown; the native path preserves a clean process exit
and is applied identically to every H008 arm.

## Data boundary

H008 reuses validated `CHM-W-WG1` generator evidence from H005. It does not
rebuild or manually re-review the dataset. The suite verifies that the SHA-256
of the generator configuration matches the earlier integrity record before it
accepts replay and leakage evidence.

Only `train` and `validation` are opened during development. Frozen validation
seeds `260923..260925` and every test split remain sealed unless the development
gate passes.

## Development gate

The counterfactual mixed arm must satisfy all of the following versus the direct
mixed arm:

- intervention-effect NRMSE ratio at most `0.90`;
- four-step rollout NRMSE ratio at most `1.00`;
- intervention-effect 90% coverage at least `0.85`;
- maximum algebraic identity residual at most `1e-6`;
- matched parameter count;
- deterministic replay rate `1.0` and zero split-leakage findings;
- finite metrics and unopened test splits.

Passing development does not promote a checkpoint. It only permits the three
registered frozen-validation seeds to be opened.

## Commands

```powershell
$env:PYTHONPATH=(Resolve-Path src).Path
python -m chimera.cli meta-world-h008-preflight `
  --config configs/meta_world/world_h008_development_counterfactual_mixed.yaml `
  --output runs/h008_development_counterfactual_mixed

python -m chimera.cli meta-world-h008-suite `
  --config configs/meta_world/world_h008_development_suite.yaml `
  --output runs/h008_development `
  --report research/preflights/CHM-W-H008-development.json
```

## Claim boundary

A passing result would support this algebraic outcome-head constraint only in
the registered generated simulators. It would not establish real-world causal
discovery, safe experimentation, business value, language-independent thought
or production readiness.
