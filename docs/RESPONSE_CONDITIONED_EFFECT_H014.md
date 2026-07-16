# Response-Conditioned Effect H014

`CHM-W-H014` tests whether the paired no-op signal discovered in H013 must be
used directly by the effect predictor.

## Shared transition

Both arms use the H013 direct dual-transition model:

```text
factual_next_hat = direct_factual_transition(z, action)
no_op_next_hat   = no_op_transition(z)
```

They receive identical factual, no-op and state-delta losses. No target tensor
is passed to the effect head.

## Controlled response source

The experimental arm pools the predicted no-op-subtracted response:

```text
response = factual_next_hat - no_op_next_hat
```

The matched control pools the predicted factual residual:

```text
response_control = factual_next_hat - final_observed_state
```

Each source is reduced with masked slot mean and standard deviation over the
four registered state channels, encoded by the same-size response adapter and
fused with the same transition representation. Both effect heads have identical
parameter counts. Only the numerical response source differs.

## Reused data boundary

H014 reuses WG4 and the existing integrity artifact by SHA-256. It does not
repeat dataset validation. Generator provenance stays evaluator-only, the no-op
state remains a target rather than a model feature, and language is absent from
the model core.

## Development gate

Versus the parameter-matched factual-residual control, the response-conditioned
arm must achieve:

- intervention-effect NRMSE ratio at most `0.90`;
- four-step rollout, state-delta and no-op-state NRMSE ratios at most `1.00`;
- effect 90% coverage at least `0.85`;
- factual/no-op outcome identity residual at most `1e-6`;
- exact dataset replay, zero leakage, finite metrics and sealed test access.

Failure keeps frozen validation and test sealed and promotes no checkpoint.

## Commands

```powershell
python -m chimera.cli meta-world-h014-preflight `
  --config configs/meta_world/world_h014_development_response.yaml `
  --output runs/h014_development_response

python -m chimera.cli meta-world-h014-suite
```

## Claim boundary

This protocol can establish a simulator representation result only. It cannot
establish real-world causality, safe interventions, profitable ideas,
language-independent thought or production readiness.
