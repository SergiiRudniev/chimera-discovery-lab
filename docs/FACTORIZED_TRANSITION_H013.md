# Factorized Counterfactual Transition H013

`CHM-W-H013` tests whether the factual/no-op structure from H008 must constrain
the dynamics rather than only the terminal utility head.

## Paired world transition

For every sampled state, the generator evaluates two transitions under the
same hidden mechanism, state, external event and renderer noise:

```text
factual next state = step(state, sampled action, event)
no-op next state   = step(state, zero action, event)
```

The factual branch alone advances the trajectory. The no-op branch is a
training and evaluation target and is never a model input feature.

## Matched models

Both primary arms share the relational encoder, factual/no-op outcome
semantics, optimizer, seed, training examples and two equal-size state heads.

The direct control predicts factual and no-op next state independently:

```text
factual_hat = direct_head(z, action)
no_op_hat   = no_op_head(z)
```

The factorized arm predicts a no-op state and intervention delta:

```text
no_op_hat   = no_op_head(z)
delta_hat   = delta_head(z, action)
factual_hat = no_op_hat + delta_hat
```

The factorized identity is audited in FP32. Parameter counts must match
exactly before a comparison is eligible.

## Isolation

- Model inputs contain numeric observations, relations, actions and time only.
- Generator IDs and family labels stay evaluator-only.
- Paired counterfactual targets are never passed as input features.
- Training is online; evaluation shards are fixed by SHA-256.
- Development opens only `train` and `validation`.
- Frozen validation and all test splits remain sealed until the gate passes.

## Development gate

Versus the matched direct model, the factorized arm must achieve:

- intervention-state-delta NRMSE ratio at most `0.90`;
- factual four-step rollout NRMSE ratio at most `1.00`;
- no-op-state NRMSE ratio at most `1.00`;
- intervention-effect NRMSE ratio at most `1.00`;
- exact identity residual at most `1e-6`;
- deterministic replay rate `1.00`, zero leakage and finite metrics.

Failure keeps frozen validation and test sealed and promotes no checkpoint.

## Claim boundary

This protocol can establish a numerical representation result inside the
registered procedural simulators only. It does not establish real-world
causality, safe business interventions, idea quality, language-independent
thought or production readiness.
