# Paired Response Consistency H011

`CHM-W-H011` tests whether direct agreement between counterfactual response
predictions is more useful than global mechanism-embedding alignment.

## Controlled change

H010 showed that alignment can alter the predictive path and improve mechanism
retrieval without reducing intervention-effect error. H011 therefore disables
global mechanism alignment in both arms and changes one scalar:

```text
response_consistency_weight = 1.0  # H011 treatment
response_consistency_weight = 0.0  # matched control
```

Both arms use the same relational model, generator, seed, optimizer, train
budget and checkpoint selector.

## Pair semantics

WG2 emits two renderer views of one latent trajectory. A valid pair shares:

- hidden world configuration;
- latent initial state;
- action sequence;
- exogenous events;
- numerical outcomes.

Object and feature order, units, observation noise, nuisance channels and
visibility may differ. A SHA-256-derived `world_instance_key` groups the views
for the loss and evaluator. The key is stored outside `GeneratedWorldBatch`, is
copied into the loss-only label field after the model batch is created and is
never read by model forward.

## Objective

For the primary intervention-effect channel, H011 applies Smooth L1 agreement
to the predicted means and log variances inside each renderer pair:

```text
L_response = SmoothL1(mu_view, mean(mu_pair))
           + 0.1 * SmoothL1(logvar_view, mean(logvar_pair))
```

The complete loss is the unchanged H002 transition/effect objective plus
`response_consistency_weight * L_response`. Parameter count is unchanged.

## Evaluation

The development comparison uses seed `260934`, 1,000 BF16 steps and opens only
`train` and `validation`. It measures:

- intervention-effect NRMSE;
- four-step rollout NRMSE;
- intervention-effect 90% coverage;
- mean paired effect disagreement;
- mean paired uncertainty disagreement.

Development passes only if treatment/control effect ratio is at most `0.90`,
rollout ratio at most `1.00`, coverage at least `0.85` and pair-disagreement
ratio at most `0.80`. Frozen validation seeds `260935..260937` and every test
split remain sealed until all gates pass.

## Commands

```powershell
chimera meta-world-h011-preflight `
  --config configs/meta_world/world_h011_development_smoke.yaml `
  --output runs/h011_development_smoke

chimera meta-world-h011-preflight `
  --config configs/meta_world/world_h011_development_consistency.yaml `
  --output runs/h011_development_consistency
```

These commands produce engineering preflights. They cannot update the
registered H011 result or open a test split.

## Claim boundary

A passing result would support renderer-invariant response prediction only in
the registered simulator distribution. It would not demonstrate real-world
causal discovery, profitable business ideas, language-independent thought or a
production-ready generator.
