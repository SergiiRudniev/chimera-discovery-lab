# Compositional World Transfer H018

`CHM-W-H018` tests whether a numerical world model can transfer known dynamic
operators in combinations that were never present in training. It extends the
H002/H009/H012 generator stack; it does not replace the existing world laws.

## Generation architecture

```text
seed
  -> MechanismProgramGenerator
  -> WorldGenerator
  -> ObservationRenderer
  -> numeric trajectory
  -> language-free tensor batch
```

`MechanismProgramGenerator` composes hidden retention/loss, threshold,
delay/feedback, saturation, competition, coupling and event operators. The
program and exact mechanism are evaluator-only metadata. The model receives no
operator ID, family ID, renderer ID, seed, business label or text.

Six two-operator programs are available in training. Three other two-operator
programs use only individually known primitives but are held out as complete
compositions for `test_world_transfer`. Two three-operator programs are held
for `test_mechanism`.

The compiled mechanism is realized independently as `FlowWorld`,
`CompetitionWorld` and `FunnelWorld`. `ObservationRenderer` changes object and
channel order, units, time scale, invertible nonlinearity, noise, visibility
and nuisance channels without changing the latent transition.

## Tensor contract

```text
observations:       [batch, time, objects, 8]
object_mask:        [batch, time, objects]
relations:          [batch, time, objects, objects, 4]
relation_mask:      [batch, time, objects, objects]
actions:            [batch, time, 2]
action_targets:     [batch, time, objects]
delta_time:         [batch, time]
outcomes:           [batch, time, 4]
sequence_mask:      [batch, time]
```

Counterfactual no-op observations are target tensors. Generator provenance is
stored outside the model batch.

## Split policy

| Split | Mechanism programs | World mapping | Renderer |
| --- | --- | --- | --- |
| `train` | two-operator programs `0..5` | two source families | profiles `0, 1` |
| `validation` | new parameters of programs `0..5` | source families | new profile `0, 1` configs |
| `test_world_transfer` | unseen two-operator programs `6..8` | held family | profiles `0, 1` |
| `test_mechanism` | unseen three-operator programs `9, 10` | generated families | profiles `0, 1` |
| `test_renderer` | new parameters of programs `0..5` | source families | unseen profile `2` |

No exact program, mechanism, world instance, seed or exact
mechanism/world/renderer configuration may cross a registered boundary.

## Frozen comparison

H018 compares aligned cross-world pretraining, matched pretraining without
alignment, target-family-only training, a non-relational temporal predictor and
legal random interventions. Trainable arms share the seed, trajectory count,
optimizer budget and validation selector. Random interventions are eligible
only for intervention regret.

H018 passes only if both intervention-effect NRMSE and four-step rollout NRMSE
on `test_world_transfer` are no more than `0.90` of the strongest eligible
predictive baseline and both paired 90% bootstrap ratio bounds are below
`1.00`. Replay must be exact, all metrics finite, program overlap zero and
leakage findings zero.

## Output boundary

The model freezes a numerical intervention hypothesis before any external
language interpretation. A language model may describe it afterward but may
not change its action, predicted effect, uncertainty or provenance.

## Commands

```powershell
chimera meta-world-h018-smoke-dataset `
  --output artifacts/meta_world_h018_smoke

chimera meta-world-h018-preflight `
  --config configs/meta_world/world_h018_development_smoke.yaml `
  --output runs/h018_development_smoke

chimera meta-world-h018-suite `
  --config configs/meta_world/world_h018_suite.yaml `
  --output runs/h018_development `
  --report research/preflights/CHM-W-H018-development.json

pytest
ruff check .
mypy src
```

Smoke and development commands cannot update the registered result or promote
a checkpoint.

## Known limitations

- The operator vocabulary and priors are human-designed.
- Compiled programs remain small synthetic dynamic systems.
- Three world families cannot represent the full world or a business.
- Simulator counterfactuals are not observational causal discovery.
- Passing would not establish safe intervention, profitable ideas or
  production readiness.
