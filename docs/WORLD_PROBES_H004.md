# System-Identification Probes H004

`CHM-W-H004` changes how generated worlds are excited before Meta-World is
asked to transfer a mechanism. The state, relation and outcome laws are the
same programmatic families used by H002. Only the numeric action policy and
trajectory length change.

## WG1 Dataset

`CHM-W-WG1` contains 16-step trajectories from `FlowWorld`,
`CompetitionWorld` and `FunnelWorld`. Every split retains H002 isolation by
concrete mechanism, world instance, seed, world configuration and renderer
configuration.

The model tensor contract is unchanged:

```text
observations:    [batch, 16, objects, 8]
object_mask:     [batch, 16, objects]
relations:       [batch, 16, objects, objects, 4]
relation_mask:   [batch, 16, objects, objects]
actions:         [batch, 16, 2]
action_targets:  [batch, 16, objects]
delta_time:      [batch, 16]
outcomes:        [batch, 16, 4]
sequence_mask:   [batch, 16]
```

Action-policy IDs and generator provenance remain in the manifest/evaluator
boundary and never enter this batch.

## Probe Program

Training trajectories repeat an eight-step numeric excitation block across two
object pairs:

| Phase | Magnitude | Control | Direction |
| --- | ---: | ---: | --- |
| baseline | 0.00 | 0.0 | forward |
| low impulse | 0.25 | 0.0 | forward |
| high positive | 0.85 | 1.0 | forward |
| high negative | 0.85 | -1.0 | reverse |
| recovery | 0.00 | 0.0 | forward |
| alternate impulse | 0.25 | 0.0 | alternate edge |
| alternate negative | 0.85 | -1.0 | alternate edge |
| alternate positive | 0.85 | 1.0 | reverse alternate edge |

The schedule exposes magnitude response, control polarity, directionality,
recovery, delay and feedback without naming the simulated objects or mechanism.

## Evaluation Policy

Validation and every sealed test split use the same policy for every model arm:

```text
four registered probe steps -> seeded legal random interventions
```

This gives the model a short active-identification prefix and measures whether
the inferred law predicts later interventions. The random-curriculum comparator
uses exactly the same hybrid evaluator; only its training actions differ.

## Fixed Smoke Dataset

The fixed smoke artifact contains 16 trajectories per split and is not
committed. Its manifest SHA-256 is:

```text
cc0305bd99f05cf5d528f045dd494652d377be2c5836d5922d589eb6d3b96461
```

All 20 integrity, replay, shape, finite-value, isolation, policy and excitation
checks pass. The diagnostic between/within probe-response separation is
`1.235036`; this confirms non-zero distinguishable responses but is not a model
quality result.

## Commands

```powershell
chimera build-world-probe-dataset `
  --output artifacts/meta_world_probe_smoke `
  --trajectories-per-split 16

chimera validate-world-probe-dataset `
  --manifest artifacts/meta_world_probe_smoke/manifest.json
```

## Claim Boundary

WG1 is synthetic numerical training/evaluation data. It does not establish
real-world causal discovery, safe real-world probing, business utility or
production readiness.
