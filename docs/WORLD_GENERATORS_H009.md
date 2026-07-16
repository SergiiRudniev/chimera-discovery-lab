# Paired Generated Worlds H009

`CHM-W-H009` is a preregistered successor to H002. It tests transfer between
numerical simulator worlds; it does not test real-world causal discovery or
business profitability.

## Generation stack

```text
seed
  -> MechanismGenerator
  -> two WorldGenerator realizations
  -> two ObservationRenderer views per world
  -> shared latent interventions and exogenous events per renderer pair
  -> language-free tensor batch
```

`MechanismGenerator` samples hidden retention, nonlinear response, thresholds,
delay, feedback, saturation, competition, interaction, coupling, event rate and
latent weights. The exact mechanism configuration is available only to dataset
generation, alignment loss and evaluation. It is never passed to model forward.

`WorldGenerator` maps the mechanism into three minimal dynamic families:

| Family | Numerical dynamics |
| --- | --- |
| `FlowWorld` | finite flows, loss, queues, delays, feedback and bottlenecks |
| `CompetitionWorld` | capacity allocation, saturation, accumulated advantage, cooperation and displacement |
| `FunnelWorld` | staged transitions, conversion, returns, queues, delayed release and saturation |

`ObservationRenderer` changes only representation: object and channel order,
relation-channel order, units, offsets, time scale, invertible nonlinearities,
noise, nuisance channels and partial visibility. Schema v2 gives renderer noise
an RNG stream independent of hidden dynamics.

Each mechanism group contains two world realizations. Each realization has two
renderer views with identical latent initial state, action sequence, exogenous
events and outcomes. Actions are translated from latent object coordinates into
the coordinates of each renderer. This provides an exact alignment positive
without exposing the renderer or mechanism ID to model forward.

## Tensor contract

The model receives only:

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

Outcome channels are utility, throughput, constraint load and paired
intervention effect. Generator metadata (`world_family_id`, instance,
mechanism, renderer and seeds) is stored beside fixed evaluation tensors and is
excluded from `GeneratedWorldBatch`.

## Split policy

| Split | Mechanisms | World mapping | Renderer |
| --- | --- | --- | --- |
| `train` | templates `0..3` | two known families per mechanism | profiles `0, 1` |
| `validation` | new configurations of templates `0..3` | known families | new profile `0, 1` configs |
| `test_world_transfer` | new configurations of templates `0..3` | held family | profiles `0, 1` |
| `test_mechanism` | unseen templates `4, 5` | generated families | profiles `0, 1` |
| `test_renderer` | new configurations of templates `0..3` | known families | unseen profile `2` |

No two splits may share a concrete mechanism ID, world instance, generation
seed, exact world configuration or exact renderer configuration. Validation
checks file and source SHA-256, exact replay, tensor shapes, finite values,
split isolation and equality of outcomes inside each renderer pair.

## Generation modes

Online CPU batches use the seed-addressable pipeline directly. Fixed smoke and
evaluation datasets are small deterministic NPZ shards with a JSON manifest.
Large online training streams are not committed.

```powershell
chimera world-generator-smoke `
  --config configs/meta_world/world_generators_h009.yaml `
  --batch-size 4

chimera meta-world-h009-smoke-dataset `
  --output artifacts/meta_world_h009_smoke

chimera meta-world-h009-preflight `
  --config configs/meta_world/world_h009_development_smoke.yaml `
  --output runs/h009_smoke
```

The smoke preflight opens only `train` and `validation`. It is an engineering
check and cannot update `research/results/CHM-W-H009.json`.

## Registered comparison

The future evidence-bearing trial compares paired cross-world pretraining with
alignment against the matched no-alignment model, target-family-only training,
a non-relational temporal predictor and legal random intervention. H009 passes
only if intervention-effect NRMSE and four-step rollout NRMSE on
`test_world_transfer` are each at most `0.90` of the strongest baseline and
both paired 90% bootstrap ratio upper bounds are below `1.00`.

## Limitations

- The generator family and priors are human-designed mathematical structures.
- Three simulators do not approximate the full world or business environment.
- Static topology and simple exogenous events limit mechanism diversity.
- Simulator no-op effects are not observational causal identification.
- A passing trial would not demonstrate profitable ideas or thought independent
  of all human conceptual choices.
