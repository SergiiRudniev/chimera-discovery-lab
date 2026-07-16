# Procedural Pretraining H012

`CHM-W-H012` qualifies a generator-first training protocol. It reuses the
audited H002/H009 simulator stack and adds a frozen five-arm comparison. No
language or business labels enter the model.

## Generation architecture

```text
seed
  -> MechanismGenerator
  -> WorldGenerator
  -> ObservationRenderer
  -> numeric trajectory
  -> language-free tensor batch
```

`MechanismGenerator` samples a family-independent hidden transition law:
retention, nonlinear response, threshold, delay, positive and negative
feedback, saturation, competition, interaction, hidden coupling, event rate
and latent outcome weights. The exact configuration is generator/evaluator
metadata and is excluded from model forward.

`WorldGenerator` maps the law into a concrete topology, parameters, initial
state, legal action space and exogenous event stream. H012 uses three families:

| Family | Minimal dynamics |
| --- | --- |
| `FlowWorld` | finite flows, loss, queues, delays, feedback and bottlenecks |
| `CompetitionWorld` | capacity allocation, saturation, accumulated advantage, cooperation and displacement |
| `FunnelWorld` | staged transitions, conversion, returns, queues, delayed release and saturation |

`ObservationRenderer` changes representation without changing hidden dynamics.
It permutes objects and channels, changes units and time scale, applies an
invertible nonlinearity, adds noise and nuisance channels, and masks part of the
state. Paired renderer views share latent actions, events and outcomes.

## Runtime contract

Every world implements reset, legal numeric action sampling and action-driven
transition. Seeds deterministically reproduce the mechanism, world,
renderer, state and trajectory.

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

The outcome channels are utility, throughput, constraint load and paired
intervention effect. `world_family_id`, `world_instance_id`, `mechanism_id`,
`renderer_id` and every generation seed are stored outside this batch.

## Dataset modes

- Online CPU generation provides new seed-addressed train trajectories while
  the GPU trains the model.
- Fixed NPZ shards provide small validation, test and smoke datasets. The JSON
  manifest records shard, source and configuration SHA-256 values.

Large generated streams are not committed to Git.

## Split policy

| Split | Mechanisms | World mapping | Renderer |
| --- | --- | --- | --- |
| `train` | templates `0..3` | two source families per template | profiles `0, 1` |
| `validation` | new configs of templates `0..3` | source families | new profile `0, 1` configs |
| `test_world_transfer` | new configs of templates `0..3` | held family per template | profiles `0, 1` |
| `test_mechanism` | unseen templates `4, 5` | generated families | profiles `0, 1` |
| `test_renderer` | new configs of templates `0..3` | source families | unseen profile `2` |

The target-family-only arm changes only online train family allocation: it uses
the held family with train-only seeds. Validation and test configurations remain
disjoint. Every fixed dataset validator checks that splits do not share a
concrete mechanism, world instance, generation seed, exact mechanism/world
configuration or exact renderer configuration.

## Frozen comparison

H012 compares:

1. relational cross-world pretraining with evaluator-only mechanism alignment;
2. the same model without alignment;
3. the same model trained only on the held target family;
4. a temporal predictor without relational world state;
5. legal random intervention, eligible only for intervention regret.

The trainable arms use the same seed, trajectory count, optimizer budget and
validation selector. The random baseline samples legal one-step interventions
and reports regret against seeded alternative actions from the same latent
state; it is not eligible for predictive-error comparison.

The primary test remains sealed until all arms and checkpoints are frozen. H012
passes only if intervention-effect NRMSE and four-step rollout NRMSE on
`test_world_transfer` are each no more than `0.90` of the strongest eligible
predictive baseline and both paired 90% bootstrap ratio bounds are below
`1.00`. Replay must be exact, all metrics finite and leakage findings zero.

## Commands

```powershell
chimera meta-world-h012-smoke-dataset `
  --output artifacts/meta_world_h012_smoke

chimera meta-world-h012-preflight `
  --config configs/meta_world/world_h012_development_smoke.yaml `
  --output runs/h012_development_smoke

pytest
ruff check .
mypy src
```

The smoke commands are engineering checks and cannot update
`research/results/CHM-W-H012.json`.

## Numerical output boundary

The future model output is a frozen numeric proposal: legal intervention
vectors, affected object slots, predicted outcome distributions, uncertainty
and evaluator provenance. An external language model may describe that proposal
only after generation; it cannot change the intervention or add a mechanism.

## Known limitations

- The mechanism primitives and parameter priors are human-designed.
- Three simulator families do not represent the full world or a business.
- Topologies and exogenous events remain deliberately small and synthetic.
- Simulator counterfactual effects are not observational causal discovery.
- A passing result would not establish real-world transfer, safe intervention,
  profitable ideas or thought independent of all human concepts.
