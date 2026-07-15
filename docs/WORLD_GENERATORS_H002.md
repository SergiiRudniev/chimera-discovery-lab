# Generated Worlds H002

`CHM-W-H002` tests whether cross-world pretraining transfers numerical
mechanisms to unseen world-family mappings. It does not test business value or
real-world causal discovery.

## Generation Stack

```text
seed
  -> MechanismGenerator
  -> WorldGenerator
  -> ObservationRenderer
  -> actions and trajectory
  -> language-free tensor batch
```

### MechanismGenerator

The hidden mechanism is family-agnostic. It samples retention, nonlinearity,
threshold, delay, positive and negative feedback, saturation, competition,
interaction, hidden coupling, event rate and four latent weights. The exact
configuration is available only to generation and evaluation code.

Templates `0..3` are known mechanism classes. Templates `4..5` are reserved for
`test_mechanism`. A concrete mechanism ID is a SHA-256 fingerprint of its
numeric law and cannot cross a dataset split.

### WorldGenerator

One mechanism is mapped into one concrete numerical environment:

| Family | Hidden process |
| --- | --- |
| `FlowWorld` | finite resource movement, loss, queues, feedback and bottlenecks |
| `CompetitionWorld` | capacity allocation, accumulated advantage, cooperation and displacement |
| `FunnelWorld` | staged conversion, returns, queues, delayed release and saturation |

Each world has four hidden state channels per object. Their meanings are
family-specific. Every action has a magnitude in `[0, 1]`, a control in
`[-1, 1]` and distinct source and target objects.

The four outcome channels are:

1. family utility;
2. throughput;
3. constraint load, concentration or backlog;
4. paired intervention effect: utility under the action minus utility under a
   legal no-op from the same state and the same exogenous event.

### ObservationRenderer

The renderer cannot change the transition law. It may permute objects and
channels, change units and time scale, apply an invertible numeric
nonlinearity, add noise and nuisance channels, or hide values. Profile `2` is
held out for `test_renderer`.

## Tensor Contract

The model receives only:

```text
observations:    [batch, time, objects, 8]
object_mask:     [batch, time, objects]
relations:       [batch, time, objects, objects, 4]
relation_mask:   [batch, time, objects, objects]
actions:         [batch, time, 2]
action_targets:  [batch, time, objects]
delta_time:      [batch, time]
outcomes:        [batch, time, 4]
sequence_mask:   [batch, time]
```

`action_targets` uses `-1` for the source and `+1` for the target. Family,
instance, mechanism, renderer and seed identifiers are stored in separate
evaluator metadata and are absent from `GeneratedWorldBatch`.

## Split Policy

| Split | Mechanism class | World mapping | Renderer |
| --- | --- | --- | --- |
| `train` | templates `0..3` | two known families per template | profiles `0, 1` |
| `validation` | templates `0..3`, new concrete laws | known families | profiles `0, 1`, new configs |
| `test_world_transfer` | templates `0..3`, new concrete laws | held family per template | profiles `0, 1` |
| `test_mechanism` | templates `4..5` | all families | profiles `0, 1` |
| `test_renderer` | templates `0..3`, new concrete laws | known families | unseen profile `2` |

The exact held family map is `0→Funnel`, `1→Flow`, `2→Competition`,
`3→Funnel`. Concrete mechanism IDs, world IDs, generation seeds, exact world
configs and exact renderer configs are pairwise disjoint across all splits.
Mechanism templates may repeat where the registered question requires a known
law class under a new configuration or family.

## Generation Modes

Online CPU generation uses the same seed-addressable pipeline as fixed data:

```python
from chimera.meta_world.generators import (
    GeneratedWorldDatasetConfig,
    SplitName,
    WorldGenerationPipeline,
)

config = GeneratedWorldDatasetConfig.from_yaml(
    "configs/meta_world/world_generators_h002.yaml"
)
batch = WorldGenerationPipeline(config).online_batch(
    SplitName.TRAIN,
    batch_size=16,
    start_index=0,
)
```

Smoke and fixed evaluation commands:

```powershell
chimera world-generator-smoke --batch-size 8
chimera build-world-generator-dataset `
  --output artifacts/meta_world_generator_smoke `
  --trajectories-per-split 12
chimera validate-world-generator-dataset `
  --manifest artifacts/meta_world_generator_smoke/manifest.json
```

The fixed builder writes five small, object-free NPZ shards and a stable JSON
manifest. NPZ member order, timestamps and SHA-256 hashes are deterministic.
Large online training streams are not committed to Git.

## Registered Comparison

The evidence-bearing run compares:

1. cross-world pretraining with mechanism alignment;
2. the same model without mechanism alignment;
3. target-family-only training;
4. a temporal predictor without relational state;
5. legal random interventions.

H002 passes only if both intervention-effect RMSE and four-step rollout NRMSE
on `test_world_transfer` are at most `0.90` times the strongest baseline and
both paired 90% bootstrap ratio upper bounds are below `1.00`. Test splits stay
closed until validation checkpoint selection is frozen.

State prediction and rollout metrics cover the four renderer signal channels.
Renderer nuisance channels remain visible as model-input distractors but are
excluded from prediction targets because each step resamples them independently.

## Limitations

- The worlds encode human-selected mathematical priors; they are non-linguistic
  model inputs, not proof of thought free from human concepts.
- Three small simulator families do not approximate the full real world.
- Topology is static within a trajectory and exogenous events are simple.
- The effect target uses a simulator no-op counterfactual, not observational
  causal identification.
- Cross-platform bitwise equality across different NumPy versions is not yet a
  registered guarantee.
- H002 remains `not_run`; no transfer metrics or checkpoint exist.
