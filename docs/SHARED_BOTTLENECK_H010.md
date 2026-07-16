# Shared Mechanism Bottleneck H010

`CHM-W-H010` tests whether mechanism alignment must act on the same numerical
state that conditions prediction. It does not change the generated worlds,
outcome targets or evaluation policy.

## Diagnosis

H009 improved mechanism retrieval from `0.29688` to `0.45312`, but its aligned
effect and rollout errors were effectively unchanged from no alignment. The
separate architecture allowed two paths:

```text
mechanism_state -> mechanism_condition -> prediction
mechanism_state -> mechanism_projection -> alignment
```

The projection could satisfy the retrieval objective without making its
invariants useful to the transition or effect heads.

## Registered change

H010 uses one projected bottleneck for both consumers:

```text
mechanism_state -> mechanism_projection -> mechanism_condition -> prediction
                                      \-> normalize -> alignment
```

The shared and separate models contain exactly the same parameters. Only the
path through existing modules changes.

The development design crosses two variables:

| Variant | Alignment |
| --- | --- |
| shared bottleneck | on |
| separate projection | on |
| shared bottleneck | off |
| separate projection | off |

All arms use WG2 paired worlds, seed `260930`, 1,000 BF16 steps, the same
optimizer and the same validation evaluator.

## Structural audit

The preflight perturbs `mechanism_projection` after loading the selected
checkpoint and records the maximum change in state or effect prediction.
The registered shared path must exceed `1e-6`; the separate path must remain at
or below `1e-7`.

## Commands

```powershell
chimera meta-world-h010-preflight `
  --config configs/meta_world/world_h010_development_smoke.yaml `
  --output runs/h010_smoke

chimera meta-world-h010-preflight `
  --config configs/meta_world/world_h010_development_shared_aligned.yaml `
  --output runs/h010_development_shared_aligned
```

The smoke command is engineering evidence only. Frozen validation seeds and all
test splits remain sealed until the development gate passes.
