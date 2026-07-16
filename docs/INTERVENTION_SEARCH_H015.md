# Numerical Intervention Search H015

`CHM-W-H015` is the first Meta-World hypothesis whose primary operation is
candidate generation rather than prediction.

## Language-free candidate

Each candidate is a legal numerical vector:

```text
[source_slot, target_slot, magnitude, control]
```

The model emits a predicted effect distribution and uncertainty. Names,
sentences and business interpretation are absent from search and evaluation.

## Equal-budget comparison

For each of 32 validation states, an evaluator creates a fixed pool of 256 legal
actions. Its best realized simulator effect is the finite-pool oracle used only
to compute regret.

Both model-guided arms receive 256 model scores and may execute only eight final
candidates in the simulator. The legal-random arm executes eight candidates
from the same fixed pool. No arm can use oracle effects during selection.

The experimental score is:

```text
predicted effect mean - predicted effect standard deviation
```

The matched control uses predicted mean only. Both searches use four rounds of
64 candidates, eight elites per round and the same source/target/magnitude-bin
quality-diversity archive.

## Gate

The uncertainty-aware arm must reduce realized best-candidate regret to:

- at most `0.75` of legal random regret;
- at most `0.90` of mean-only search regret.

Every action must be legal, simulator and model budgets must match exactly,
search and dataset replay must be deterministic, metrics finite, leakage zero
and model test splits sealed.

## Output boundary

The frozen output contains ranked numerical interventions, predicted effect,
uncertainty, archive cell and evaluator provenance. An external grounder may
describe it later but cannot change ranks or values.

## Claim boundary

A passing result demonstrates candidate search only within the registered
procedural worlds and finite action pools. It does not demonstrate real-world
causality, safe business action, profitable or creative ideas,
language-independent thought or production readiness.

## Development result

H015 did not pass. Mean realized regret was `0.008411` for uncertainty-aware
search, `0.007959` for mean-only search and `0.008673` for legal random. The
uncertainty-aware ratios were `0.969786` versus random and `1.056837` versus
mean-only, missing the preregistered `0.75` and `0.90` gates.

All legality, exact budget, deterministic search replay, deterministic dataset
replay, finite-metric and leakage guards passed. Frozen validation seeds and
model test splits remained sealed, and no checkpoint was promoted.

The selected-candidate prediction/realization correlations were approximately
zero. The result rejects pointwise effect NLL plus raw predicted standard
deviation as the H015 search critic; it does not reject numerical intervention
search itself. The next experiment must train within-state action ordering from
multiple exact simulator interventions per state.
