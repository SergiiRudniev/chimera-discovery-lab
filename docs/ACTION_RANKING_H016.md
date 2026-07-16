# Within-State Action Ranking H016

`CHM-W-H016` changes the search critic rather than the search algorithm.

## Failure addressed

H015 proved that legal numerical candidates can be generated, replayed and
compared under exact budgets. It failed because pointwise effect predictions
did not order alternative actions inside the same state. Candidate generation
needs relative action quality, not only marginal effect likelihood.

## Generated ranking groups

Each training group contains one numeric world state and 16 legal interventions:

```text
same state + same event + candidate action -> realized numerical effect
```

The world generator is reset and replayed to the same state before every
candidate. External event and renderer-noise state are therefore shared across
the group. Only the intervention vector changes. Labels and generator metadata
remain outside the model forward contract.

Training uses prediction step 3 with four context steps. State groups advance
sequentially through train mechanism groups while renderer views cycle
deterministically. Candidate seed for state ordinal `n` is
`260954 + n * 1000003 + 101`.

## Critic

The H015 factual-residual backbone is retrained under seed `260954`, then frozen.
A small ranking head maps each candidate-conditioned transition state to one
scalar logit. The head is trained for 600 steps on two state groups per step.

For each group, realized effects are standardized within state. A ListNet loss
matches their soft ordering at temperature `0.50`; a `0.50`-weighted signed
pairwise logistic loss reinforces pairs separated by at least `1e-5` realized
effect. Search consumes rank logits only. It receives no effect labels.

## Equal-budget gate

The rank critic, H015 pointwise mean-only control and legal random arm each
execute eight candidates per validation state. Both learned arms receive
exactly 256 scores through the unchanged H015 quality-diversity search. A fixed
256-action oracle pool is evaluator-only.

The ranking critic must reduce mean realized regret to at most:

- `0.75` of legal random;
- `0.85` of the H015 pointwise control.

After candidate selection finishes, the evaluator may score the fixed oracle
pool once per learned arm to report within-state Spearman and NDCG@8. These
diagnostic scores are separate from the 256-score search budget and cannot
change selected candidates.

Training-candidate replay, search replay, dataset replay, legality, exact
budgets, finite metrics and zero leakage are hard guards. Frozen validation
seeds and every model test split stay sealed until the development gate passes.

## Claim boundary

A passing result demonstrates action ordering only in registered generated
worlds. It does not establish real-world causality, business value, creativity,
language-independent thought or production readiness.
