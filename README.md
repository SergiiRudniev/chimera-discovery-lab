<div align="center">

# Chimera Meta-World

**Cross-domain causal world models inside Chimera Discovery Lab**

[![Python](https://img.shields.io/badge/Python-3.10--3.13-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.4%2B-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![CI](https://github.com/SergiiRudniev/chimera-discovery-lab/actions/workflows/ci.yml/badge.svg?branch=chimera-meta-world)](https://github.com/SergiiRudniev/chimera-discovery-lab/actions/workflows/ci.yml?query=branch%3Achimera-meta-world)
[![Model](https://img.shields.io/badge/model-Meta--World%20W0-7B5CFA)](#chimera-meta-world-w0)
[![Parameters](https://img.shields.io/badge/parameters-65.214M-2E8B57)](#hardware-envelope)
[![Status](https://img.shields.io/badge/status-engineering%20qualified-F0B429)](#current-status)
[![License](https://img.shields.io/badge/license-Apache--2.0-4C566A)](LICENSE)

</div>

Chimera Meta-World is the causal-dynamics model family in Chimera Discovery
Lab. It is designed to learn a shared numerical representation of changing
systems, discover mechanisms across domains and propose interventions before
language grounding.

> [!IMPORTANT]
> W0 has passed a fixed-batch engineering qualification, not a model release.
> This branch contains no promoted Meta-World checkpoint and no evidence for
> causal discovery, cross-domain transfer or idea quality.

## Chimera Meta-World W0

W0 tests whether structurally related processes from different domains can be
represented in one learned state space without passing text through the model
core.

```text
Z(t) + intervention -> Z(t+1), effect, uncertainty
```

The output is a frozen numerical proposal. A separate grounder may describe or
instantiate it, but cannot modify its structure or predicted effect.

## Representation Boundary

| Inside W0 | Outside W0 |
| --- | --- |
| Numeric observations | Names and prose |
| Latent object slots | Language-model embeddings |
| Learned soft roles | Human-readable labels |
| Relations, flows and constraints | Final explanation |
| Time and interventions | Business presentation |

This boundary does not imply completely language-free thought. Dataset
selection, observation design, objectives and evaluation remain human choices.

## Architecture

```mermaid
flowchart TB
    subgraph INPUT["1. Cross-domain observations"]
        OBS["Numeric states and events"]
        ACTIONS["Actions and later outcomes"]
        TIME["Time and uncertainty"]
    end

    subgraph STATE["2. Shared world state"]
        ADAPTERS["Domain adapters"]
        SLOTS["Latent object slots"]
        RELATIONS["Soft roles, relations and flows"]
    end

    subgraph DYNAMICS["3. Causal dynamics"]
        TEMPORAL["Multi-horizon transition model"]
        COUNTERFACTUAL["Intervention-conditioned prediction"]
        EFFECT["Effect and uncertainty heads"]
    end

    subgraph DISCOVERY["4. Mechanism discovery"]
        SEARCH["Constrained intervention search"]
        TRANSFER["Cross-domain mechanism alignment"]
        CONTROL["Random and ablated controls"]
    end

    subgraph BOUNDARY["5. Grounding boundary"]
        FREEZE["Frozen numerical proposal"]
        GROUND["External audited grounder"]
    end

    OBS --> ADAPTERS
    ACTIONS --> ADAPTERS
    TIME --> ADAPTERS
    ADAPTERS --> SLOTS --> RELATIONS
    RELATIONS --> TEMPORAL --> COUNTERFACTUAL --> EFFECT
    EFFECT --> SEARCH
    RELATIONS --> TRANSFER --> SEARCH
    CONTROL --> SEARCH
    SEARCH --> FREEZE --> GROUND
```

The detailed boundary and qualification requirements are frozen in the
[W0 design contract](docs/META_WORLD_W0.md).

## Generated Worlds H002

H002 replaces static trajectory enumeration with seed-addressable mathematical
world generators. `FlowWorld`, `CompetitionWorld` and `FunnelWorld` share hidden
mechanisms while independent renderers change objects, channels, units, time,
noise and visibility. Model batches contain only numeric tensors; generator IDs
remain outside the model boundary.

The hypothesis, split isolation and acceptance rule were frozen before target
metrics. See the [generated-world contract](docs/WORLD_GENERATORS_H002.md).

## Closed-Loop Training H003

H002 validation showed that relational state improved intervention-effect
prediction over a temporal baseline, while its in-batch alignment objective did
not beat the same relational architecture without alignment. H003 therefore
trains four autoregressive steps through model-generated states and expands the
hard-negative pool with a detached cross-batch mechanism queue.

Stable mechanism fingerprints are evaluator-only labels. They pair losses and
never enter the model forward contract. The registered test splits remain
sealed. See the [H003 training contract](docs/CLOSED_LOOP_H003.md).

## Active Identification H004

H004 replaces passive random-action pretraining with controlled numerical
system-identification probes. WG1 applies zero baselines, low/high impulses,
control polarity, reversals and recovery across generated worlds, then evaluates
every arm on the same short probe prefix followed by seeded random actions.

The action policy is evaluator metadata, not a model feature. See the
[H004 probe contract](docs/WORLD_PROBES_H004.md).

## Counterfactual Outcome Head H008

H008 tests an algebraically constrained outcome head: the model predicts
factual and no-op utility internally, then emits intervention effect as their
exact difference. Its direct-head controls have the same parameter count and
receive the same data, actions, optimizer and evaluator. See the
[H008 counterfactual-head contract](docs/COUNTERFACTUAL_HEAD_H008.md).

## Numerical Output

```text
source_state
intervention_code
affected_slot_ids
intervention_parameters
predicted_next_state
predicted_effect
epistemic_uncertainty
structural_novelty
```

Interpretation is allowed only after deterministic replay succeeds and the
intervention is complete. Rendered outputs must round-trip to this frozen
proposal.

## Research Controls

The first evidence-bearing W0 comparison must include:

- W0 interventions;
- legal random interventions;
- an ablated dynamics model;
- a matched language baseline;
- structural metrics before grounding;
- blind quality metrics after grounding.

The same frozen grounder is used for every structural arm. If it adds a
mechanism, that mechanism is attributed to the grounder rather than W0.

## Git and Artifact Registry

| Item | Reserved form |
| --- | --- |
| Family branch | `chimera-meta-world` |
| Feature branches | `agent/meta-world-*` |
| Hypotheses | `CHM-W-H###` |
| Trials | `CHM-W-T###` |
| Corpora | `CHM-W-C###` |
| Configs | `configs/meta_world/` |
| Checkpoints | `chimera-meta-world-w0-step######.pt` |
| Release tag | `meta-world-w0` after qualification only |

Model-family branches are protected against deletion and force pushes. Changes
arrive through linear, squash-merged pull requests with both Python CI jobs.

## Hardware Envelope

The current relational W0 candidate contains **65,213,950 trainable parameters**
for local mixed-precision
training on an NVIDIA GeForce RTX 5070 with 12,227 MiB VRAM. Gradient
accumulation and activation checkpointing remain available if the final temporal
context exceeds the initial memory budget.

## Current Status

| Item | Status |
| --- | --- |
| Family name and namespace | Registered |
| Protected family branch | Active |
| W0 design contract | Registered |
| Numerical output boundary | Registered |
| Architecture implementation | Implemented |
| BF16 CUDA engineering gate | Accepted: H001/T001 |
| Generated-world corpus | Implemented and validated |
| H002 | Validation preflight negative; test sealed; result `not_run` |
| H003 | Exploratory validation negative; test sealed; result `not_run` |
| H004 | Preregistered; WG1 implemented and validated |
| H008 | Implemented; development gate pending |
| W0 configuration | `meta_world_w0_t1.yaml` |
| Promoted checkpoint | None |
| Empirical claims | None |

## Repository Scope

This family branch inherits shared Chimera Discovery Lab infrastructure. The
existing Venture graph model remains visible for reproducibility but is not the
Meta-World W0 implementation. W0 code will use its own modules, configurations,
datasets and research identifiers.

## Validation

```powershell
.\.venv\Scripts\python.exe -m ruff check .
.\.venv\Scripts\python.exe -m mypy src
.\.venv\Scripts\python.exe -m pytest
.\.venv\Scripts\python.exe -m chimera.cli validate-research
```

## Documentation

- [Meta-World W0 design contract](docs/META_WORLD_W0.md)
- [Generated Worlds H002](docs/WORLD_GENERATORS_H002.md)
- [Closed-Loop Training H003](docs/CLOSED_LOOP_H003.md)
- [System-Identification Probes H004](docs/WORLD_PROBES_H004.md)
- [Counterfactual Outcome Head H008](docs/COUNTERFACTUAL_HEAD_H008.md)
- [Model registry](docs/MODEL_REGISTRY.md)
- [Repository governance](docs/GOVERNANCE.md)
- [Research protocol](docs/RESEARCH_PROTOCOL.md)
- [Research journal](docs/RESEARCH_JOURNAL.md)
- [GPU setup](docs/GPU_SETUP.md)

## License

Apache License 2.0. See [LICENSE](LICENSE).
