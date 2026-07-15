# Research Journal

## 2026-07-15 — Venture V0.1 foundation

### CHM-V-H000

- **Question:** Does the full graph-to-edit training path run with finite
  gradients and reduce loss on a fixed synthetic transition batch?
- **Registration:** Complete before execution.
- **Result:** Loss decreased from 7.1843 to 1.0263 in 20 fixed-batch steps;
  every recorded loss and gradient norm was finite.
- **Decision:** Accepted as engineering validation.
- **Claim boundary:** A passing result validates wiring only, not creativity.

### CHM-V-H001

- **Question:** Does non-linguistic graph generation improve blind-rated novelty
  at matched feasibility versus a text baseline?
- **Registration:** Complete.
- **Result:** `not_run`; no real dataset or checkpoint exists.
- **Decision:** Pending.

### CHM-V-H002

- **Question:** Does MAP-Elites improve structural diversity without reducing
  median feasibility?
- **Registration:** Complete.
- **Result:** `not_run`.
- **Decision:** Deferred until a frozen CHM-V-H001 policy exists.

### CHM-V-H003

- **Question:** Does latent next-state prediction improve held-out coherence?
- **Registration:** Complete.
- **Result:** `not_run`.
- **Decision:** Deferred until real transition data is split and frozen.
