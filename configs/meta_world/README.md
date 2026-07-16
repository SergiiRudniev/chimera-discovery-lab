# Meta-World Configurations

This directory contains isolated Chimera Meta-World configurations. The first
registered executable contract is `meta_world_w0.yaml`; its parameter count is
derived from code and its H000 qualification gates are frozen before execution.

Corrective trial configs are append-only. `meta_world_w0_t1.yaml` preserves the
H001 retry after H000 exposed a BF16 domain-selection failure.

`world_generators_h002.yaml` freezes the generated-world tensor contract,
split boundaries, comparison arms and acceptance thresholds before H002 target
metrics are opened.

`world_h008_development_suite.yaml` freezes the six trainable outcome-head
comparison arms, legal-random baseline, reused WG1 integrity evidence and the
development gate for CHM-W-H008.

`world_h011_development_consistency.yaml` and its matched control freeze the
direct paired-response objective before H011 development execution.

`world_generators_h012.yaml` and `world_h012_suite.yaml` freeze the WG3 seeds,
five comparison arms, sealed-test policy and primary transfer gate.

`world_generators_h013.yaml` and `world_h013_suite.yaml` freeze the WG4 paired
factual/no-op transition contract and parameter-matched additive-dynamics gate.
The three `world_h013_development_*.yaml` files hold the factorized, matched
direct and factual-only development arms; the smoke config is engineering-only.

`world_h014_suite.yaml` reuses WG4 integrity evidence and freezes the
parameter-matched predicted-response versus factual-residual effect-head test.
The two `world_h014_development_*.yaml` files hold its matched arms; the smoke
config is engineering-only.

`world_h015_suite.yaml` freezes equal model-score and simulator-execution
budgets for uncertainty-aware, mean-only, random and finite-pool oracle search.

No configuration in this directory may reuse Venture graph semantics implicitly.
