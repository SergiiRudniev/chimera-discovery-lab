# Meta-World Configurations

This directory contains isolated Chimera Meta-World configurations. The first
registered executable contract is `meta_world_w0.yaml`; its parameter count is
derived from code and its H000 qualification gates are frozen before execution.

Corrective trial configs are append-only. `meta_world_w0_t1.yaml` preserves the
H001 retry after H000 exposed a BF16 domain-selection failure.

`world_generators_h002.yaml` freezes the generated-world tensor contract,
split boundaries, comparison arms and acceptance thresholds before H002 target
metrics are opened.

`world_h011_development_consistency.yaml` and its matched control freeze the
direct paired-response objective before H011 development execution.

No configuration in this directory may reuse Venture graph semantics implicitly.
