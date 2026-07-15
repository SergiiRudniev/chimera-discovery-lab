# Model Registry

Names and namespaces are reserved before experiments begin.

| Family | Branch | Prefix | State |
| --- | --- | --- | --- |
| Chimera Venture | `chimera-venture` | `CHM-V` | Active |
| Chimera Meta-World | `chimera-meta-world` | `CHM-W` | Registered design |
| Chimera Catalyst | `chimera-catalyst` | `CHM-C` | Reserved |
| Chimera Oracle | `chimera-oracle` | `CHM-O` | Reserved |
| Chimera Architect | `chimera-architect` | `CHM-A` | Reserved |
| Chimera Nexus | `chimera-nexus` | `CHM-N` | Reserved |
| Chimera Frontier | `chimera-frontier` | `CHM-F` | Reserved |

Reserved names are not reassigned to a different domain. A family begins at
`H000`; IDs are append-only and never reused after a failed or cancelled run.

Generation symbols are family-specific and form part of the public model name:

| Family | Generation form | Meaning | First generation |
| --- | --- | --- | --- |
| Chimera Venture | `M{generation}` | Morph | `Chimera Venture M0` |
| Chimera Meta-World | `W{generation}` | World | `Chimera Meta-World W0` |

Meta-World artifacts use these reserved forms:

- hypotheses: `CHM-W-H###`;
- trials: `CHM-W-T###`;
- corpora: `CHM-W-C###`;
- configs: `configs/meta_world/`;
- checkpoints: `chimera-meta-world-w0-step######.pt`.

The release tag `meta-world-w0` remains unavailable until W0 passes its
registered qualification protocol. Checkpoint files use lowercase,
repository-safe names.
