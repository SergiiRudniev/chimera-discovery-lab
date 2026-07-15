# Model Registry

Names and namespaces are reserved before experiments begin.

| Family | Branch | Prefix | State |
| --- | --- | --- | --- |
| Chimera Venture | `chimera-venture` | `CHM-V` | Active |
| Chimera Catalyst | `chimera-catalyst` | `CHM-C` | Reserved |
| Chimera Oracle | `chimera-oracle` | `CHM-O` | Reserved |
| Chimera Architect | `chimera-architect` | `CHM-A` | Reserved |
| Chimera Nexus | `chimera-nexus` | `CHM-N` | Reserved |
| Chimera Frontier | `chimera-frontier` | `CHM-F` | Reserved |

Reserved names are not reassigned to a different domain. A family begins at
`H000`; IDs are append-only and never reused after a failed or cancelled run.

Model generations use `{Family} M{generation}`, where `M` denotes a Morph.
Checkpoint files use lowercase repository-safe names such as
`chimera-venture-m0-step020000.pt`.
