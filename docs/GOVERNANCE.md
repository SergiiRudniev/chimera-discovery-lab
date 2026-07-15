# Repository Governance

`main` and every `chimera-*` family branch use the versioned ruleset in
`.github/rulesets/protected-research-branches.json`.

The ruleset blocks deletion and force pushes, requires linear history, accepts
only squash merges, requires resolved review threads and waits for both Python
CI jobs. Repository administrators retain an emergency bypass so a broken
workflow cannot permanently lock the research record.

Model-family branches are long-lived namespaces. Feature branches are deleted
after merge; family branches are never repurposed or renamed.
