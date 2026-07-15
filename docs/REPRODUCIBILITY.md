# Reproducibility

Every experiment uses a committed YAML configuration and an immutable ID. Seeds
cover Python, NumPy and PyTorch. Dataset and split hashes are required before a
real result can move beyond `not_run`.

The synthetic generator is deterministic for a fixed configuration and seed. It
exists only to validate tensor contracts, forward passes, gradients and
checkpoint wiring.

Meta-World Corpus C0 stores compact `int64` indices rather than materialized
trajectories. Each row has a unique seed, so materialization is independent of
batch order. Its manifest binds the exact W0 tensor contract, generator source
hashes, source revision, split policy and every shard hash. The validation command
replays a stratified sample and verifies file integrity, split isolation,
combination balance, effect invariants and parameter sensitivity:

```powershell
.\.venv\Scripts\python.exe -m chimera.cli validate-meta-world-corpus
```

Corpus C0 is rebuilt from `source_graphs.yaml` with seed 1701. The manifest
records the annotation hash, every shard hash, source-isolated case IDs and
tensor capacities. `chimera validate-corpus` reconstructs every canonical target
from its corrupted graph and edit program before accepting the corpus.

The public repository does not commit checkpoints, source filing text or run directories.
Trials T0 and T1 publish inference checkpoints as release assets and commit
SHA-256 manifests containing the source code commit, configuration hash, dataset
hash, environment, parameter count and evaluation outputs. T2 does not create a
new checkpoint: it publishes a policy bundle bound to the immutable T1 checkpoint
SHA-256. A checkpoint or policy can be published for audit without qualifying
creativity or commercial use.
