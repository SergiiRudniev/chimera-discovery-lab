# Reproducibility

Every experiment uses a committed YAML configuration and an immutable ID. Seeds
cover Python, NumPy and PyTorch. Dataset and split hashes are required before a
real result can move beyond `not_run`.

The synthetic generator is deterministic for a fixed configuration and seed. It
exists only to validate tensor contracts, forward passes, gradients and
checkpoint wiring.

Corpus C0 is rebuilt from `source_graphs.yaml` with seed 1701. The manifest
records the annotation hash, every shard hash, source-isolated case IDs and
tensor capacities. `chimera validate-corpus` reconstructs every canonical target
from its corrupted graph and edit program before accepting the corpus.

The public repository does not commit checkpoints, source filing text or run directories.
Trial T0 publishes its inference checkpoint as a release asset and commits a
SHA-256 manifest containing the source code commit, configuration hash, dataset
hash, environment, parameter count and evaluation outputs. A checkpoint can be
published for audit without being accepted as a qualified model.
