# Reproducibility

Every experiment uses a committed YAML configuration and an immutable ID. Seeds
cover Python, NumPy and PyTorch. Dataset and split hashes are required before a
real result can move beyond `not_run`.

The synthetic generator is deterministic for a fixed configuration and seed. It
exists only to validate tensor contracts, forward passes, gradients and
checkpoint wiring.

The public repository does not commit checkpoints, raw data or run directories.
Accepted releases must publish a manifest containing code commit, configuration
hash, dataset hash, environment, parameter count and evaluation outputs.
