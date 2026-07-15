# Meta-World Trial T000

`CHM-W-T000` is the first full-architecture engineering qualification for
Chimera Meta-World W0. It uses one deterministic mechanistic batch to test CUDA
execution, finite optimization, exact evaluation replay and the registered
parameter envelope.

The fixture contains four known numerical dynamics, all eight intervention
codes and four domain-specific observation transforms. It is not a corpus of
real-world events and cannot support a causal-discovery or idea-quality claim.

## Result

The BF16 CUDA run failed during the first domain-adapter selection, before a
complete forward pass or optimizer step. The selection buffer was FP32 while
the adapter output under autocast was BF16, and indexed assignment requires an
exact dtype match.

Status: `rejected`. No target loss metric or checkpoint was produced. The fix
must be preregistered under a new hypothesis and trial ID.
