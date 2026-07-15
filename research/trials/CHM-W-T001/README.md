# Meta-World Trial T001

`CHM-W-T001` is the preregistered corrective run after T000 failed before its
first complete BF16 forward pass. It changes only domain-output selection:
adapter outputs retain their autocast dtype and are selected with `torch.where`.

Architecture dimensions, fixed batch, seed, optimizer, steps and qualification
thresholds are unchanged from T000.

Status: `not_run`.
