# Meta-World Trial T001

`CHM-W-T001` is the preregistered corrective run after T000 failed before its
first complete BF16 forward pass. It changes only domain-output selection:
adapter outputs retain their autocast dtype and are selected with `torch.where`.

Architecture dimensions, fixed batch, seed, optimizer, steps and qualification
thresholds are unchanged from T000.

## Result

The full 61,854,120-parameter W0 core completed 20 BF16 CUDA steps in 1.37
seconds with 1.90 GB peak allocated VRAM. Total fixed-batch loss moved from
0.110891 to -1.892130, all recorded metrics were finite and deterministic
evaluation replay had an exact maximum delta of 0 before and after training.

Status: `accepted` for engineering qualification. No checkpoint is promoted:
this trial measures fixed-batch wiring, not held-out dynamics, causal transfer
or production generation quality.
