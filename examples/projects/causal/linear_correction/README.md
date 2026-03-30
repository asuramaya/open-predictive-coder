# linear_correction

This folder is a from-scratch causal linear-plus-correction replica built on top of extracted primitives.

Kernel promotion in this round:

- `LinearMemorySubstrate` moved into `src/` because both the linear-correction and residual-repair descendants need the same
  frozen linear multiscale memory bank.

Project-local decisions:

- using that linear bank as the main path
- pairing it with a smaller nonlinear correction expert
- keeping the learned mixer in project code rather than calling it a general kernel primitive
- runtime rollout and slow-update knobs already live in the kernel; this folder only owns the mixture policy and expert pairing
