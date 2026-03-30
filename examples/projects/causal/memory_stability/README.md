# memory_stability

This folder is a from-scratch causal memory-plus-stability replica built on top of extracted primitives.

What stays in the kernel:

- frozen substrates
- feature/readout primitives
- basic evaluation surfaces

What stays here:

- the decision to use two causal experts
- the choice of a memory path plus a stability path
- the token-wise mixer policy used to combine them
- the stability branch now uses the kernel `OscillatoryMemorySubstrate` as the pressure test for oscillatory memory
- even though the kernel now has sampled readout, this descendant does not need it; the mixer policy is still the real project-local seam

That keeps this causal dual-path policy out of `src/` until it proves useful beyond this descendant while still
exercising the oscillatory substrate in a real causal example.
