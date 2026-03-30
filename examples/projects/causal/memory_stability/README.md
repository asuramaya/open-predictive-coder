# memory_stability

This folder is a from-scratch causal memory-plus-stability replica built on top of extracted primitives.

What stays in the kernel:

- frozen substrates
- feature/readout primitives
- basic evaluation surfaces

What stays here:

- the choice of a memory path plus a stability path
- the token-wise mixer policy used to combine them
- the stability branch uses the kernel `OscillatoryMemorySubstrate` as the pressure test for oscillatory memory

That keeps the dual-path policy out of `src/` while still exercising the oscillatory substrate in a real causal
example.
