# Statistical Memory

This example stays in the project layer and composes existing kernel primitives directly:

- `NgramMemory` for smoothed unigram, bigram, and trigram statistics
- `ExactContextMemory` for exact-context repair
- `SupportWeightedMixer` for the local blend policy

The policy here is intentionally local. The example fits both memories on the same corpus and uses the kernel
memory outputs directly without inventing a new shared abstraction.

Files:

- `model.py` for the composition logic
- `probe.py` for a quick inspection run
- `smoke.py` for a small end-to-end check
