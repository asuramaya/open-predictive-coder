# Statistical Memory

This example stays in the project layer and composes two existing kernel primitives:

- `NgramMemory` for smoothed unigram, bigram, and trigram statistics
- `ExactContextMemory` for exact-context repair

The policy here is intentionally local. The example fits both memories on the same corpus, uses n-gram probabilities as the base distribution, and uses exact-context predictions as repair experts through `SupportWeightedMixer`.

Files:

- `model.py` for the composition logic
- `probe.py` for a quick inspection run
- `smoke.py` for a small end-to-end check
