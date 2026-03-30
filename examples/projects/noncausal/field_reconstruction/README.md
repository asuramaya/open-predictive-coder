# field_reconstruction

This example stays in the project layer and wraps the shared `NoncausalReconstructiveAdapter` with local
field-reconstruction policy.

Kernel pieces reused here:

- `NoncausalReconstructiveAdapter` for bidirectional reconstruction, replay spans, and accounting
- `ExactContextMemory` for forward and reverse exact-context distributions
- `BidirectionalContextProbe` for left/right determinism statistics
- `ReplaySpan` and `ArtifactAccounting` for replay-boundary accounting

Project policy that stays local:

- the corpus
- the replay threshold
- how reconstructed bytes are rendered in probe and smoke output

## Run

From the repository root:

```bash
PYTHONPATH=src python3 examples/projects/noncausal/field_reconstruction/probe.py
PYTHONPATH=src python3 examples/projects/noncausal/field_reconstruction/smoke.py
```

## What It Shows

- both directions can be composed without widening `src/`
- replay spans can be inferred from bidirectional confidence
- noncausal reconstruction can stay local while still using shared kernel primitives
