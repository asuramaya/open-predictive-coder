# payload_choice

This example rebuilds a noncausal descendant around a local choice between dense and sparse position payload layouts.

Kernel pieces reused here:

- `BidirectionalContextProbe` for noncausal neighborhood statistics
- `ReplaySpan` and `span_selection` for sparse span selection
- `ArtifactAccounting` for local payload/replay size accounting

Project policy that stays local:

- how position payloads are dictionary-encoded
- when the model prefers dense vs sparse layout
- how probe and smoke output describe the selection

Run from the repository root:

```bash
PYTHONPATH=src python3 examples/projects/noncausal/payload_choice/probe.py
PYTHONPATH=src python3 examples/projects/noncausal/payload_choice/smoke.py
```
