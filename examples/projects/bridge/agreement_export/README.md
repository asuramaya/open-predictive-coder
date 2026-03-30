# Agreement Export

This example is a bridge-style consumer that focuses on agreement between two locally generated probability streams.

It is intentionally separate from the other bridge examples:

- `proxy_features` compares a causal stream and a lagged proxy stream
- `feature_export` packages paired source/proxy streams into a compact export report
- `agreement_export` measures agreement, consensus ratio, and disagreement across two local streams

Kernel pieces reused here:

- [`bridge_feature_arrays`](../../../../src/open_predictive_coder/bridge_features.py)

Project-local policy:

- how the two streams are generated
- how agreement is summarized
- how probe and smoke output is formatted

Run from the repository root:

```bash
PYTHONPATH=src python3 examples/projects/bridge/agreement_export/probe.py
PYTHONPATH=src python3 examples/projects/bridge/agreement_export/smoke.py
```
