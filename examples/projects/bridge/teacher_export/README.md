# Teacher Export

This example is a bridge descendant for export labels and attack-aware reporting.

It stays project-local on purpose:

- it synthesizes teacher and student probability streams locally
- it exports teacher labels through the shared bridge adapter
- it measures attack drift with a local token mutation pass and bidirectional-context probing
- it does not widen `src/` with descendant-specific policy

Kernel pieces reused here:

- [`BridgeExportAdapter`](/Users/asuramaya/Code/carving_machine_v3/open-predictive-coder/src/open_predictive_coder/bridge_export.py)
- [`bridge_feature_arrays`](/Users/asuramaya/Code/carving_machine_v3/open-predictive-coder/src/open_predictive_coder/bridge_features.py)
- [`BidirectionalContextProbe`](/Users/asuramaya/Code/carving_machine_v3/open-predictive-coder/src/open_predictive_coder/bidirectional_context.py)

Entry points:

- [`probe.py`](/Users/asuramaya/Code/carving_machine_v3/open-predictive-coder/examples/projects/bridge/teacher_export/probe.py)
- [`smoke.py`](/Users/asuramaya/Code/carving_machine_v3/open-predictive-coder/examples/projects/bridge/teacher_export/smoke.py)

Run from the repo root:

```bash
PYTHONPATH=src python3 examples/projects/bridge/teacher_export/probe.py
PYTHONPATH=src python3 examples/projects/bridge/teacher_export/smoke.py
```
