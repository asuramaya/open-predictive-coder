# proxy_features

This folder is a small bridge-style example that turns a causal probability stream and a lagged proxy stream into
shared bridge features.

It is intentionally project-local. The shared kernel primitive is the bridge transform in
[`bridge_feature_arrays`](../../../../src/decepticons/bridge_features.py).

Kernel primitives used here:

- `bridge_feature_arrays`
- `BridgeFeatureConfig`

Project policy that stays local:

- the causal/proxy probability source
- the sample corpus
- output formatting for probe and smoke flows

## Run

From the repository root:

```bash
PYTHONPATH=src python3 examples/projects/bridge/proxy_features/probe.py
PYTHONPATH=src python3 examples/projects/bridge/proxy_features/smoke.py
```

## What It Shows

- bridge features can be computed directly from probability arrays
- a proxy stream can be compared to a causal stream without introducing a new kernel abstraction
- the shared transform remains reusable while the example stays small
