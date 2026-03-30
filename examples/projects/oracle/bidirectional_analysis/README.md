# bidirectional_analysis

This folder is a small analysis-only descendant probe.

It does not implement a causal runtime codec. Instead, it compares a forward
causal scan against a reverse oracle scan so we can test which kernel surfaces
generalize beyond the causal runtime examples.

Kernel primitives used here:

- `OracleAnalysisAdapter`
- `OracleAnalysisConfig`
- `HierarchicalSubstrate`
- `HierarchicalFeatureView`
- `SampledMultiscaleReadout`
- `SummaryRouter`
- `TrainModeConfig`

Project policy that stays local:

- example corpus choice
- output formatting for probe and smoke flows

## Run

From the repository root:

```bash
PYTHONPATH=src python3 examples/projects/oracle/bidirectional_analysis/probe.py
PYTHONPATH=src python3 examples/projects/oracle/bidirectional_analysis/smoke.py
```

## What It Shows

- sampled multiscale readout can be used as an analysis feature surface, not only as a causal model input
- train-mode checkpointing can drive analysis cadence
- diagnostics can compare forward and reverse feature traces without turning the example into a codec
