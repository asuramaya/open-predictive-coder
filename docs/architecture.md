# Architecture

This document is the shortest route to understanding the repo as code rather than as a historical story.

## Reading Order

If you are new to the repo, read things in this order:

1. [`README.md`](../README.md)
2. [`kernel_matrix.md`](./kernel_matrix.md)
3. this file
4. [`examples/README.md`](../examples/README.md)
5. [`lineage.md`](./lineage.md)
6. the specific example project README you care about

Use [`related_work.md`](./related_work.md) and [`landscape.md`](./landscape.md) for research and ecosystem context, not
for code orientation.

## Layer Model

The repo is intentionally split into three layers.

### 1. Kernel

The `src/open_predictive_coder/` package contains reusable mechanisms only.

It owns:

- substrate dynamics
- controller-side summaries, gates, routing, and modulation
- memory and latent primitives
- feature views and sampled readout
- readouts, experts, and scoring utilities
- runtime surfaces like traces, eval, train modes, and artifact accounting

It does not own:

- project-specific legality rules
- benchmark claims
- one descendant's routing policy
- one descendant's latent composition policy
- one descendant's oracle privileges

### 2. Project-Layer Descendants

The `examples/projects/` tree is where the kernel gets pressure-tested by actual descendant shapes.

These are not toy demos. They are boundary tests.

- `ancestor/hierarchical_predictive/`
  ancestor-style hierarchical substrate plus predictor/gating policy
- `causal/exact_context_repair/`
  early exact-context causal memory shape
- `causal/memory_stability/`, `causal/linear_correction/`, `causal/residual_repair/`
  three different causal composition policies built from kernel primitives
- `oracle/bidirectional_analysis/`
  analysis-only descendant that reuses sampled readout, routing, and train-mode checkpoints
- `byte_latent/patch_latent/`
  byte-patch latent descendant shaped as a general patch-latent example

If a mechanism is repeated across multiple descendants, it is a candidate for promotion into `src/`.
The first causal adapter is the next example of that rule: it should live in `src/` only when it reads as a shared
contract rather than a descendant-specific model.

### 3. Development Tooling

The `examples/tools/` tree is for development and analysis support, not kernel code.

Right now that mainly means:

- `examples/tools/diagnostics/`

This keeps `look.py` / `look2.py` / `silence_test.py` style workflows available without polluting the public package.

## Package Map

The kernel is easiest to understand by category rather than by filename order.

### Foundation

- [`codecs.py`](../src/open_predictive_coder/codecs.py)
- [`config.py`](../src/open_predictive_coder/config.py)
- [`metrics.py`](../src/open_predictive_coder/metrics.py)

### Substrates

- [`reservoir.py`](../src/open_predictive_coder/reservoir.py)
- [`delay.py`](../src/open_predictive_coder/delay.py)
- [`linear_memory.py`](../src/open_predictive_coder/linear_memory.py)
- [`mixed_memory.py`](../src/open_predictive_coder/mixed_memory.py)
- [`hierarchical.py`](../src/open_predictive_coder/hierarchical.py)
- [`substrates.py`](../src/open_predictive_coder/substrates.py)
- [`factories.py`](../src/open_predictive_coder/factories.py)

### Control And Side Channels

- [`control.py`](../src/open_predictive_coder/control.py)
- [`gating.py`](../src/open_predictive_coder/gating.py)
- [`routing.py`](../src/open_predictive_coder/routing.py)
- [`modulation.py`](../src/open_predictive_coder/modulation.py)
- [`predictive_surprise.py`](../src/open_predictive_coder/predictive_surprise.py)

### Memory, Latents, And Views

- [`exact_context.py`](../src/open_predictive_coder/exact_context.py)
- [`latents.py`](../src/open_predictive_coder/latents.py)
- [`segmenters.py`](../src/open_predictive_coder/segmenters.py)
- [`views.py`](../src/open_predictive_coder/views.py)
- [`linear_views.py`](../src/open_predictive_coder/linear_views.py)
- [`hierarchical_views.py`](../src/open_predictive_coder/hierarchical_views.py)
- [`sampled_readout.py`](../src/open_predictive_coder/sampled_readout.py)

### Readouts, Experts, And Runtime

- [`readout.py`](../src/open_predictive_coder/readout.py)
- [`readouts.py`](../src/open_predictive_coder/readouts.py)
- [`experts.py`](../src/open_predictive_coder/experts.py)
- [`runtime.py`](../src/open_predictive_coder/runtime.py)
- [`eval.py`](../src/open_predictive_coder/eval.py)
- [`train_eval.py`](../src/open_predictive_coder/train_eval.py)
- [`train_modes.py`](../src/open_predictive_coder/train_modes.py)
- [`artifacts.py`](../src/open_predictive_coder/artifacts.py)

### Adapters And Presets

- [`adapters.py`](../src/open_predictive_coder/adapters.py)
- [`causal_predictive.py`](../src/open_predictive_coder/causal_predictive.py)
- [`oracle_analysis.py`](../src/open_predictive_coder/oracle_analysis.py)
- [`model.py`](../src/open_predictive_coder/model.py)
- [`presets.py`](../src/open_predictive_coder/presets.py)
- [`cli.py`](../src/open_predictive_coder/cli.py)

## Promotion Rule

Code moves from a project into `src/` only when all of these are true:

1. it is a mechanism rather than a project policy
2. at least two descendants want the same thing
3. the generalized API is simpler than keeping the duplication in project code

That rule is the main defense against turning the kernel into a renamed collection of branches.

## Current Boundary

Stable kernel examples of the right kind of promotion:

- `LinearMemorySubstrate`
- `FrozenReadoutExpert`
- `PredictiveSurpriseController`
- `HormoneModulator`
- `SampledMultiscaleReadout`
- `TrainModeConfig`
- `ArtifactMetadata` / `ReplaySpan` / `ArtifactAccounting`

Still project-local on purpose:

- descendant mixer and residual-repair policy
- interpretation and reporting around oracle comparisons
- patch-boundary tuning and local/global bridge composition in the patch-latent example
- ancestor-specific predictor head choices

Recent shared promotion:

- `CausalPredictiveAdapter`
- `OracleAnalysisAdapter`

## Immediate Architectural Direction

The next real jump is not another single descendant. It is pressure-testing the shared causal and oracle adapters
across more than one consumer while keeping descendant policy out of `src/`.

That means:

- thinning the causal descendants around the causal adapter
- keeping the bidirectional-analysis example thin around the oracle adapter
- hardening the shared runtime/accounting contract
- starting the first truly noncausal reconstructive or bridge/export consumer

That is how the repo graduates from "first shared contracts exist" to "the contracts are actually stable enough to keep."
