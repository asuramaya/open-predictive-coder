# Next Pass

This document turns the current state into the next concrete implementation pass.

## Goal

Harden the shared `src/`-level causal, oracle, and bridge adapters, then use that cleaner family surface to extract
the first real `noncausal_reconstructive` contract.

## Why This Is The Next Pass

The repo now has:

- enough substrate families
- enough controller primitives
- enough memory and view surfaces
- enough runtime scaffolding
- enough project descendants to test the boundary
- a first `src/`-level `causal_predictive` adapter
- a first `src/`-level `oracle_analysis` adapter
- a first `src/`-level `bridge_export` adapter
- a first learned patch-latent kernel slice for segmentation, local encoding, pooling, and bridging
- the next statistical/kernel slice for oscillatory memory, n-gram memory, bridge features, and bidirectional context

That changes the problem.

The next step is not to invent another abstract substrate mechanism immediately. The next step is to pressure-test the
new shared adapters and learned patch-latent blocks across descendants and keep the kernel readable while doing it.

That now also applies to the new statistical/kernel additions: they need consumers before they justify wider
abstraction or backend work.

## Workstreams

### 1. Thin causal descendants around the causal contract

Make the causal descendants consume the shared causal layer wherever the behavior is actually common.

What should stay local:

- mixer policy
- residual-repair policy
- legality framing
- benchmark and frontier claims

Acceptance criteria:

- less duplicate causal boilerplate in `examples/projects/causal_*`
- descendant policy still clearly lives in project code

### 2. Harden causal runtime and accounting

The causal adapter now exists, so the next reusable layer is better reporting and accounting around it.

Focus on:

- replay-safe accounting
- metadata tagging
- lightweight causal report wrappers
- evaluation hooks that do not import descendant policy

Acceptance criteria:

- causal reports are useful outside one example
- artifact/replay helpers stay policy-free

### 3. Add a second consumer of the shared family layer

The shared adapters are not stable until something besides the current causal and oracle examples pushes on them.

Best candidates:

- a second bridge/export-shaped descendant beyond the first export/report example
- a noncausal reconstructive descendant with replay and side-data accounting

Acceptance criteria:

- at least one more descendant reuses some of the same causal contract
- any new promotion into `src/` is justified by repeated use

### 4. Keep refining `hierarchical_predictive`, but only in project space

The ancestor example remains a boundary test, not a reason to widen `src/` recklessly.

Likely next work there:

- stronger predictor behavior
- more faithful routed/modulated variants
- better checkpoint reporting

Acceptance criteria:

- ancestor-specific policy stays outside the kernel unless it repeats elsewhere

### 5. Keep tightening repo readability

Every pass should also improve orientation:

- architecture docs stay current
- examples index stays current
- tests remain grouped by purpose
- root exports stay legible by category

## Recommended Order

1. thin causal and oracle examples around the shared adapters
2. harden runtime and accounting
3. add the first noncausal reconstructive consumer
4. only then consider wider family abstractions

## Non-Goals For The Next Pass

These should wait:

- full optimizer/training harness extraction
- full legality framework
- noncausal replay economics in `src/`
- patch-latent rate-distortion, QAT, or second-stage downsampling in `src/` before a second consumer asks for them
- preset stabilization

## Definition Of Done

The pass is done when:

- the shared causal, oracle, and bridge adapters clearly read as `src/` contracts
- the corresponding examples look thinner around those contracts
- at least one more descendant pushes on one of the same shared surfaces
- docs explain the boundary without requiring project history to decode the codebase
