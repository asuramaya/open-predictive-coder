# Examples

This directory is split into three roles.

## 1. Quickstart

- [`quickstart.py`](./quickstart.py)

This is the smallest end-to-end package example. It is for basic install and import smoke, not for boundary testing.

## 2. Project Descendants

These are under [`projects/`](./projects). They are the main way the repo tests whether the kernel boundary is drawn
honestly.

They are self-contained descendants that live in this repo. They are not imports from sibling repositories.

- `hierarchical_predictive`
  ancestor-style hierarchical predictive example in `projects/ancestor/`
- `exact_context_repair`
  early exact-context causal example in `projects/causal/`
- `memory_stability`, `linear_correction`, `residual_repair`
  three causal composition variants in `projects/causal/`
- `bidirectional_analysis`
  analysis-only descendant in `projects/oracle/`
- `patch_latent`
  byte-patch latent descendant in `projects/byte_latent/`

Read the project README before reading the code. Each one explains which parts are kernel reuse and which parts are
still project policy.

## 3. Development Tools

These are under [`tools/`](./tools).

Right now the main package is:

- [`tools/diagnostics/`](./tools/diagnostics)

That tooling exists to replace ad hoc analysis scripts with reusable example-facing helpers while keeping them outside
`src/`.

## Recommended Reading Order

If you want to understand the repo through examples:

1. `quickstart.py`
2. `projects/ancestor/hierarchical_predictive`
3. `projects/causal/exact_context_repair`
4. `projects/causal/memory_stability` through `causal/residual_repair`
5. `projects/oracle/bidirectional_analysis`
6. `projects/byte_latent/patch_latent`
