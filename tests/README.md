# Tests

The test tree is organized by purpose, not by historical order.

## Kernel Primitive Tests

These validate reusable mechanisms in `src/`.

Examples:

- `test_reservoir.py`
- `test_delay.py`
- `test_hierarchical.py`
- `test_linear_memory.py`
- `test_oscillatory_memory.py`
- `test_ngram_memory.py`
- `test_bridge_export.py`
- `test_bridge_features.py`
- `test_bidirectional_context.py`
- `test_learned_segmentation.py`
- `test_patch_latent_blocks.py`
- `test_causal_adapter.py`
- `test_control.py`
- `test_gating.py`
- `test_routing.py`
- `test_predictive_surprise.py`
- `test_modulation.py`
- `test_sampled_readout.py`
- `test_artifacts.py`

## Runtime And Evaluation Tests

These validate scoring, reporting, and runtime surfaces.

Examples:

- `test_eval.py`
- `test_train_eval.py`
- `test_train_modes.py`
- `test_runtime_knobs.py`

## Boundary Tests

These check the split between kernel and project policy.

Examples:

- `test_kernel_project_split.py`
- `test_boundary_surfaces.py`
- `test_bridge_export_isolation.py`
- `test_example_projects.py`

## Project Descendant Tests

These validate the example descendants directly.

Examples:

- `test_hierarchical_predictive_example.py`
- `test_bridge_proxy_example.py`
- `test_feature_export_example.py`
- `test_bidirectional_analysis_example.py`
- `test_patch_latent_example.py`
- `test_statistical_memory_example.py`
- `test_diagnostics_examples.py`

## Practical Rule

When adding a new feature:

- if it lands in `src/`, add a kernel primitive test
- if it changes runtime behavior, add a runtime test
- if it changes the kernel/project boundary, add or update a boundary test
- if it is project-local, test it in the corresponding descendant suite
