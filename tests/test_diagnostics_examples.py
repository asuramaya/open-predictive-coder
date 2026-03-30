from __future__ import annotations

import unittest

from examples.tools.diagnostics import (
    diagnose_exact_context_repair,
    diagnose_hierarchical_predictive,
    diagnose_linear_correction,
    diagnose_memory_stability,
    diagnose_residual_repair,
    format_example_diagnostics,
)


class DiagnosticsExampleTests(unittest.TestCase):
    def test_hierarchical_predictive_report_includes_trace_and_series(self) -> None:
        report = diagnose_hierarchical_predictive(
            corpus=(
                "hierarchical prediction keeps fast, mid, and slow state aligned.\n"
                "the sampled readout should stay text-first and diagnostic.\n"
            )
            * 2
        )

        self.assertEqual(report.project, "hierarchical_predictive")
        self.assertIsNotNone(report.snapshot)
        self.assertIsNotNone(report.series)
        self.assertIn("features", report.snapshot.signal_names())
        self.assertIn("gates", report.snapshot.signal_names())
        self.assertIn("fit vs score", format_example_diagnostics(report))
        self.assertIn("hierarchical_trace", format_example_diagnostics(report))

    def test_exact_context_repair_report_includes_ablation_summary(self) -> None:
        report = diagnose_exact_context_repair(
            corpus=(
                "exact history should help when the local suffix is stable.\n"
                "support should matter more than lore and less than evidence.\n"
            )
            * 2
        )

        self.assertEqual(report.project, "exact_context_repair")
        self.assertTrue(report.ablations)
        self.assertIn("base vs mixed", format_example_diagnostics(report))
        self.assertIn("exact_context_repair", format_example_diagnostics(report))

    def test_causal_variant_helpers_are_supported(self) -> None:
        variant_1 = diagnose_memory_stability(
            corpus=(
                "memory should beat stability when the suffix is narrow.\n"
                "stability should win when the substrate is already clean.\n"
            )
            * 2
        )
        variant_2 = diagnose_linear_correction(
            corpus=(
                "linear memory carries the main path while local correction stays smaller.\n"
                "the correction expert should only matter when the base path misses detail.\n"
            )
            * 2
        )
        variant_3 = diagnose_residual_repair(
            corpus=(
                "local residual repair should stay narrow and selective.\n"
                "the base path should remain responsible for most of the distribution.\n"
            )
            * 2
        )

        self.assertEqual(variant_1.project, "memory_stability")
        self.assertEqual(variant_2.project, "linear_correction")
        self.assertEqual(variant_3.project, "residual_repair")
        self.assertIn("causal_variant", format_example_diagnostics(variant_1))
        self.assertIn("causal_variant", format_example_diagnostics(variant_2))
        self.assertIn("causal_variant", format_example_diagnostics(variant_3))
        self.assertTrue(variant_1.ablations)
        self.assertTrue(variant_2.ablations)
        self.assertTrue(variant_3.ablations)


if __name__ == "__main__":
    unittest.main()
