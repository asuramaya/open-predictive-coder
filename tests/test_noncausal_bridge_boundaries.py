from __future__ import annotations

from dataclasses import fields
import inspect
import unittest

import open_predictive_coder as opc


def _field_names(cls: type[object]) -> set[str]:
    return {field.name for field in fields(cls)}


class NoncausalBridgeBoundaryTests(unittest.TestCase):
    def test_public_kernel_does_not_prematurely_export_noncausal_policy(self) -> None:
        public_names = set(dir(opc))

        self.assertFalse(any(name.startswith("Noncausal") for name in public_names))
        self.assertFalse(hasattr(opc, "NoncausalReconstructiveAdapter"))
        self.assertFalse(hasattr(opc, "NoncausalFieldReconstructionModel"))

    def test_causal_oracle_and_bridge_constructors_stay_disjoint(self) -> None:
        causal_params = tuple(inspect.signature(opc.CausalPredictiveAdapter.__init__).parameters)
        oracle_params = tuple(inspect.signature(opc.OracleAnalysisAdapter.__init__).parameters)
        bridge_params = tuple(inspect.signature(opc.BridgeExportAdapter.__init__).parameters)

        self.assertNotIn("bridge", causal_params)
        self.assertNotIn("bridge_features", causal_params)
        self.assertNotIn("right_context", causal_params)
        self.assertNotIn("noncausal", causal_params)

        self.assertNotIn("bridge", oracle_params)
        self.assertNotIn("bridge_features", oracle_params)
        self.assertNotIn("exact_context", oracle_params)
        self.assertNotIn("noncausal", oracle_params)

        self.assertNotIn("exact_context", bridge_params)
        self.assertNotIn("experts", bridge_params)
        self.assertNotIn("bidirectional", bridge_params)
        self.assertNotIn("noncausal", bridge_params)

    def test_surface_reports_do_not_cross_the_family_boundary(self) -> None:
        causal_fit_fields = _field_names(opc.CausalPredictiveFitReport)
        causal_score_fields = _field_names(opc.CausalPredictiveScore)
        oracle_report_fields = _field_names(opc.OracleAnalysisReport)
        bridge_report_fields = _field_names(opc.BridgeExportReport)

        causal_forbidden = {
            "bridge_features",
            "bridge_export",
            "bidirectional_context",
            "left_context_count",
            "right_context_count",
            "pair_context_count",
            "neighborhoods",
        }
        oracle_forbidden = {
            "bridge_features",
            "bridge_export",
            "candidate4",
            "candidate_count",
            "exact_support",
            "ngram_bits_per_byte",
        }
        bridge_forbidden = {
            "exact_support",
            "left_context_count",
            "right_context_count",
            "pair_context_count",
            "neighborhoods",
            "ngram_bits_per_byte",
        }

        self.assertFalse(causal_fit_fields & causal_forbidden)
        self.assertFalse(causal_score_fields & causal_forbidden)
        self.assertFalse(oracle_report_fields & oracle_forbidden)
        self.assertFalse(bridge_report_fields & bridge_forbidden)

        self.assertIn("features", bridge_report_fields)
        self.assertIn("accounting", bridge_report_fields)
        self.assertIn("mean_candidate4", bridge_report_fields)
        self.assertIn("mean_agreement_mass", bridge_report_fields)
        self.assertIn("oracle_preference_rate", oracle_report_fields)
        self.assertIn("points", oracle_report_fields)


if __name__ == "__main__":
    unittest.main()
