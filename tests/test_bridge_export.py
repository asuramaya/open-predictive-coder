from __future__ import annotations

import unittest

import numpy as np

from open_predictive_coder.artifacts import ArtifactAccounting
from open_predictive_coder.bridge_export import (
    BridgeExportAdapter,
    BridgeExportConfig,
    BridgeExportFitReport,
    BridgeExportReport,
)
from open_predictive_coder.bridge_features import BridgeFeatureConfig, bridge_feature_arrays
from open_predictive_coder.metrics import bits_per_byte_from_probabilities


class BridgeExportTests(unittest.TestCase):
    def setUp(self) -> None:
        self.base = np.asarray(
            [
                [0.70, 0.30],
                [0.20, 0.80],
                [0.60, 0.40],
            ],
            dtype=np.float64,
        )
        self.proxy = np.asarray(
            [
                [0.55, 0.45],
                [0.15, 0.85],
                [0.65, 0.35],
            ],
            dtype=np.float64,
        )
        self.targets = np.asarray([0, 1, 0], dtype=np.int64)

    def test_export_uses_bridge_feature_helper(self) -> None:
        adapter = BridgeExportAdapter(
            BridgeExportConfig(vocabulary_size=2, candidate_count=2, replay_threshold=0.0)
        )
        report = adapter.export(self.base, self.proxy, targets=self.targets)
        expected = bridge_feature_arrays(
            self.base,
            self.proxy,
            2,
            config=BridgeFeatureConfig(candidate_count=2),
        )

        self.assertIsInstance(report, BridgeExportReport)
        self.assertEqual(report.tokens, 3)
        self.assertEqual(report.source_names, ("base", "proxy"))
        self.assertEqual(report.features.entropy.shape, self.base.shape[:-1])
        np.testing.assert_allclose(report.features.entropy, expected.entropy)
        np.testing.assert_allclose(report.features.peak, expected.peak)
        np.testing.assert_allclose(report.features.candidate4, expected.candidate4)
        np.testing.assert_allclose(report.features.agreement, expected.agreement)
        np.testing.assert_allclose(report.features.agreement_mass, expected.agreement_mass)
        self.assertIsInstance(report.accounting, ArtifactAccounting)
        self.assertEqual(report.accounting.artifact_name, "bridge_export")
        self.assertGreaterEqual(report.mean_entropy, 0.0)
        self.assertLessEqual(report.mean_peak, 1.0)
        self.assertGreaterEqual(report.mean_agreement, 0.0)

    def test_export_with_targets_produces_probability_scores(self) -> None:
        adapter = BridgeExportAdapter(BridgeExportConfig(vocabulary_size=2, candidate_count=1))
        report = adapter.score(self.base, self.proxy, targets=self.targets, source_names=("oracle", "bridge"))

        self.assertIsInstance(report, BridgeExportReport)
        self.assertAlmostEqual(
            report.base_bits_per_byte,
            bits_per_byte_from_probabilities(self.base, self.targets),
            places=12,
        )
        self.assertAlmostEqual(
            report.proxy_bits_per_byte,
            bits_per_byte_from_probabilities(self.proxy, self.targets),
            places=12,
        )
        self.assertAlmostEqual(
            report.mean_bits_per_byte,
            0.5 * (report.base_bits_per_byte + report.proxy_bits_per_byte),
            places=12,
        )
        self.assertEqual(report.source_names, ("oracle", "bridge"))
        self.assertGreaterEqual(report.bits_per_byte, 0.0)
        self.assertGreaterEqual(report.accounting.replay_span_count, 1)

    def test_fit_aliases_score_and_accumulates_accounting(self) -> None:
        adapter = BridgeExportAdapter()
        fit_report = adapter.fit(self.base, self.proxy, targets=self.targets)

        self.assertIsInstance(fit_report, BridgeExportFitReport)
        self.assertEqual(fit_report.sequences, 1)
        self.assertEqual(fit_report.tokens, 3)
        self.assertEqual(fit_report.accounting, adapter.accounting())
        self.assertAlmostEqual(fit_report.bits_per_byte, fit_report.report.bits_per_byte, places=12)

    def test_shape_validation_is_explicit(self) -> None:
        adapter = BridgeExportAdapter(BridgeExportConfig(vocabulary_size=2))
        with self.assertRaises(ValueError):
            adapter.export(self.base, self.proxy[:, :1], targets=self.targets)
        with self.assertRaises(ValueError):
            adapter.export(self.base, self.proxy, targets=np.asarray([0, 1], dtype=np.int64))


if __name__ == "__main__":
    unittest.main()
