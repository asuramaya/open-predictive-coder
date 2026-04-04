from __future__ import annotations

import unittest

import numpy as np

from decepticons.probability_diagnostics import (
    ProbabilityDiagnosticsConfig,
    normalized_entropy,
    overlap_mass,
    probability_diagnostics,
    shared_top_k_mass,
    top1_agreement,
    top1_peak,
    top2_margin,
    top_k_mass,
)


class ProbabilityDiagnosticsTests(unittest.TestCase):
    def test_shapes_and_dtypes(self) -> None:
        base = np.array(
            [
                [0.0, 0.0, 2.0, 0.0],
                [1.0, 3.0, 0.0, 0.0],
            ],
            dtype=np.float64,
        )
        proxy = np.array(
            [
                [0.0, 1.0, 1.0, 0.0],
                [0.0, 4.0, 0.0, 0.0],
            ],
            dtype=np.float64,
        )

        diagnostics = probability_diagnostics(base, proxy, config=ProbabilityDiagnosticsConfig(top_k=2))

        self.assertEqual(diagnostics.entropy.shape, (2,))
        self.assertEqual(diagnostics.peak.shape, (2,))
        self.assertEqual(diagnostics.top_k_mass.shape, (2,))
        self.assertEqual(diagnostics.overlap.shape, (2,))
        self.assertEqual(diagnostics.top1_agreement.shape, (2,))
        self.assertEqual(diagnostics.shared_top_k_mass.shape, (2,))
        self.assertEqual(diagnostics.top2_margin.shape, (2,))
        for array in diagnostics.as_dict().values():
            self.assertEqual(array.dtype, np.float64)

    def test_normalization_safe_behavior(self) -> None:
        base = np.array([[0.0, 0.0, 0.0, 0.0], [2.0, 0.0, 0.0, 0.0]], dtype=np.float64)
        proxy = np.array([[0.0, 0.0, 0.0, 0.0], [0.0, 2.0, 0.0, 0.0]], dtype=np.float64)

        diagnostics = probability_diagnostics(base, proxy)

        self.assertTrue(np.all(np.isfinite(diagnostics.entropy)))
        self.assertTrue(np.all(np.isfinite(diagnostics.peak)))
        self.assertTrue(np.all(np.isfinite(diagnostics.top_k_mass)))
        self.assertTrue(np.all(np.isfinite(diagnostics.overlap)))
        self.assertTrue(np.all(np.isfinite(diagnostics.shared_top_k_mass)))
        self.assertTrue(np.all(np.isfinite(diagnostics.top2_margin)))
        self.assertTrue(np.all((diagnostics.top1_agreement == 0.0) | (diagnostics.top1_agreement == 1.0)))
        self.assertTrue(np.allclose(normalized_entropy(base), np.array([1.0, 0.0], dtype=np.float64)))
        self.assertTrue(np.allclose(diagnostics.peak, np.array([0.25, 1.0], dtype=np.float64)))

    def test_golden_values(self) -> None:
        base = np.array(
            [
                [4.0, 0.0, 0.0, 0.0],
                [3.0, 1.0, 0.0, 0.0],
            ],
            dtype=np.float64,
        )
        proxy = np.array(
            [
                [0.0, 4.0, 0.0, 0.0],
                [1.0, 3.0, 0.0, 0.0],
            ],
            dtype=np.float64,
        )

        diagnostics = probability_diagnostics(base, proxy, config=ProbabilityDiagnosticsConfig(top_k=2))

        expected_entropy_row2 = -(
            0.75 * np.log(0.75) + 0.25 * np.log(0.25)
        ) / np.log(4.0)

        self.assertTrue(np.allclose(normalized_entropy(base), np.array([0.0, expected_entropy_row2], dtype=np.float64)))
        self.assertTrue(np.allclose(top1_peak(base), np.array([1.0, 0.75], dtype=np.float64)))
        self.assertTrue(np.allclose(top_k_mass(base, top_k=2), np.array([1.0, 1.0], dtype=np.float64)))
        self.assertTrue(np.allclose(overlap_mass(base, proxy), np.array([0.0, 0.5], dtype=np.float64)))
        self.assertTrue(np.allclose(top1_agreement(base, proxy), np.array([0.0, 0.0], dtype=np.float64)))
        self.assertTrue(np.allclose(shared_top_k_mass(base, proxy, top_k=2), np.array([0.0, 1.0], dtype=np.float64)))
        self.assertTrue(np.allclose(top2_margin(base), np.array([1.0, 0.5], dtype=np.float64)))
        self.assertTrue(np.allclose(diagnostics.entropy, np.array([0.0, expected_entropy_row2], dtype=np.float64)))
        self.assertTrue(np.allclose(diagnostics.peak, np.array([1.0, 0.75], dtype=np.float64)))
        self.assertTrue(np.allclose(diagnostics.top_k_mass, np.array([1.0, 1.0], dtype=np.float64)))
        self.assertTrue(np.allclose(diagnostics.overlap, np.array([0.0, 0.5], dtype=np.float64)))
        self.assertTrue(np.allclose(diagnostics.top1_agreement, np.array([0.0, 0.0], dtype=np.float64)))
        self.assertTrue(np.allclose(diagnostics.shared_top_k_mass, np.array([0.0, 1.0], dtype=np.float64)))
        self.assertTrue(np.allclose(diagnostics.top2_margin, np.array([1.0, 0.5], dtype=np.float64)))

    def test_top_k_and_margin_behaviors(self) -> None:
        probabilities = np.array(
            [
                [0.1, 0.2, 0.3, 0.4],
                [0.0, 0.0, 1.0, 0.0],
            ],
            dtype=np.float64,
        )

        self.assertTrue(np.allclose(top_k_mass(probabilities, top_k=1), np.array([0.4, 1.0], dtype=np.float64)))
        self.assertTrue(np.allclose(top_k_mass(probabilities, top_k=3), np.array([0.9, 1.0], dtype=np.float64)))
        self.assertTrue(np.allclose(top2_margin(probabilities), np.array([0.1, 1.0], dtype=np.float64)))

    def test_rejects_shape_mismatch(self) -> None:
        base = np.array([[0.5, 0.5]], dtype=np.float64)
        proxy = np.array([[0.5, 0.25, 0.25]], dtype=np.float64)

        with self.assertRaises(ValueError):
            probability_diagnostics(base, proxy)


if __name__ == "__main__":
    unittest.main()
