from __future__ import annotations

import unittest

import numpy as np

from decepticons.bridge_features import BridgeFeatureConfig, bridge_feature_arrays


class BridgeFeatureTests(unittest.TestCase):
    def test_bridge_feature_shapes(self) -> None:
        base = np.array([[0.4, 0.3, 0.2, 0.1], [0.1, 0.2, 0.3, 0.4]], dtype=np.float64)
        proxy = np.array([[0.1, 0.2, 0.3, 0.4], [0.4, 0.3, 0.2, 0.1]], dtype=np.float64)

        features = bridge_feature_arrays(base, proxy, vocab_size=4)

        self.assertEqual(features.entropy.shape, (2,))
        self.assertEqual(features.peak.shape, (2,))
        self.assertEqual(features.candidate4.shape, (2,))
        self.assertEqual(features.agreement.shape, (2,))
        self.assertEqual(features.agreement_mass.shape, (2,))

    def test_bridge_feature_ranges(self) -> None:
        base = np.array([[0.0, 2.0, 1.0, 0.0], [1.0, 1.0, 1.0, 1.0]], dtype=np.float64)
        proxy = np.array([[0.5, 0.0, 0.5, 0.0], [4.0, 0.0, 0.0, 0.0]], dtype=np.float64)

        features = bridge_feature_arrays(base, proxy, vocab_size=4)

        for array in features.as_dict().values():
            self.assertTrue(np.all(np.isfinite(array)))
            self.assertTrue(np.all(array >= 0.0))
            self.assertTrue(np.all(array <= 1.0))

    def test_bridge_feature_golden_values(self) -> None:
        base = np.array([[1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]], dtype=np.float64)
        proxy = np.array([[1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]], dtype=np.float64)

        features = bridge_feature_arrays(base, proxy, vocab_size=8, config=BridgeFeatureConfig(candidate_count=4))

        self.assertAlmostEqual(float(features.entropy[0]), 0.0, places=7)
        self.assertAlmostEqual(float(features.peak[0]), 1.0, places=7)
        self.assertAlmostEqual(float(features.candidate4[0]), 1.0, places=7)
        self.assertAlmostEqual(float(features.agreement[0]), 1.0, places=7)
        self.assertAlmostEqual(float(features.agreement_mass[0]), 1.0, places=7)

    def test_bridge_feature_golden_values_uniform(self) -> None:
        base = np.full((1, 8), 1.0 / 8.0, dtype=np.float64)
        proxy = np.full((1, 8), 1.0 / 8.0, dtype=np.float64)

        features = bridge_feature_arrays(base, proxy, vocab_size=8, config=BridgeFeatureConfig(candidate_count=4))

        self.assertAlmostEqual(float(features.entropy[0]), 1.0, places=7)
        self.assertAlmostEqual(float(features.peak[0]), 1.0 / 8.0, places=7)
        self.assertAlmostEqual(float(features.candidate4[0]), 0.5, places=7)
        self.assertAlmostEqual(float(features.agreement[0]), 1.0, places=7)
        self.assertAlmostEqual(float(features.agreement_mass[0]), 0.5, places=7)


if __name__ == "__main__":
    unittest.main()
