from __future__ import annotations

import unittest

import numpy as np

from open_predictive_coder import bits_per_byte_from_probabilities, bits_per_token_from_probabilities


class MetricsTests(unittest.TestCase):
    def test_bits_per_token_alias_matches_bits_per_byte(self) -> None:
        probabilities = np.asarray([[0.75, 0.25], [0.10, 0.90]], dtype=np.float64)
        targets = np.asarray([0, 1], dtype=np.int64)
        self.assertAlmostEqual(
            bits_per_token_from_probabilities(probabilities, targets),
            bits_per_byte_from_probabilities(probabilities, targets),
            places=12,
        )


if __name__ == "__main__":
    unittest.main()
