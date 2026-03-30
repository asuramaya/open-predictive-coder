from __future__ import annotations

import unittest

from open_predictive_coder.bidirectional_context import (
    BidirectionalContextConfig,
    BidirectionalContextProbe,
)


class BidirectionalContextTests(unittest.TestCase):
    def test_deterministic_corpus_reports_full_determinism(self) -> None:
        probe = BidirectionalContextProbe(BidirectionalContextConfig(left_order=1, right_order=1))
        stats = probe.scan(b"abababab")

        self.assertEqual(stats.sequence_length, 8)
        self.assertGreater(stats.pair_context_count, 0)
        self.assertAlmostEqual(stats.deterministic_fraction, 1.0, places=7)
        self.assertAlmostEqual(stats.candidate_le_2_rate, 1.0, places=7)
        self.assertLessEqual(stats.max_candidate_size, 1)

    def test_branching_corpus_reports_multiple_candidates(self) -> None:
        probe = BidirectionalContextProbe(BidirectionalContextConfig(left_order=1, right_order=1))
        stats = probe.scan([1, 2, 9, 4, 1, 2, 8, 4, 1, 2, 7, 4])

        self.assertGreater(stats.max_candidate_size, 1)
        self.assertLess(stats.deterministic_fraction, 1.0)
        self.assertGreater(stats.candidate_le_4_rate, 0.0)

    def test_leave_one_out_reduces_pair_support(self) -> None:
        sequence = [1, 2, 9, 4, 1, 2, 8, 4]
        probe = BidirectionalContextProbe(BidirectionalContextConfig(left_order=1, right_order=1))
        stats = probe.scan(sequence)
        neighborhood = stats.neighborhoods[2]
        leave_one_out = probe.leave_one_out(sequence, 2)

        self.assertEqual(leave_one_out.position, 2)
        self.assertLessEqual(leave_one_out.pair_support, neighborhood.pair_support - 1)
        self.assertLessEqual(leave_one_out.candidate_count, neighborhood.candidate_count)


if __name__ == "__main__":
    unittest.main()
