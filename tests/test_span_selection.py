from __future__ import annotations

import unittest

import numpy as np

from open_predictive_coder import ReplaySpan, ScoredSpan, SpanSelectionConfig, replay_spans_from_scores, select_scored_spans


class SpanSelectionTests(unittest.TestCase):
    def test_select_scored_spans_groups_contiguous_scores(self) -> None:
        scores = np.asarray([0.1, 0.8, 0.9, 0.2, 0.75, 0.1], dtype=np.float64)

        spans = select_scored_spans(scores, SpanSelectionConfig(threshold=0.7))

        self.assertEqual(len(spans), 2)
        self.assertIsInstance(spans[0], ScoredSpan)
        self.assertEqual((spans[0].start, spans[0].stop), (1, 3))
        self.assertEqual(spans[0].count, 2)
        self.assertAlmostEqual(spans[0].max_score, 0.9, places=12)
        self.assertEqual((spans[1].start, spans[1].stop), (4, 5))

    def test_min_span_and_gap_are_respected(self) -> None:
        scores = np.asarray([0.8, 0.1, 0.82, 0.0, 0.9], dtype=np.float64)

        no_gap = select_scored_spans(scores, SpanSelectionConfig(threshold=0.7, min_span=2, max_gap=0))
        with_gap = select_scored_spans(scores, SpanSelectionConfig(threshold=0.7, min_span=2, max_gap=1))

        self.assertEqual(no_gap, ())
        self.assertEqual(len(with_gap), 1)
        self.assertEqual((with_gap[0].start, with_gap[0].stop), (0, 5))
        self.assertEqual(with_gap[0].count, 3)

    def test_replay_spans_from_scores_wraps_make_replay_span(self) -> None:
        scores = np.asarray([0.0, 0.7, 0.8, 0.0], dtype=np.float64)

        spans = replay_spans_from_scores(
            scores,
            SpanSelectionConfig(threshold=0.6),
            label="bridge",
            source_names=("left", "right"),
        )

        self.assertEqual(len(spans), 1)
        self.assertIsInstance(spans[0], ReplaySpan)
        self.assertEqual((spans[0].start, spans[0].stop), (1, 3))
        self.assertEqual(spans[0].label, "bridge")
        self.assertEqual(spans[0].metadata.get("source_names"), ("left", "right"))


if __name__ == "__main__":
    unittest.main()
