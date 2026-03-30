from __future__ import annotations

import unittest

import numpy as np

from open_predictive_coder.ngram_memory import NgramMemory, NgramMemoryConfig


class NgramMemoryTests(unittest.TestCase):
    def test_fit_reports_counts_and_bytes(self) -> None:
        config = NgramMemoryConfig(vocabulary_size=8, bigram_alpha=0.5, trigram_alpha=0.5, trigram_bucket_count=16)
        memory = NgramMemory(config)

        report = memory.fit([[0, 1, 2, 1], [1, 2, 3]])

        self.assertEqual(report.sequences, 2)
        self.assertEqual(report.tokens, 7)
        self.assertEqual(report.vocabulary_size, 8)
        self.assertGreater(report.bigram_contexts, 0)
        self.assertGreater(report.trigram_buckets_used, 0)
        self.assertEqual(report.total_bytes, report.unigram_bytes + report.bigram_bytes + report.trigram_bytes)

    def test_probability_tables_sum_to_one(self) -> None:
        config = NgramMemoryConfig(vocabulary_size=8, bigram_alpha=0.25, trigram_alpha=0.25, trigram_bucket_count=16)
        memory = NgramMemory(config)
        memory.fit([[0, 1, 2, 1, 0, 1, 3], [3, 2, 1, 0]])

        unigram = memory.unigram_probs()
        bigram = memory.bigram_probs(1)
        trigram = memory.trigram_probs(0, 1)

        self.assertAlmostEqual(float(np.sum(unigram)), 1.0, places=7)
        self.assertAlmostEqual(float(np.sum(bigram)), 1.0, places=7)
        self.assertAlmostEqual(float(np.sum(trigram)), 1.0, places=7)

    def test_unseen_contexts_still_smooth(self) -> None:
        config = NgramMemoryConfig(vocabulary_size=8, bigram_alpha=0.75, trigram_alpha=0.75, trigram_bucket_count=16)
        memory = NgramMemory(config)
        memory.fit([[0, 1, 0, 1, 0]])

        bigram = memory.bigram_probs(7)
        trigram = memory.trigram_probs(6, 7)

        self.assertTrue(np.all(bigram > 0.0))
        self.assertTrue(np.all(trigram > 0.0))
        self.assertAlmostEqual(float(np.sum(bigram)), 1.0, places=7)
        self.assertAlmostEqual(float(np.sum(trigram)), 1.0, places=7)

    def test_log_probs_shape_and_finiteness(self) -> None:
        config = NgramMemoryConfig(vocabulary_size=8, bigram_alpha=0.5, trigram_alpha=0.5, trigram_bucket_count=16)
        memory = NgramMemory(config)
        memory.fit([[0, 1, 2, 3, 2, 1, 0]])

        values = memory.log_probs([0, 1, 2, 3, 2, 1, 0])

        self.assertEqual(values.shape, (7,))
        self.assertTrue(np.all(np.isfinite(values)))


if __name__ == "__main__":
    unittest.main()
