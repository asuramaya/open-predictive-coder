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

    def test_chosen_probs_supports_order_modes(self) -> None:
        config = NgramMemoryConfig(vocabulary_size=8, bigram_alpha=0.5, trigram_alpha=0.5, trigram_bucket_count=16)
        memory = NgramMemory(config)
        sequence = np.asarray([0, 1, 2, 1, 2, 1, 0], dtype=np.int64)
        memory.fit([sequence])

        unigram = memory.chosen_probs(sequence, order="unigram")
        bigram = memory.chosen_probs(sequence, order="bigram")
        trigram = memory.chosen_probs(sequence, order="trigram")
        maximum = memory.chosen_probs(sequence, order="max")

        self.assertEqual(unigram.shape, sequence.shape)
        self.assertEqual(bigram.shape, sequence.shape)
        self.assertEqual(trigram.shape, sequence.shape)
        self.assertEqual(maximum.shape, sequence.shape)
        self.assertAlmostEqual(float(trigram[0]), float(unigram[0]), places=12)
        self.assertAlmostEqual(float(trigram[1]), float(bigram[1]), places=12)
        self.assertTrue(np.allclose(trigram, maximum))

    def test_update_matches_fit(self) -> None:
        config = NgramMemoryConfig(vocabulary_size=8, bigram_alpha=0.5, trigram_alpha=0.5, trigram_bucket_count=16)
        fit_memory = NgramMemory(config)
        update_memory = NgramMemory(config)
        sequences = [[0, 1, 2, 1], [1, 2, 3, 2]]

        fit_memory.fit(sequences)
        for sequence in sequences:
            update_memory.update(sequence)

        self.assertTrue(np.array_equal(fit_memory.unigram_counts, update_memory.unigram_counts))
        self.assertTrue(np.array_equal(fit_memory.bigram_counts, update_memory.bigram_counts))
        self.assertTrue(np.array_equal(fit_memory.trigram_counts, update_memory.trigram_counts))
        self.assertEqual(fit_memory.report(), update_memory.report())

    def test_high_token_ids_are_preserved(self) -> None:
        config = NgramMemoryConfig(vocabulary_size=2048, bigram_alpha=0.5, trigram_alpha=0.5, trigram_bucket_count=64)
        memory = NgramMemory(config)
        memory.fit([[1000, 1001, 1002, 1001, 1002, 1001, 1002]])
        probs = memory.bigram_probs(1001)
        self.assertEqual(int(np.argmax(probs)), 1002)


if __name__ == "__main__":
    unittest.main()
