from __future__ import annotations

import unittest

from open_predictive_coder import ByteCodec, OracleAnalysisAdapter, OracleAnalysisConfig, OracleAnalysisReport, TrainModeConfig
from open_predictive_coder.train_eval import evaluate_dataset


class OracleAdapterTests(unittest.TestCase):
    def test_compare_uses_checkpoints_and_route_coverage(self) -> None:
        config = OracleAnalysisConfig(
            train_mode=TrainModeConfig(
                state_mode="through_state",
                slow_update_stride=3,
                rollout_checkpoints=(4, 8),
                rollout_checkpoint_stride=4,
            )
        )
        model = OracleAnalysisAdapter(config)
        tokens = ByteCodec.encode_text("oracle analysis contrasts forward and reverse hierarchical state.")
        report = model.compare(tokens)

        self.assertIsInstance(report, OracleAnalysisReport)
        self.assertEqual(report.checkpoints, config.train_mode.resolve_rollout_checkpoints(len(tokens)))
        self.assertEqual(len(report.points), len(report.checkpoints))
        self.assertGreaterEqual(report.oracle_preference_rate, 0.0)
        self.assertLessEqual(report.oracle_preference_rate, 1.0)
        self.assertTrue(all(point.selected_route in {"causal", "oracle"} for point in report.points))
        self.assertTrue(all(point.route_weights.shape[0] == 2 for point in report.points))
        self.assertTrue(any(point.slow_update_active for point in report.points))
        self.assertGreaterEqual(report.bits_per_byte, 0.0)
        self.assertIsInstance(report.accounting.artifact_bytes, int)

    def test_score_and_dataset_evaluation_are_compatible(self) -> None:
        model = OracleAnalysisAdapter(
            OracleAnalysisConfig(
                train_mode=TrainModeConfig(
                    state_mode="detached",
                    slow_update_stride=2,
                    rollout_checkpoints=(4, 8),
                )
            )
        )
        corpus = (
            ByteCodec.encode_text("oracle comparison uses forward and reverse scans."),
            ByteCodec.encode_text("analysis should stay separate from runtime codec claims."),
        )

        fit_report = model.fit(corpus)
        score = model.score(corpus[0])
        dataset_eval = evaluate_dataset(model, corpus)

        self.assertGreaterEqual(fit_report.train_bits_per_byte, 0.0)
        self.assertAlmostEqual(score.bits_per_byte, score.mean_route_bits_per_byte, places=12)
        self.assertGreaterEqual(dataset_eval.bits_per_byte, 0.0)
        self.assertEqual(dataset_eval.sequences, len(corpus))
        self.assertEqual(model.accounting().artifact_name, "oracle_analysis")

    def test_short_sequence_is_rejected(self) -> None:
        model = OracleAnalysisAdapter()
        with self.assertRaises(ValueError):
            model.compare(ByteCodec.encode_text("x"))


if __name__ == "__main__":
    unittest.main()
