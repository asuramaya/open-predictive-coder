from __future__ import annotations

import unittest

import numpy as np

from decepticons.learned_segmentation import (
    BoundaryFeatures,
    BoundaryScorerConfig,
    LearnedBoundaryScorer,
    LearnedSegmenter,
)


def _mean_completed_patch_length(segmenter: LearnedSegmenter, steps: int) -> float:
    completed: list[int] = []
    for _ in range(steps):
        decision = segmenter.step()
        if decision.boundary:
            completed.append(decision.patch_length)
    if not completed:
        raise AssertionError("expected at least one completed patch")
    return float(np.mean(completed))


class LearnedSegmentationTests(unittest.TestCase):
    def test_boundary_features_shape_and_probability_range(self) -> None:
        config = BoundaryScorerConfig(feature_dim=5)
        scorer = LearnedBoundaryScorer(config)
        features = BoundaryFeatures(
            novelty=0.25,
            drift=0.1,
            patch_progress=0.5,
            patch_utilization=0.25,
        ).as_array()

        self.assertEqual(features.shape, (5,))
        probability = scorer.probability(features)
        logit = scorer.logit(features)

        self.assertGreaterEqual(probability, 0.0)
        self.assertLessEqual(probability, 1.0)
        self.assertTrue(np.isfinite(logit))
        self.assertEqual(scorer.state.weights.shape, (5,))

    def test_online_update_changes_probability(self) -> None:
        config = BoundaryScorerConfig(
            feature_dim=5,
            learning_rate=0.4,
            initial_bias=-1.5,
            target_regularization=0.0,
        )
        scorer = LearnedBoundaryScorer(config)
        features = BoundaryFeatures(
            novelty=0.0,
            drift=0.0,
            patch_progress=0.75,
            patch_utilization=0.5,
        ).as_array()

        before = scorer.probability(features)
        scorer.update(features, True)
        after = scorer.probability(features)

        self.assertGreater(after, before)
        self.assertGreaterEqual(after, 0.0)
        self.assertLessEqual(after, 1.0)

    def test_segmenter_enforces_min_and_max_patch_sizes(self) -> None:
        min_config = BoundaryScorerConfig(
            feature_dim=5,
            min_patch_size=3,
            max_patch_size=5,
            threshold=0.0,
            initial_bias=-3.0,
        )
        min_segmenter = LearnedSegmenter(min_config)
        min_decisions = [min_segmenter.step() for _ in range(3)]
        self.assertFalse(min_decisions[0].boundary)
        self.assertFalse(min_decisions[1].boundary)
        self.assertTrue(min_decisions[2].boundary)
        self.assertEqual(min_decisions[2].patch_length, 3)

        max_config = BoundaryScorerConfig(
            feature_dim=5,
            min_patch_size=3,
            max_patch_size=5,
            threshold=1.0,
            initial_bias=-3.0,
        )
        max_segmenter = LearnedSegmenter(max_config)
        max_decisions = [max_segmenter.step() for _ in range(5)]
        self.assertFalse(any(decision.boundary for decision in max_decisions[:4]))
        self.assertTrue(max_decisions[4].boundary)
        self.assertEqual(max_decisions[4].patch_length, 5)

    def test_learning_nudges_mean_patch_length_toward_target(self) -> None:
        config = BoundaryScorerConfig(
            feature_dim=5,
            learning_rate=0.35,
            initial_bias=-1.5,
            min_patch_size=2,
            max_patch_size=8,
            target_patch_size=4.0,
            target_regularization=0.15,
            threshold=0.5,
        )

        baseline_segmenter = LearnedSegmenter(config)
        baseline_mean = _mean_completed_patch_length(baseline_segmenter, steps=48)

        training_features = []
        training_targets = []
        target_patch_size = 4
        for index in range(96):
            patch_position = (index % target_patch_size) + 1
            training_features.append(
                BoundaryFeatures(
                    novelty=0.0,
                    drift=0.0,
                    patch_progress=patch_position / float(target_patch_size),
                    patch_utilization=patch_position / float(config.max_patch_size),
                ).as_array()
            )
            training_targets.append(patch_position == target_patch_size)

        learned_segmenter = LearnedSegmenter(config)
        learned_segmenter.fit(np.vstack(training_features), np.asarray(training_targets, dtype=np.float64), epochs=80)
        learned_mean = _mean_completed_patch_length(learned_segmenter, steps=48)

        self.assertLess(abs(learned_mean - 4.0), abs(baseline_mean - 4.0))
        self.assertLess(learned_mean, baseline_mean)


if __name__ == "__main__":
    unittest.main()
