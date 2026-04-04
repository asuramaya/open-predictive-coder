from __future__ import annotations

import unittest

import numpy as np

from decepticons.oscillatory_memory import OscillatoryMemoryConfig, OscillatoryMemorySubstrate


class OscillatoryMemorySubstrateTests(unittest.TestCase):
    def test_initial_state_shape_and_zeros(self) -> None:
        config = OscillatoryMemoryConfig(
            vocabulary_size=32,
            embedding_dim=6,
            decay_rates=(0.3, 0.6, 0.9),
            oscillatory_modes=2,
            seed=13,
        )
        substrate = OscillatoryMemorySubstrate(config)
        state = substrate.initial_state()

        self.assertEqual(state.shape, (config.state_dim,))
        self.assertTrue(np.all(state == 0.0))

    def test_seeded_construction_and_rollout_are_deterministic(self) -> None:
        config = OscillatoryMemoryConfig(
            vocabulary_size=64,
            embedding_dim=8,
            decay_rates=(0.25, 0.5, 0.75),
            oscillatory_modes=3,
            seed=29,
        )
        substrate_a = OscillatoryMemorySubstrate(config)
        substrate_b = OscillatoryMemorySubstrate(config)

        tokens = [3, 7, 11, 13, 17, 19, 23, 29]
        state_a = substrate_a.initial_state()
        state_b = substrate_b.initial_state()
        for token in tokens:
            state_a = substrate_a.step(state_a, token)
            state_b = substrate_b.step(state_b, token)

        np.testing.assert_allclose(state_a, state_b)

    def test_step_remains_finite_over_repeated_rollout(self) -> None:
        config = OscillatoryMemoryConfig(
            vocabulary_size=128,
            embedding_dim=10,
            decay_rates=(0.2, 0.45, 0.7, 0.9),
            oscillatory_modes=4,
            oscillatory_period_range=(5.0, 21.0),
            seed=41,
        )
        substrate = OscillatoryMemorySubstrate(config)
        state = substrate.initial_state()

        for token in [0, 1, 2, 3, 4, 5, 6, 7] * 20:
            state = substrate.step(state, token)

        self.assertEqual(state.shape, (config.state_dim,))
        self.assertTrue(np.all(np.isfinite(state)))
        self.assertGreater(float(np.linalg.norm(state)), 0.0)


if __name__ == "__main__":
    unittest.main()
