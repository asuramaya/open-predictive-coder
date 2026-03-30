from __future__ import annotations

import unittest

import numpy as np

from open_predictive_coder.patch_latent_blocks import (
    GlobalLocalBridge,
    GlobalLocalBridgeConfig,
    LocalByteEncoder,
    LocalByteEncoderConfig,
    PatchPooler,
    PatchPoolerConfig,
)


class LocalByteEncoderTests(unittest.TestCase):
    def test_dimensions_and_determinism(self) -> None:
        config = LocalByteEncoderConfig(
            vocabulary_size=256,
            local_dim=6,
            state_dim=4,
            output_dim=5,
            seed=17,
        )
        encoder_a = LocalByteEncoder(config)
        encoder_b = LocalByteEncoder(config)
        tokens = np.asarray([1, 2, 3, 4, 5], dtype=np.uint8)

        features_a, state_a = encoder_a.encode(tokens)
        features_b, state_b = encoder_b.encode(tokens)

        self.assertEqual(features_a.shape, (5, 5))
        self.assertEqual(state_a.shape, (4,))
        np.testing.assert_allclose(features_a, features_b)
        np.testing.assert_allclose(state_a, state_b)

    def test_fit_output_reduces_feature_error(self) -> None:
        config = LocalByteEncoderConfig(
            vocabulary_size=256,
            local_dim=6,
            state_dim=5,
            output_dim=4,
            output_l2=1e-6,
            seed=23,
        )
        encoder = LocalByteEncoder(config)
        tokens = np.asarray([7, 11, 7, 13, 17, 19, 23, 29], dtype=np.uint8)
        hidden, _ = encoder.hidden_states(tokens)
        rng = np.random.default_rng(5)
        true_weights = rng.normal(scale=0.4, size=(config.state_dim, config.output_dim))
        true_bias = rng.normal(scale=0.1, size=(config.output_dim,))
        targets = np.tanh(hidden @ true_weights + true_bias)

        before = encoder.output_error(hidden, targets)
        after = encoder.fit_output(hidden, targets)
        encoded, _ = encoder.encode(tokens)

        self.assertLess(after, before)
        self.assertEqual(encoded.shape, (tokens.size, config.output_dim))


class PatchPoolerTests(unittest.TestCase):
    def test_mean_last_and_mix_pooling(self) -> None:
        block = np.asarray(
            [
                [1.0, 2.0, 3.0],
                [4.0, 5.0, 6.0],
                [7.0, 8.0, 9.0],
            ],
            dtype=np.float64,
        )

        mean_pooler = PatchPooler(PatchPoolerConfig(mode="mean"))
        last_pooler = PatchPooler(PatchPoolerConfig(mode="last"))
        mix_pooler = PatchPooler(PatchPoolerConfig(mode="mix", mix_weight=0.25))

        np.testing.assert_allclose(mean_pooler.pool(block), np.asarray([4.0, 5.0, 6.0]))
        np.testing.assert_allclose(last_pooler.pool(block), np.asarray([7.0, 8.0, 9.0]))
        np.testing.assert_allclose(
            mix_pooler.pool(block),
            (0.75 * np.asarray([4.0, 5.0, 6.0])) + (0.25 * np.asarray([7.0, 8.0, 9.0])),
        )


class GlobalLocalBridgeTests(unittest.TestCase):
    def test_update_reduces_reconstruction_error(self) -> None:
        config = GlobalLocalBridgeConfig(
            global_dim=3,
            latent_dim=2,
            local_dim=4,
            learning_rate=0.15,
            l2=0.0,
            seed=31,
        )
        bridge = GlobalLocalBridge(config)
        rng = np.random.default_rng(9)
        inputs = rng.normal(scale=0.2, size=(96, config.input_dim))
        true_weights = rng.normal(scale=0.5, size=(config.input_dim, config.local_dim))
        true_bias = rng.normal(scale=0.1, size=(config.local_dim,))
        targets = inputs @ true_weights + true_bias

        before = bridge.reconstruction_error(inputs, targets)
        after = bridge.update(inputs, targets, steps=120)

        self.assertLess(after, before)
        self.assertEqual(
            bridge.predict(inputs[0, : config.global_dim], np.zeros(config.latent_dim)).shape,
            (config.local_dim,),
        )


if __name__ == "__main__":
    unittest.main()
