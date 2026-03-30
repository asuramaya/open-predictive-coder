from __future__ import annotations

import unittest

import numpy as np

from open_predictive_coder import ByteCodec, ensure_byte_tokens, ensure_tokens


class CodecTests(unittest.TestCase):
    def test_ensure_tokens_preserves_integer_token_ids(self) -> None:
        tokens = ensure_tokens(np.asarray([0, 255, 256, 1023], dtype=np.int64))
        self.assertEqual(tokens.dtype, np.int64)
        self.assertEqual(tokens.tolist(), [0, 255, 256, 1023])

    def test_ensure_byte_tokens_rejects_out_of_range_values(self) -> None:
        with self.assertRaises(ValueError):
            ensure_byte_tokens([0, 1, 256])

    def test_byte_codec_decode_bytes_rejects_non_byte_tokens(self) -> None:
        with self.assertRaises(ValueError):
            ByteCodec.decode_bytes([0, 512])


if __name__ == "__main__":
    unittest.main()
