from __future__ import annotations

import unittest

from simulation.common.fp16 import FP16Class, classify_fp16, decode_fp16_bits, encode_fp16_bits
from simulation.golden.model import CANONICAL_QNAN_BITS, golden_fp16, gmpy2


@unittest.skipIf(gmpy2 is None, "gmpy2 is not installed")
class GoldenFp16Tests(unittest.TestCase):
    def test_codec_and_classify(self):
        self.assertEqual(classify_fp16(0x0000), FP16Class.ZERO)
        self.assertEqual(classify_fp16(0x8000), FP16Class.ZERO)
        self.assertEqual(classify_fp16(0x0001), FP16Class.SUBNORMAL)
        self.assertEqual(classify_fp16(0x3C00), FP16Class.NORMAL)
        self.assertEqual(classify_fp16(0x7C00), FP16Class.INF)
        self.assertEqual(classify_fp16(0x7E00), FP16Class.NAN)

        d = decode_fp16_bits(0x3C00)
        self.assertEqual(encode_fp16_bits(d.sign, d.exp, d.frac), 0x3C00)

    def test_special_values_policy(self):
        # NaN / inf -> canonical qNaN
        for bits in (0x7E00, 0x7D00, 0xFE00, 0x7C00, 0xFC00):
            self.assertEqual(golden_fp16("sin", bits), CANONICAL_QNAN_BITS)
            self.assertEqual(golden_fp16("cos", bits), CANONICAL_QNAN_BITS)

        # sin(±0) = ±0
        self.assertEqual(golden_fp16("sin", 0x0000), 0x0000)
        self.assertEqual(golden_fp16("sin", 0x8000), 0x8000)

        # cos(±0) = +1
        self.assertEqual(golden_fp16("cos", 0x0000), 0x3C00)
        self.assertEqual(golden_fp16("cos", 0x8000), 0x3C00)

    def test_repeatability_bit_exact(self):
        vectors = [0x0001, 0x03FF, 0x0400, 0x3555, 0x3C00, 0xBC00, 0x7BFF, 0xFBFF]
        for x in vectors:
            for mode in ("sin", "cos"):
                a = golden_fp16(mode, x)
                b = golden_fp16(mode, x)
                self.assertEqual(a, b)

    def test_precision_stability(self):
        vectors = [0x3555, 0x3C00, 0x4200, 0x7BFF]
        for x in vectors:
            for mode in ("sin", "cos"):
                y_low = golden_fp16(mode, x, init_precision=128, max_precision=2048)
                y_high = golden_fp16(mode, x, init_precision=512, max_precision=8192)
                self.assertEqual(y_low, y_high)


if __name__ == "__main__":
    unittest.main()
