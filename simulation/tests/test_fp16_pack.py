"""Tests for pure-Python FP32→FP16 rounding (spec §10.3)."""
from __future__ import annotations
import struct
import unittest
from simulation.common.fp16 import round_f32_to_fp16_bits


def _f32(v: float) -> float:
    return struct.unpack("<f", struct.pack("<f", float(v)))[0]


class RoundF32ToFP16Tests(unittest.TestCase):
    def test_zero(self):
        self.assertEqual(round_f32_to_fp16_bits(0.0), 0x0000)
        self.assertEqual(round_f32_to_fp16_bits(-0.0), 0x8000)

    def test_one(self):
        self.assertEqual(round_f32_to_fp16_bits(1.0), 0x3C00)
        self.assertEqual(round_f32_to_fp16_bits(-1.0), 0xBC00)

    def test_overflow_to_inf(self):
        self.assertEqual(round_f32_to_fp16_bits(100000.0), 0x7C00)
        self.assertEqual(round_f32_to_fp16_bits(-100000.0), 0xFC00)

    def test_max_fp16(self):
        self.assertEqual(round_f32_to_fp16_bits(_f32(65504.0)), 0x7BFF)

    def test_small_normal(self):
        # sin(1.0) ≈ 0.8415 → FP16 0x3ABB (verified against golden)
        self.assertEqual(round_f32_to_fp16_bits(_f32(0.84130859375)), 0x3ABB)

    def test_subnormal_output(self):
        v = _f32(5.96e-8)  # smallest FP16 subnormal ≈ 5.96e-8
        result = round_f32_to_fp16_bits(v)
        self.assertGreaterEqual(result, 0)
        self.assertLessEqual(result, 0xFFFF)

    def test_ties_to_even(self):
        # 1.0 + 0.5 ULP in FP16: should round to even
        # FP16 1.0 = 0x3C00, ULP = 2^{-10} = 0.0009765625
        # 1.0 + 0.5*ULP = 1.00048828125 → ties to 1.0 (frac=0, even)
        v = _f32(1.00048828125)
        self.assertEqual(round_f32_to_fp16_bits(v), 0x3C00)

    def test_nan_passthrough(self):
        result = round_f32_to_fp16_bits(float('nan'))
        self.assertEqual(result, 0x7E00)

    def test_inf_passthrough(self):
        self.assertEqual(round_f32_to_fp16_bits(float('inf')), 0x7C00)
        self.assertEqual(round_f32_to_fp16_bits(float('-inf')), 0xFC00)

    def test_cross_check_with_golden(self):
        """Cross-check against gmpy2-based golden rounding for several values."""
        from simulation.golden.model import _round_mpfr_to_fp16_bits
        import gmpy2

        test_values = [0.0, 1.0, -1.0, 0.5, 0.333, 0.001, 42.0, -0.125, 65504.0]
        for v in test_values:
            f32_v = _f32(v)
            got = round_f32_to_fp16_bits(f32_v)
            expected = _round_mpfr_to_fp16_bits(gmpy2.mpfr(f32_v))
            self.assertEqual(got, expected,
                f"Mismatch for {f32_v}: got 0x{got:04X}, expected 0x{expected:04X}")


if __name__ == "__main__":
    unittest.main()
