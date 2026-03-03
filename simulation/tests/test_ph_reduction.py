"""Tests for spec-aligned Payne-Hanek reduction (SinCos_Derivation §6-§7)."""
from __future__ import annotations
import unittest
from simulation.dut.ph_reduction import (
    stage1_unpack,
    stage2_ph_reduce,
    TWO_OVER_PI_ROM,
    ROM_DEPTH,
    _extract_window,
)


class Stage1UnpackTests(unittest.TestCase):
    """§6: mx = {1, frac_x}, ex = exp_x - 25."""

    def test_one_point_zero(self):
        # x = 1.0 → exp_x=15, frac_x=0 → mx=1024, ex=-10
        mx, ex = stage1_unpack(0x3C00)
        self.assertEqual(mx, 1024)
        self.assertEqual(ex, -10)

    def test_max_normal(self):
        # 0x7BFF → exp_x=30, frac_x=0x3FF → mx=2047, ex=+5
        mx, ex = stage1_unpack(0x7BFF)
        self.assertEqual(mx, 2047)
        self.assertEqual(ex, 5)

    def test_min_normal(self):
        # 0x0400 → exp_x=1, frac_x=0 → mx=1024, ex=-24
        mx, ex = stage1_unpack(0x0400)
        self.assertEqual(mx, 1024)
        self.assertEqual(ex, -24)

    def test_pi_approx(self):
        # 0x4248 → exp_x=16, frac_x=0x248 → mx=1024+0x248=1608, ex=-9
        mx, ex = stage1_unpack(0x4248)
        self.assertEqual(mx, 1608)
        self.assertEqual(ex, -9)

    def test_rejects_special(self):
        for bits in (0x0000, 0x8000, 0x7C00, 0xFC00, 0x7E00):
            with self.assertRaises(ValueError):
                stage1_unpack(bits)

    def test_rejects_subnormal(self):
        with self.assertRaises(ValueError):
            stage1_unpack(0x0001)


class TwoOverPiROMTests(unittest.TestCase):
    """Verify the precomputed 2/π ROM constant."""

    def test_rom_accuracy(self):
        approx = TWO_OVER_PI_ROM / (1 << ROM_DEPTH)
        self.assertAlmostEqual(approx, 0.6366197723675814, places=12)

    def test_rom_matches_gmpy2(self):
        import gmpy2
        ctx = gmpy2.get_context().copy()
        ctx.precision = 256
        with gmpy2.context(ctx):
            expected = int(gmpy2.floor(
                gmpy2.mpfr(2) / gmpy2.const_pi() * (gmpy2.mpz(1) << ROM_DEPTH)
            ))
        self.assertEqual(TWO_OVER_PI_ROM, expected)

    def test_window_width(self):
        """Extracted window must always be exactly 40 bits wide."""
        for ex in range(-24, 6):  # all FP16 normal exponents
            w = _extract_window(ex)
            self.assertLess(w, 1 << 40, f"window too wide for ex={ex}")

    def test_window_nonzero_large_input(self):
        """For large inputs (ex >= -1), window must be non-zero."""
        for ex in range(-1, 6):
            w = _extract_window(ex)
            self.assertGreater(w, 0, f"window is zero for ex={ex}")

class Stage2PHReduceTests(unittest.TestCase):
    """§7: Payne-Hanek reduction with pure integer arithmetic."""

    def test_pi_approx_sin(self):
        # x ≈ 3.140625 (0x4248), mode=SIN
        # From spec §13: p ≈ 1.999, q=1, y ≈ 0.999
        mx, ex = stage1_unpack(0x4248)
        q, y_fixed, use_cos = stage2_ph_reduce(mx, ex, "sin")
        self.assertEqual(q, 1)
        y = y_fixed / (1 << 26)
        self.assertAlmostEqual(y, 0.999, delta=0.01)
        self.assertTrue(use_cos)  # use_cos = 0 ^ 1 = 1

    def test_one_sin(self):
        # x = 1.0, mode=SIN → p ≈ 0.6366, q=0, y ≈ 0.6366
        mx, ex = stage1_unpack(0x3C00)
        q, y_fixed, use_cos = stage2_ph_reduce(mx, ex, "sin")
        self.assertEqual(q, 0)
        y = y_fixed / (1 << 26)
        self.assertAlmostEqual(y, 0.6366, delta=0.001)
        self.assertFalse(use_cos)  # 0^0=0

    def test_one_cos(self):
        # x = 1.0, mode=COS → same q,y but use_cos = 1^0 = 1
        mx, ex = stage1_unpack(0x3C00)
        q, y_fixed, use_cos = stage2_ph_reduce(mx, ex, "cos")
        self.assertEqual(q, 0)
        self.assertTrue(use_cos)  # 1^0=1

    def test_max_normal(self):
        # 0x7BFF → |x| ≈ 65504, p ≈ 41680
        mx, ex = stage1_unpack(0x7BFF)
        q, y_fixed, use_cos = stage2_ph_reduce(mx, ex, "sin")
        self.assertIn(q, range(4))
        self.assertGreaterEqual(y_fixed, 0)
        self.assertLess(y_fixed, 1 << 26)

    def test_min_normal(self):
        # 0x0400 → |x| ≈ 6.1e-5, p ≈ 3.88e-5, q=0
        # With 40-bit window + integer mx, y_fixed may be very small but q=0
        mx, ex = stage1_unpack(0x0400)
        q, y_fixed, use_cos = stage2_ph_reduce(mx, ex, "sin")
        self.assertEqual(q, 0)
        self.assertLess(y_fixed, 1 << 26)

    def test_y_fixed_range(self):
        """y_fixed must be in [0, 2^26) for all normal inputs."""
        test_bits = [0x0400, 0x0401, 0x3C00, 0x4248, 0x7000, 0x7BFF]
        for xb in test_bits:
            mx, ex = stage1_unpack(xb)
            q, y_fixed, use_cos = stage2_ph_reduce(mx, ex, "sin")
            self.assertGreaterEqual(y_fixed, 0)
            self.assertLess(y_fixed, 1 << 26)
            self.assertIn(q, range(4))


if __name__ == "__main__":
    unittest.main()
