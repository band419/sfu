"""Tests for qflow_common spec alignment (SinCos_Derivation §9)."""
from __future__ import annotations
import unittest
from simulation.dut.qflow_common import use_cos_signal, sign_u_bit, sign_out_bit, handle_special


class SpecSignalTests(unittest.TestCase):
    """Verify §9 truth table: use_cos, sign_u, sign_out."""

    # §9.3.1: use_cos = mode ^ q[0], where SIN=0, COS=1
    def test_use_cos_sin_mode(self):
        self.assertFalse(use_cos_signal("sin", 0))
        self.assertTrue(use_cos_signal("sin", 1))
        self.assertFalse(use_cos_signal("sin", 2))
        self.assertTrue(use_cos_signal("sin", 3))

    def test_use_cos_cos_mode(self):
        self.assertTrue(use_cos_signal("cos", 0))
        self.assertFalse(use_cos_signal("cos", 1))
        self.assertTrue(use_cos_signal("cos", 2))
        self.assertFalse(use_cos_signal("cos", 3))

    # §9.3.2: SIN: sign_u = q[1]; COS: sign_u = q[1] ^ q[0]
    def test_sign_u_sin_mode(self):
        self.assertEqual(sign_u_bit("sin", 0), 0)
        self.assertEqual(sign_u_bit("sin", 1), 0)
        self.assertEqual(sign_u_bit("sin", 2), 1)
        self.assertEqual(sign_u_bit("sin", 3), 1)

    def test_sign_u_cos_mode(self):
        self.assertEqual(sign_u_bit("cos", 0), 0)
        self.assertEqual(sign_u_bit("cos", 1), 1)
        self.assertEqual(sign_u_bit("cos", 2), 1)
        self.assertEqual(sign_u_bit("cos", 3), 0)

    # §9.3.3: SIN: sign_out = sign_u ^ sign_x; COS: sign_out = sign_u
    def test_sign_out_sin_mode(self):
        self.assertEqual(sign_out_bit("sin", 0, 0), 0)
        self.assertEqual(sign_out_bit("sin", 0, 1), 1)
        self.assertEqual(sign_out_bit("sin", 2, 0), 1)
        self.assertEqual(sign_out_bit("sin", 2, 1), 0)

    def test_sign_out_cos_ignores_sign_x(self):
        self.assertEqual(sign_out_bit("cos", 0, 0), 0)
        self.assertEqual(sign_out_bit("cos", 0, 1), 0)
        self.assertEqual(sign_out_bit("cos", 1, 0), 1)
        self.assertEqual(sign_out_bit("cos", 1, 1), 1)

    # Special value handling (§5.4)
    def test_special_nan(self):
        for bits in (0x7E00, 0x7E01, 0x7FFF, 0xFE00):
            ok, r = handle_special("sin", bits)
            self.assertTrue(ok)
            self.assertEqual(r, 0x7E00)

    def test_special_inf(self):
        ok, r = handle_special("sin", 0x7C00)
        self.assertTrue(ok)
        self.assertEqual(r, 0x7E00)

    def test_special_zero_sin(self):
        ok, r = handle_special("sin", 0x0000)
        self.assertTrue(ok)
        self.assertEqual(r, 0x0000)
        ok, r = handle_special("sin", 0x8000)
        self.assertTrue(ok)
        self.assertEqual(r, 0x8000)

    def test_special_zero_cos(self):
        ok, r = handle_special("cos", 0x0000)
        self.assertTrue(ok)
        self.assertEqual(r, 0x3C00)

    def test_normal_not_special(self):
        ok, _ = handle_special("sin", 0x3C00)
        self.assertFalse(ok)

    def test_subnormal_is_special_ftz(self):
        """Subnormal inputs are flushed to zero (FTZ) per spec §5.3."""
        # sin(subnormal) -> +0
        ok, bits = handle_special("sin", 0x0001)
        self.assertTrue(ok)
        self.assertEqual(bits, 0x0000)
        # cos(subnormal) -> +1
        ok, bits = handle_special("cos", 0x0001)
        self.assertTrue(ok)
        self.assertEqual(bits, 0x3C00)
        # negative subnormal: sin(-sub) -> -0
        ok, bits = handle_special("sin", 0x8001)
        self.assertTrue(ok)
        self.assertEqual(bits, 0x8000)


if __name__ == "__main__":
    unittest.main()
