from __future__ import annotations

import unittest

import simulation.dut.adapters  # noqa: F401
from simulation.compare import compare_with_dut
from simulation.dut import get_dut, list_duts


class DutInterfaceTests(unittest.TestCase):
    def test_builtin_adapter_registered(self):
        self.assertIn("golden-mirror", list_duts())
        self.assertIn("qflow-ph", list_duts())
        self.assertIn("qflow-fixedconst-fp16", list_duts())
        self.assertIn("qflow-fixedconst-fp32", list_duts())
        self.assertIn("qflow-fixedconst-fp128", list_duts())
        self.assertIn("qflow-ph-interp-u64-lin", list_duts())
        self.assertIn("qflow-ph-interp-u64-quad", list_duts())

    def test_golden_mirror_matches(self):
        rows = [
            ("sin", 0x0000, 0x0000),
            ("cos", 0x0000, 0x3C00),
            ("sin", 0x3C00, 0x3ABB),
            ("cos", 0x3C00, 0x3853),
        ]
        results = compare_with_dut("golden-mirror", rows)
        self.assertTrue(all(r.match for r in results))

    def test_qflow_ph_matches_key_vectors(self):
        rows = [
            ("sin", 0x0000, 0x0000),
            ("cos", 0x0000, 0x3C00),
            ("sin", 0x8000, 0x8000),
            ("cos", 0x8000, 0x3C00),
            ("sin", 0x3C00, 0x3ABB),
            ("cos", 0x3C00, 0x3853),
            ("sin", 0xBC00, 0xBABB),
            ("cos", 0xBC00, 0x3853),
        ]
        results = compare_with_dut("qflow-ph", rows)
        self.assertTrue(all(r.match for r in results))

    def test_fixedconst_variants_run(self):
        names = [n for n in list_duts() if n.startswith("qflow-fixedconst-")]
        self.assertTrue(names)

        for name in names:
            dut = get_dut(name)
            for mode in ("sin", "cos"):
                y_bits = dut.eval(mode, 0x3C00).y_bits
                self.assertGreaterEqual(y_bits, 0)
                self.assertLessEqual(y_bits, 0xFFFF)

    def test_interp_ph_variants_run(self):
        names = [n for n in list_duts() if n.startswith("qflow-ph-interp-")]
        self.assertTrue(names)

        for name in names:
            dut = get_dut(name)
            for mode in ("sin", "cos"):
                y_bits = dut.eval(mode, 0x3C00).y_bits
                self.assertGreaterEqual(y_bits, 0)
                self.assertLessEqual(y_bits, 0xFFFF)


if __name__ == "__main__":
    unittest.main()
