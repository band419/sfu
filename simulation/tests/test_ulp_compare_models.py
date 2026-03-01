from __future__ import annotations

import unittest

import simulation.dut.adapters  # noqa: F401
from simulation.ulp_compare_models import build_csv_table, run_matrix_stats


class ULPCompareModelsTests(unittest.TestCase):
    def test_matrix_stats_supports_multi_model_single_golden(self):
        models = ["golden-mirror", "qflow-ph"]
        golden = "golden-fp16"
        x_inputs = [0x0000, 0x3C00, 0xBC00]

        stats = run_matrix_stats(models, golden, x_inputs)

        self.assertEqual(len(stats), len(models))
        self.assertEqual({st.golden for st in stats}, {golden})
        self.assertEqual({st.name for st in stats}, set(models))
        for st in stats:
            self.assertEqual(st.total_rows, len(x_inputs) * 2)

    def test_csv_table_contains_expected_columns_and_rows(self):
        rows = [
            {
                "golden": "golden-fp16",
                "model": "golden-mirror",
                "rows_total": 6,
                "exact_rows": 6,
                "finite_mismatch_rows": 0,
                "finite_mismatch_rate_pct": 0.0,
                "ulp_max": 0,
                "ulp_mean": 0.0,
                "ulp_p50": 0.0,
                "ulp_p90": 0.0,
                "ulp_p99": 0.0,
                "ulp_p99_9": 0.0,
            }
        ]

        csv_text = build_csv_table(rows)

        self.assertIn("golden,model,exact_rows_over_total", csv_text)
        self.assertIn("golden-fp16,golden-mirror,6/6,0,0.00%", csv_text)


if __name__ == "__main__":
    unittest.main()
