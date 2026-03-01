from __future__ import annotations

import sys
from pathlib import Path

# Allow direct file execution: python .\simulation\run_ulp_compare.py
if __package__ in (None, ""):
    _repo_root = Path(__file__).resolve().parents[1]
    if str(_repo_root) not in sys.path:
        sys.path.insert(0, str(_repo_root))

from simulation.ulp_compare_config import COMPARE_CONFIG
from simulation.ulp_compare_models import build_csv_table, run_stats_from_config, summarize_stats


# Output CSV path (edit if needed)
OUT_CSV = Path("simulation") / "vectors" / "ulp_compare_summary.csv"


def main() -> None:
    # 1) Run compare from config
    stats = run_stats_from_config(COMPARE_CONFIG)

    # 2) Build summary rows
    rows = summarize_stats(stats)
    inputs_count = (stats[0].total_rows // 2) if stats else 0

    # 3) Build CSV and write to file
    csv_text = build_csv_table(rows)
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    OUT_CSV.write_text(csv_text + "\n", encoding="utf-8")

    print(f"Wrote CSV: {OUT_CSV}")
    print(f"Golden: {COMPARE_CONFIG.golden}")
    print(f"Models: {len(COMPARE_CONFIG.models)}")
    print(f"Inputs: {inputs_count} ({'exhaustive' if COMPARE_CONFIG.exhaustive else 'random'})")


if __name__ == "__main__":
    main()
