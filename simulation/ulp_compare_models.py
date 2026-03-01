from __future__ import annotations

import csv
from io import StringIO
from dataclasses import dataclass
from statistics import mean

from simulation.dut import get_dut, list_duts
import simulation.dut.adapters  # noqa: F401
from simulation.common.fp16 import FP16Class, decode_fp16_bits
from simulation.golden.model import golden_fp16
from simulation.ulp_sweep import build_inputs, percentile, ulp_distance_fp16


@dataclass
class ModelStats:
    golden: str
    name: str
    total_rows: int = 0
    finite_rows: int = 0
    exact_rows: int = 0
    finite_mismatch_rows: int = 0
    nan_rows: int = 0
    inf_rows: int = 0
    sign_zero_rows: int = 0
    ulps: list[int] | None = None

    def __post_init__(self) -> None:
        if self.ulps is None:
            self.ulps = []


@dataclass
class CompareConfig:
    """Configuration-only script model.

    You only need to set:
    - golden: single golden reference
    - models: DUT model names to compare against the same golden
    - exhaustive/samples/seed: input domain control
    """

    golden: str = "golden-fp16"
    models: list[str] | None = None
    include_golden_mirror: bool = False
    exhaustive: bool = False
    samples: int = 200000
    seed: int = 20260223

    def __post_init__(self) -> None:
        if self.models is None:
            self.models = select_models(include_golden_mirror=self.include_golden_mirror)


def select_models(include_golden_mirror: bool) -> list[str]:
    models: list[str] = []
    if include_golden_mirror:
        models.append("golden-mirror")
    # DUT 1: fixed-format 2/π constant (fp16 / fp32 / fp128)
    fixedconst = sorted(name for name in list_duts() if name.startswith("qflow-fixedconst-"))
    models.extend(fixedconst)
    # DUT 2: PH effective-window reduction
    models.append("qflow-ph")
    # DUT 3: PH reduction + hardware interpolation
    interp = sorted(name for name in list_duts() if name.startswith("qflow-ph-interp-"))
    models.extend(interp)
    return models


def _eval_reference(name: str, mode: str, x_bits: int) -> int:
    if name == "golden-fp16":
        return golden_fp16(mode, x_bits)
    return get_dut(name).eval(mode, x_bits).y_bits


def run_stats(model_name: str, golden_name: str, x_inputs: list[int]) -> ModelStats:
    dut = get_dut(model_name)
    st = ModelStats(golden=golden_name, name=model_name)

    for x_bits in x_inputs:
        for mode in ("sin", "cos"):
            st.total_rows += 1
            g_bits = _eval_reference(golden_name, mode, x_bits)
            d_bits = dut.eval(mode, x_bits).y_bits

            if g_bits == d_bits:
                st.exact_rows += 1

            g_cls = decode_fp16_bits(g_bits).cls
            d_cls = decode_fp16_bits(d_bits).cls

            if g_cls is FP16Class.ZERO and d_cls is FP16Class.ZERO and g_bits != d_bits:
                st.sign_zero_rows += 1

            if g_cls is FP16Class.NAN or d_cls is FP16Class.NAN:
                st.nan_rows += 1
                continue
            if g_cls is FP16Class.INF or d_cls is FP16Class.INF:
                st.inf_rows += 1
                continue

            st.finite_rows += 1
            u = ulp_distance_fp16(g_bits, d_bits)
            if u is None:
                continue
            st.ulps.append(u)
            if u != 0:
                st.finite_mismatch_rows += 1

    return st


def run_matrix_stats(model_names: list[str], golden_name: str, x_inputs: list[int]) -> list[ModelStats]:
    stats: list[ModelStats] = []
    for model_name in model_names:
        stats.append(run_stats(model_name, golden_name, x_inputs))
    return stats


def run_stats_from_config(config: CompareConfig) -> list[ModelStats]:
    """Run compare flow from a config object (no CLI parsing)."""
    golden_name = config.golden.strip()
    if not golden_name:
        raise ValueError("config.golden must be non-empty")

    x_inputs = build_inputs(config.exhaustive, config.samples, config.seed)
    return run_matrix_stats(config.models, golden_name, x_inputs)


def summarize_stats(
    stats: list[ModelStats],
    *,
    inputs_count: int | None = None,
) -> list[dict[str, str | int | float]]:
    """Convert stats to table-ready summary rows.

    If inputs_count is omitted, infer it from stats.total_rows / 2 (sin + cos).
    """
    if inputs_count is None:
        inputs_count = (stats[0].total_rows // 2) if stats else 0
    return [_to_summary_row(st, inputs_count) for st in stats]


def _to_summary_row(st: ModelStats, inputs_count: int) -> dict[str, str | int | float]:
    ulps_sorted = sorted(st.ulps)
    ulp_max = max(ulps_sorted) if ulps_sorted else 0
    ulp_mean = mean(ulps_sorted) if ulps_sorted else 0.0
    p50 = percentile(ulps_sorted, 0.50)
    p90 = percentile(ulps_sorted, 0.90)
    p99 = percentile(ulps_sorted, 0.99)
    p999 = percentile(ulps_sorted, 0.999)

    return {
        "golden": st.golden,
        "model": st.name,
        "inputs": inputs_count,
        "rows_total": st.total_rows,
        "finite_rows": st.finite_rows,
        "exact_rows": st.exact_rows,
        "finite_mismatch_rows": st.finite_mismatch_rows,
        "finite_mismatch_rate_pct": (100.0 * st.finite_mismatch_rows / st.finite_rows) if st.finite_rows else 0.0,
        "nan_rows": st.nan_rows,
        "inf_rows": st.inf_rows,
        "sign_zero_rows": st.sign_zero_rows,
        "ulp_max": ulp_max,
        "ulp_mean": ulp_mean,
        "ulp_p50": p50,
        "ulp_p90": p90,
        "ulp_p99": p99,
        "ulp_p99_9": p999,
    }


def build_csv_table(rows: list[dict[str, str | int | float]]) -> str:
    """Build CSV-formatted summary text from summary rows."""
    fieldnames = [
        "golden",
        "model",
        "exact_rows_over_total",
        "finite_mismatch_rows",
        "finite_mismatch_rate_pct",
        "ulp_max",
        "ulp_mean",
        "ulp_p50",
        "ulp_p90",
        "ulp_p99",
        "ulp_p99_9",
    ]

    buf = StringIO()
    writer = csv.DictWriter(buf, fieldnames=fieldnames)
    writer.writeheader()

    for row in rows:
        writer.writerow(
            {
                "golden": row["golden"],
                "model": row["model"],
                "exact_rows_over_total": f"{row['exact_rows']}/{row['rows_total']}",
                "finite_mismatch_rows": row["finite_mismatch_rows"],
                "finite_mismatch_rate_pct": f"{float(row['finite_mismatch_rate_pct']):.2f}%",
                "ulp_max": row["ulp_max"],
                "ulp_mean": f"{float(row['ulp_mean']):.3f}",
                "ulp_p50": f"{float(row['ulp_p50']):.1f}",
                "ulp_p90": f"{float(row['ulp_p90']):.1f}",
                "ulp_p99": f"{float(row['ulp_p99']):.1f}",
                "ulp_p99_9": f"{float(row['ulp_p99_9']):.1f}",
            }
        )

    return buf.getvalue().strip()
