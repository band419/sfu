from __future__ import annotations

import argparse
import math
import random
from dataclasses import dataclass
from statistics import mean

from simulation.dut import get_dut
import simulation.dut.adapters  # noqa: F401  # ensure registration
from simulation.common.fp16 import FP16Class, decode_fp16_bits
from simulation.golden.model import golden_fp16


def bits_to_float(bits: int) -> float:
    """Convert FP16 bit pattern to Python float."""
    d = decode_fp16_bits(bits)
    sign = -1.0 if d.sign else 1.0

    if d.cls is FP16Class.NAN:
        return float("nan")
    if d.cls is FP16Class.INF:
        return sign * float("inf")
    if d.cls is FP16Class.ZERO:
        return math.copysign(0.0, -1.0 if d.sign else 1.0)

    if d.cls is FP16Class.SUBNORMAL:
        v = math.ldexp(float(d.frac), -24)
    else:
        v = math.ldexp(float(1024 + d.frac), d.exp - 25)
    return sign * v


@dataclass
class SweepStats:
    total_rows: int = 0
    finite_rows: int = 0
    nan_rows: int = 0
    inf_rows: int = 0
    sign_zero_rows: int = 0
    exact_match_rows: int = 0
    finite_mismatch_rows: int = 0



def _ordered_key(bits: int) -> int:
    # Monotonic key for IEEE-like bitwise ULP distance.
    if bits & 0x8000:
        return 0xFFFF - bits
    return bits + 0x8000



def ulp_distance_fp16(a_bits: int, b_bits: int) -> int | None:
    a = decode_fp16_bits(a_bits)
    b = decode_fp16_bits(b_bits)

    if a.cls is FP16Class.NAN or b.cls is FP16Class.NAN:
        return None
    if a.cls is FP16Class.INF or b.cls is FP16Class.INF:
        return 0 if a_bits == b_bits else None

    # Treat +0 / -0 as exact for error accounting.
    if a.cls is FP16Class.ZERO and b.cls is FP16Class.ZERO:
        return 0

    return abs(_ordered_key(a_bits) - _ordered_key(b_bits))



def percentile(sorted_values: list[int], p: float) -> float:
    if not sorted_values:
        return 0.0
    idx = int((len(sorted_values) - 1) * p)
    return float(sorted_values[idx])



def build_inputs(exhaustive: bool, samples: int, seed: int) -> list[int]:
    if exhaustive:
        return list(range(0x10000))
    rng = random.Random(seed)
    return [rng.randrange(0, 0x10000) for _ in range(samples)]



def main() -> None:
    parser = argparse.ArgumentParser(description="Sweep ULP distance between DUT and golden for FP16 sin/cos.")
    parser.add_argument("--dut", default="qflow-ph", help="Registered DUT adapter name")
    parser.add_argument("--exhaustive", action="store_true", help="Sweep all 65536 FP16 input bit patterns")
    parser.add_argument("--samples", type=int, default=1_000_000, help="Random sample count when not exhaustive")
    parser.add_argument("--seed", type=int, default=20260222, help="Random seed when not exhaustive")
    args = parser.parse_args()

    dut = get_dut(args.dut)
    x_inputs = build_inputs(args.exhaustive, args.samples, args.seed)

    stats = SweepStats()
    ulps: list[int] = []

    for x_bits in x_inputs:
        for mode in ("sin", "cos"):
            stats.total_rows += 1

            g_bits = golden_fp16(mode, x_bits)
            d_bits = dut.eval(mode, x_bits).y_bits

            if g_bits == d_bits:
                stats.exact_match_rows += 1

            g_cls = decode_fp16_bits(g_bits).cls
            d_cls = decode_fp16_bits(d_bits).cls

            if g_cls is FP16Class.ZERO and d_cls is FP16Class.ZERO and g_bits != d_bits:
                stats.sign_zero_rows += 1

            if g_cls is FP16Class.NAN or d_cls is FP16Class.NAN:
                stats.nan_rows += 1
                continue
            if g_cls is FP16Class.INF or d_cls is FP16Class.INF:
                stats.inf_rows += 1
                continue

            stats.finite_rows += 1
            u = ulp_distance_fp16(g_bits, d_bits)
            if u is None:
                continue
            ulps.append(u)
            if u != 0:
                stats.finite_mismatch_rows += 1

    ulps_sorted = sorted(ulps)
    max_ulp = max(ulps_sorted) if ulps_sorted else 0
    mean_ulp = mean(ulps_sorted) if ulps_sorted else 0.0

    print(f"DUT: {args.dut}")
    print(f"Inputs: {len(x_inputs)} ({'exhaustive' if args.exhaustive else 'random'})")
    print(f"Rows (mode-expanded): {stats.total_rows}")
    print("---")
    print(f"Exact bit matches: {stats.exact_match_rows}")
    print(f"Finite rows: {stats.finite_rows}")
    print(f"Finite mismatches: {stats.finite_mismatch_rows}")
    print(f"NaN rows: {stats.nan_rows}")
    print(f"Inf rows: {stats.inf_rows}")
    print(f"Signed-zero-only diffs: {stats.sign_zero_rows}")
    print("---")
    print(f"ULP max: {max_ulp}")
    print(f"ULP mean: {mean_ulp:.6f}")
    print(f"ULP p50: {percentile(ulps_sorted, 0.50):.1f}")
    print(f"ULP p90: {percentile(ulps_sorted, 0.90):.1f}")
    print(f"ULP p99: {percentile(ulps_sorted, 0.99):.1f}")
    print(f"ULP p99.9: {percentile(ulps_sorted, 0.999):.1f}")


if __name__ == "__main__":
    main()
