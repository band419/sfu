#!/usr/bin/env python3
"""Sweep PH window width (W) to find the ULP vs. area tradeoff.

Extends the window at the LSB side (more guard bits below y_fixed)
while keeping the MSB alignment (i_max = 1 - ex) fixed.

For each W tested:
  - n_bp_int = W - 2  (binary point in p_raw)
  - guard_bits = n_bp_int - W_Y = W - 28
  - ROM depth needed = max |i_min| = (ex_max + n_bp_int) = 5 + W - 2 = W + 3

Usage:
  python simulation/sweep_window_width.py
"""
from __future__ import annotations

import gmpy2
from simulation.common.fp16 import FP16Class, decode_fp16_bits
from simulation.golden.model import golden_fp16, _round_mpfr_to_fp16_bits
from simulation.dut.ph_reduction import stage1_unpack, W_F, W_Y
from simulation.dut.qflow_common import handle_special, use_cos_signal, sign_out_bit
from simulation.ulp_sweep import ulp_distance_fp16, build_inputs, percentile
from statistics import mean

# ── Deep 2/π ROM (128 bits, enough for any W we'd test) ──
_ctx = gmpy2.get_context().copy()
_ctx.precision = 256
with gmpy2.context(_ctx):
    _TWO_OVER_PI_DEEP = int(gmpy2.floor(gmpy2.mpfr(2) / gmpy2.const_pi() * gmpy2.mpfr(2)**128))
_DEEP_DEPTH = 128


def _extract_window_param(ex: int, w: int) -> int:
    """Extract a w-bit window from deep 2/π ROM, extending at LSB side."""
    n_bp_int = w - 2
    i_min = -(ex + n_bp_int)
    i_max = i_min + w - 1  # = 1 - ex (unchanged)

    window = 0
    for i in range(i_max, i_min - 1, -1):
        bit_pos = i - i_min
        if i < 0:
            rom_idx = (-i) - 1
            if rom_idx < _DEEP_DEPTH:
                bit_val = (_TWO_OVER_PI_DEEP >> (_DEEP_DEPTH - 1 - rom_idx)) & 1
                window |= bit_val << bit_pos
    return window


def ph_reduce_param(mx: int, ex: int, mode: str, w: int) -> tuple[int, int, bool]:
    """PH reduction with parameterized window width."""
    n_bp_int = w - 2
    c_window = _extract_window_param(ex, w)
    p_raw = mx * c_window
    q = (p_raw >> n_bp_int) & 0x3
    y_fixed = (p_raw >> (n_bp_int - W_Y)) & ((1 << W_Y) - 1)
    uc = use_cos_signal(mode, q)
    return q, y_fixed, uc


def eval_ph_param(mode: str, x_bits: int, w: int) -> int:
    """Full PH DUT evaluation with parameterized window width."""
    handled, bits = handle_special(mode, x_bits)
    if handled:
        return bits

    x = decode_fp16_bits(x_bits)
    sign_x = x.sign
    mx, ex = stage1_unpack(x_bits)
    q, y_fixed, use_cos = ph_reduce_param(mx, ex, mode, w)

    # mpfr kernel (converging)
    prev_bits = None
    for prec in (128, 256, 512, 1024, 2048, 4096, 8192):
        ctx = gmpy2.get_context().copy()
        ctx.precision = prec
        with gmpy2.context(ctx):
            pi = gmpy2.const_pi()
            y = gmpy2.mpfr(y_fixed) / gmpy2.mpfr(1 << W_Y)
            f_y = gmpy2.cos((pi * y) / 2) if use_cos else gmpy2.sin((pi * y) / 2)
            s_out = sign_out_bit(mode, q, sign_x)
            if s_out:
                f_y = -f_y
            bits = _round_mpfr_to_fp16_bits(f_y)
        if bits == prev_bits:
            return bits
        prev_bits = bits
    raise RuntimeError("did not converge")


def sweep():
    x_inputs = build_inputs(exhaustive=True, samples=0, seed=0)
    widths = [36, 38, 40, 42, 44, 46, 48, 50]

    print(f"{'W':>4}  {'n_bp_int':>8}  {'guard':>5}  {'ROM_need':>8}  "
          f"{'mismatch%':>10}  {'ULP_max':>7}  {'ULP_mean':>8}  "
          f"{'p90':>5}  {'p99':>5}  {'p99.9':>6}")
    print("-" * 90)

    for w in widths:
        n_bp_int = w - 2
        guard = n_bp_int - W_Y  # guard bits below y_fixed
        rom_need = 5 + n_bp_int  # worst case: ex_code=5

        ulps = []
        mismatch = 0
        total_finite = 0

        for x_bits in x_inputs:
            for mode in ("sin", "cos"):
                g_bits = golden_fp16(mode, x_bits)
                d_bits = eval_ph_param(mode, x_bits, w)
                g_cls = decode_fp16_bits(g_bits).cls
                d_cls = decode_fp16_bits(d_bits).cls
                if g_cls in (FP16Class.NAN, FP16Class.INF):
                    continue
                if d_cls in (FP16Class.NAN, FP16Class.INF):
                    continue
                total_finite += 1
                u = ulp_distance_fp16(g_bits, d_bits)
                if u is not None:
                    ulps.append(u)
                    if u != 0:
                        mismatch += 1

        ulps_sorted = sorted(ulps)
        ulp_max = max(ulps_sorted) if ulps_sorted else 0
        ulp_mean = mean(ulps_sorted) if ulps_sorted else 0.0
        p90 = percentile(ulps_sorted, 0.90)
        p99 = percentile(ulps_sorted, 0.99)
        p999 = percentile(ulps_sorted, 0.999)
        rate = 100.0 * mismatch / total_finite if total_finite else 0.0

        print(f"{w:>4}  {n_bp_int:>8}  {guard:>5}  {rom_need:>8}  "
              f"{rate:>9.2f}%  {ulp_max:>7}  {ulp_mean:>8.3f}  "
              f"{p90:>5.1f}  {p99:>5.1f}  {p999:>6.1f}")


if __name__ == "__main__":
    sweep()
