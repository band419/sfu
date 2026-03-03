"""Payne-Hanek range reduction for FP16 sin/cos (SinCos_Derivation §6-§7).

Provides spec-aligned APIs: stage1_unpack() + stage2_ph_reduce() — pure Python integers.

Implementation uses the spec's sliding-window approach (§7.5-§7.6):
  1. Store 2/π as a fixed-width ROM (§7.5.2)
  2. Extract a W=40-bit window based on ex (§7.5.2)
  3. Integer multiply mx × C_window (11 × 40 → 51 bits) (§7.6.1)
  4. Fixed-position slicing for q and y_fixed (§7.6.3)

Convention note:
  The spec §7.4 derives window formulas using fractional mx ∈ [1,2) with
  ex_spec = exp_x - 15. Our code uses integer mx ∈ [1024,2047] with
  ex_code = exp_x - 25 (= ex_spec - w_F). The window position formulas
  are adjusted accordingly:
    i_min = -(ex_code + W_Y + G_C + W_F) = -(ex_code + 38)
    i_max = 1 - ex_code
  This ensures the binary point lands at P_raw bit 38 (= W_Y + G_C + W_F),
  and q/y are extracted from fixed positions regardless of ex.
"""

from __future__ import annotations

import struct

from simulation.common.fp16 import FP16Class, decode_fp16_bits

from .qflow_common import use_cos_signal
from .base import Mode


# ===== Spec-aligned constants (SinCos_Derivation §1.3, §7) =====

W_F = 10       # FP16 fraction bits
G = 4          # guard bits (rounding protection for kernel + final round)
G_K = 12       # Kahan guard bits
G_C = 2        # carry guard bits (absorb right-truncation carry propagation)
W_Y = W_F + G + G_K       # = 26, y_fixed bit width (spec W_y)
W = 2 * W_F + G + G_K + G_C + 2  # = 40, PH window width

# Binary point in P_raw when using integer mx (see convention note above).
# n_bp_int = W_Y + G_C + W_F = 38. P_raw's bit 38 corresponds to p's 2^0.
_N_BP_INT = W_Y + G_C + W_F  # = 38

# ===== 2/π ROM (SinCos_Derivation §7.5) =====
#
# ROM stores the first ROM_DEPTH bits of the 2/π binary fraction.
#   bit index 0 (MSB) → c_{-1}  (weight 2^{-1})
#   bit index k       → c_{-(k+1)}  (weight 2^{-(k+1)})
#
# For FP16 normals (ex_code ∈ [-24, +5]):
#   i_min_global = -(5 + 38) = -43 → need c_{-43} → ROM index 42
# So 43 bits is the exact minimum. We use 48 bits for margin.

ROM_DEPTH = 48
TWO_OVER_PI_ROM = 0xA2F9836E4E44  # floor(2/π × 2^48), verified against gmpy2


# ===== Stage 1: Absolute value & mantissa extraction (§6) =====

def stage1_unpack(x_bits: int) -> tuple[int, int]:
    """S1: Extract (mx, ex) from normal FP16 input.

    mx = {1'b1, frac_x}   — 11-bit unsigned, range [1024, 2047]
    ex = exp_x - 25        — 6-bit signed, range [-24, +5]

    Rejects special values (Zero, Inf, NaN) and subnormals.
    Spec §6 states: "本设计假定输入一定是正规数".
    """
    x = decode_fp16_bits(x_bits)
    if x.cls is not FP16Class.NORMAL:
        raise ValueError(
            f"stage1_unpack expects normal FP16, got {x.cls.value}"
            + f" for x_bits=0x{x_bits:04X}"
        )
    mx = (1 << W_F) | x.frac      # {1, frac_x} = 1024 + frac
    ex = x.exp - (W_F + 15)       # exp_x - 25
    return mx, ex


# ===== Stage 2: Payne-Hanek reduction (§7) =====

def _extract_window(ex: int) -> int:
    """Extract a W=40-bit window from the 2/π ROM for the given exponent.

    Window covers bit positions [i_max, i_min] of the 2/π binary expansion
    where:
        i_min = -(ex + 38)
        i_max = 1 - ex

    Bits at i ≥ 0 are zero (since 2/π < 1). Bits at i < 0 are read from ROM.
    The window is returned as a W-bit unsigned integer with:
        MSB = c_{i_max}, LSB = c_{i_min}

    In hardware this is a combinational MUX from the ROM wire bundle.
    """
    i_min = -(ex + _N_BP_INT)  # -(ex + 38)
    i_max = i_min + W - 1      # 1 - ex

    window = 0
    for i in range(i_max, i_min - 1, -1):
        bit_pos = i - i_min  # position in window (LSB = 0)
        if i < 0:
            rom_idx = (-i) - 1  # 0-indexed from MSB of ROM
            if rom_idx < ROM_DEPTH:
                bit_val = (TWO_OVER_PI_ROM >> (ROM_DEPTH - 1 - rom_idx)) & 1
                window |= bit_val << bit_pos
        # i >= 0: c_i = 0, no OR needed
    return window


def stage2_ph_reduce(mx: int, ex: int, mode: Mode) -> tuple[int, int, bool]:
    """S2: PH reduction → (q, y_fixed, use_cos).

    Pure integer arithmetic, no gmpy2. Models the hardware RTL structure:

    1. Extract W=40-bit window C_window from 2/π ROM based on ex (§7.5)
    2. Integer multiply P_raw = mx × C_window (11 × 40 → 51 bits) (§7.6.1)
    3. Fixed-position slicing (§7.6.3):
       - q       = P_raw[_N_BP_INT+1 : _N_BP_INT]  (2 bits)
       - y_fixed = P_raw[_N_BP_INT-1 : _N_BP_INT-W_Y]  (W_Y=26 bits)

    The bottom (W_F + G_C) = 12 bits of P_raw (P_raw[11:0]) are sub-LSB guard
    bits that absorb truncation error from the finite window width.
    """
    # Step 1: Window extraction from ROM (§7.5)
    c_window = _extract_window(ex)

    # Step 2: Integer multiply (§7.6.1) — 11 × 40 → 51 bits
    p_raw = mx * c_window

    # Step 3: Fixed slicing (§7.6.3)
    # Binary point is at bit _N_BP_INT = 38 (for integer mx convention)
    q = (p_raw >> _N_BP_INT) & 0x3                        # bits [39:38]
    y_fixed = (p_raw >> (W_F + G_C)) & ((1 << W_Y) - 1)  # bits [37:12]

    # LUT select signal (§9.3.1)
    uc = use_cos_signal(mode, q)

    return q, y_fixed, uc


# ===== y_fixed → FP32 conversion =====

def y_fixed_to_f32(y_fixed: int) -> float:
    """Convert W_Y-bit unsigned fixed-point y to IEEE-754 binary32 (FP32).

    Models the hardware integer-to-float conversion pipeline:
      1. CLZ over W_Y bits
      2. Left-shift normalize (MSB → hidden-bit position)
      3. Extract 23-bit mantissa + GR bits
      4. Round-to-nearest-even
      5. Pack {sign=0, biased_exp, mantissa}

    y_fixed is an unsigned integer representing y = y_fixed / 2^W_Y ∈ [0, 1).
    The result is an IEEE-754 binary32 Python float.
    """
    if y_fixed == 0:
        return 0.0

    # Step 1: CLZ over W_Y bits
    lzc = W_Y - y_fixed.bit_length()

    # Step 2: Normalize — left-shift so hidden bit is at position (W_Y - 1)
    shifted = y_fixed << lzc  # bit (W_Y-1) is now 1

    # Step 3: Extract mantissa (23 bits after hidden bit)
    # shifted is W_Y=26 bits: [25]=hidden, [24:2]=mantissa(23b), [1]=G, [0]=R
    frac_bits = W_Y - 1 - 23  # = 2, sub-mantissa bits available for rounding
    mantissa = (shifted >> frac_bits) & 0x7FFFFF

    # Step 4: Round-to-nearest-even (GRS)
    guard = (shifted >> 1) & 1
    round_bit = shifted & 1
    sticky = 0  # no bits below y_fixed's LSB
    if guard and (round_bit or sticky or (mantissa & 1)):
        mantissa += 1
        if mantissa >= (1 << 23):  # mantissa overflow → bump exponent
            mantissa = 0
            lzc -= 1

    # Step 5: Biased exponent
    # y = y_fixed / 2^W_Y = 1.xxx × 2^(bit_length-1-W_Y) = 1.xxx × 2^(-1-lzc)
    biased_exp = 126 - lzc

    # Step 6: Pack IEEE-754 binary32
    fp32_int = (biased_exp << 23) | mantissa
    return struct.unpack('<f', struct.pack('<I', fp32_int))[0]
