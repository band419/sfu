from __future__ import annotations

from dataclasses import dataclass

from simulation.common.fp16 import FP16Class, decode_fp16_bits

from .qflow_common import gmpy2, require_gmpy2


@dataclass(frozen=True)
class PHConfig:
    # FP16 defaults from doc: w_F=10, g=2, g_K=12.
    w_f: int = 10
    g: int = 2
    g_k: int = 12
    # Extra margin bits for safe floor extraction in simulation.
    margin_bits: int = 8

    @property
    def y_frac_bits(self) -> int:
        return self.w_f + self.g + self.g_k

    @property
    def window_bits(self) -> int:
        # W = 2*w_F + g + g_K + 2
        return 2 * self.w_f + self.g + self.g_k + 2


@dataclass(frozen=True)
class PHReduceResult:
    q: int
    y_fixed: int
    y_frac_bits: int


_TWO_OVER_PI_CACHE: dict[int, int] = {}


def _two_over_pi_fixed(bits: int) -> int:
    if bits <= 0:
        raise ValueError("constant bit width must be positive")
    cached = _TWO_OVER_PI_CACHE.get(bits)
    if cached is not None:
        return cached

    require_gmpy2()
    ctx = gmpy2.get_context().copy()
    ctx.precision = max(128, bits + 64)
    with gmpy2.context(ctx):
        val = gmpy2.mpfr(2) / gmpy2.const_pi()
        scaled = gmpy2.floor(val * (gmpy2.mpz(1) << bits))
    out = int(scaled)
    _TWO_OVER_PI_CACHE[bits] = out
    return out


def _abs_mant_exp2_fp16_normalized(x_bits: int) -> tuple[int, int]:
    """Return normalized (mx, ex) as in doc step 10.3.

    Value relation:
      |x| = mx * 2**ex

    where mx is normalized to [2**w_f, 2**(w_f+1)-1] (i.e., 1.F form in fixed-point).
    For FP16, w_f=10 => mx in [1024, 2047].
    """
    x = decode_fp16_bits(x_bits)
    if x.cls in (FP16Class.NAN, FP16Class.INF, FP16Class.ZERO):
        raise ValueError("Payne-Hanek reduction expects finite non-zero input")

    if x.cls is FP16Class.NORMAL:
        mant = 1024 + x.frac
        exp2 = x.exp - 25
        return mant, exp2

    # subnormal: normalize frac to 1.F form.
    frac = x.frac
    lzc = 10 - frac.bit_length()
    mant = frac << (lzc + 1)
    exp2 = -24 - lzc - 1
    return mant, exp2


def reduce_q_y_fp16_ph(
    x_bits: int,
    *,
    cfg: PHConfig | None = None,
) -> PHReduceResult:
    """PH-style fixed-point reduction for FP16 input.

    Computes p = |x| * (2/pi), then extracts:
    - q = floor(p) mod 4
    - y = frac(p), represented as y_fixed / 2**y_frac_bits

    This implementation follows the documented window idea:
    - use W = 2*w_F + g + g_K + 2 effective bits
    - exponent only controls window alignment (dynamic depth)
    - extract q from integer bit1..bit0 and y from fractional bits.
    """
    cfg = cfg or PHConfig()
    y_frac_bits = cfg.y_frac_bits
    if y_frac_bits <= 0:
        raise ValueError("y_frac_bits must be positive")

    mant, exp2 = _abs_mant_exp2_fp16_normalized(x_bits)

    # Need floor(p * 2**y_frac_bits), where p = mant * 2**exp2 * (2/pi).
    # Let s be required binary shift on 2/pi.
    s = exp2 + y_frac_bits

    # Dynamic constant depth: aligns with exponent/window and keeps a safety margin.
    # This mirrors "window slides with exponent" in PH.
    const_bits = max(64, max(0, -s) + cfg.window_bits + cfg.margin_bits)
    two_over_pi_fixed = _two_over_pi_fixed(const_bits)
    raw = mant * two_over_pi_fixed

    # Apply power-of-two shift in integer domain.
    shift = s - const_bits
    if shift >= 0:
        p_scaled = raw << shift
    else:
        p_scaled = raw >> (-shift)

    k = p_scaled >> y_frac_bits
    y_mask = (1 << y_frac_bits) - 1
    y_fixed = p_scaled & y_mask
    q = k & 0x3

    return PHReduceResult(q=q, y_fixed=y_fixed, y_frac_bits=y_frac_bits)


def y_fixed_to_mpfr(y_fixed: int, y_frac_bits: int):
    require_gmpy2()
    return gmpy2.mpfr(y_fixed) / gmpy2.mpfr(1 << y_frac_bits)
