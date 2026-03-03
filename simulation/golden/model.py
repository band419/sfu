from __future__ import annotations

from typing import Literal

from simulation.common.fp16 import FP16Class, decode_fp16_bits, encode_fp16_bits

CANONICAL_QNAN_BITS = 0x7E00
POS_ONE_BITS = 0x3C00

try:
    import gmpy2
except ImportError as exc:  # pragma: no cover - exercised in environments without gmpy2
    gmpy2 = None
    _gmpy2_import_error = exc
else:
    _gmpy2_import_error = None


def _is_even_int(n: int) -> bool:
    return (n & 1) == 0


def _round_ties_to_even_nonneg(real_value):
    """Round non-negative mpfr to nearest integer, ties-to-even."""
    floor_i = int(gmpy2.floor(real_value))
    frac = real_value - floor_i
    half = gmpy2.mpfr("0.5")

    if frac > half:
        return floor_i + 1
    if frac < half:
        return floor_i
    return floor_i if _is_even_int(floor_i) else floor_i + 1


def _signed_zero_bits(sign: int) -> int:
    return 0x8000 if sign else 0x0000


def _finite_mpfr_from_fp16_bits(x_bits: int):
    """Exact finite FP16 -> mpfr conversion via power-of-two arithmetic."""
    x = decode_fp16_bits(x_bits)
    if x.cls in (FP16Class.NAN, FP16Class.INF):
        raise ValueError("non-finite input for finite conversion")

    sign = -1 if x.sign else 1
    if x.cls is FP16Class.ZERO:
        return gmpy2.mpfr(0) * sign

    if x.cls is FP16Class.SUBNORMAL:
        # value = (-1)^s * frac * 2^-24
        mag = gmpy2.mul_2exp(gmpy2.mpfr(x.frac), -24)
    else:
        # value = (-1)^s * (1024 + frac) * 2^(exp-25)
        mag = gmpy2.mul_2exp(gmpy2.mpfr(1024 + x.frac), x.exp - 25)

    return -mag if sign < 0 else mag


def _round_mpfr_to_fp16_bits(y):
    """Round finite mpfr y to binary16 using roundTiesToEven."""
    sign_bit = 1 if y < 0 else 0
    ay = -y if y < 0 else y

    if ay == 0:
        return _signed_zero_bits(sign_bit)

    exp2 = int(gmpy2.floor(gmpy2.log2(ay)))

    # Overflow to infinity after rounding
    if exp2 > 15:
        return encode_fp16_bits(sign_bit, 0x1F, 0)

    # Subnormal / underflow region
    if exp2 < -14:
        scaled = gmpy2.mul_2exp(ay, 24)  # ay / 2^-24
        m = _round_ties_to_even_nonneg(scaled)

        if m == 0:
            return _signed_zero_bits(sign_bit)
        if m >= 1024:
            # Rounded up to minimum normal
            return encode_fp16_bits(sign_bit, 1, 0)
        return encode_fp16_bits(sign_bit, 0, m)

    # Normal region: ay ~= m * 2^(exp_unbiased-10), m in [1024, 2047]
    exp_unbiased = exp2
    scaled = gmpy2.mul_2exp(ay, -(exp_unbiased - 10))
    m = _round_ties_to_even_nonneg(scaled)

    if m == 2048:
        m = 1024
        exp_unbiased += 1

    if exp_unbiased > 15:
        return encode_fp16_bits(sign_bit, 0x1F, 0)

    exp_field = exp_unbiased + 15
    frac_field = m - 1024
    return encode_fp16_bits(sign_bit, exp_field, frac_field)


def golden_fp16(
    mode: Literal["sin", "cos"],
    x_bits: int,
    *,
    init_precision: int = 128,
    max_precision: int = 8192,
) -> int:
    """Return correctly-rounded binary16 sin/cos result bits for any FP16 input x_bits.

    The golden model accepts any FP16 bit pattern. No range reduction
    is required by the caller; special values (NaN, Inf, Zero) and
    subnormals (flushed to zero per spec §5.3) are handled internally.
    """

    if mode not in ("sin", "cos"):
        raise ValueError(f"mode must be 'sin' or 'cos', got {mode!r}")

    # Caller contract: x_bits is already range-reduced.
    x = decode_fp16_bits(x_bits)

    # Special classes first.
    if x.cls is FP16Class.NAN:
        return CANONICAL_QNAN_BITS
    if x.cls is FP16Class.INF:
        return CANONICAL_QNAN_BITS

    # Zero identities are exact and policy-fixed.
    if x.cls is FP16Class.ZERO:
        if mode == "sin":
            return x_bits  # preserve signed zero
        return POS_ONE_BITS  # cos(+0) = cos(-0) = +1

    # Subnormal: flush to zero (FTZ) per spec §5.3.
    if x.cls is FP16Class.SUBNORMAL:
        if mode == "sin":
            return x_bits & 0x8000  # sin(±0) = ±0 (preserve sign)
        return POS_ONE_BITS          # cos(±0) = +1

    prev_bits = None
    precision = max(64, int(init_precision))

    while precision <= max_precision:
        ctx = gmpy2.get_context().copy()
        ctx.precision = precision
        with gmpy2.context(ctx):
            x_mp = _finite_mpfr_from_fp16_bits(x_bits)
            y = gmpy2.sin(x_mp) if mode == "sin" else gmpy2.cos(x_mp)
            bits = _round_mpfr_to_fp16_bits(y)

        if bits == prev_bits:
            return bits

        prev_bits = bits
        precision *= 2

    raise RuntimeError(
        f"golden_fp16 did not stabilize up to precision={max_precision} for mode={mode}, x_bits=0x{x_bits:04X}"
    )
