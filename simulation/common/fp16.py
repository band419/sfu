from __future__ import annotations

import math
import struct
from dataclasses import dataclass
from enum import Enum


class FP16Class(str, Enum):
    ZERO = "zero"
    SUBNORMAL = "subnormal"
    NORMAL = "normal"
    INF = "inf"
    NAN = "nan"


@dataclass(frozen=True)
class FP16Decoded:
    bits: int
    sign: int
    exp: int
    frac: int
    cls: FP16Class


def _validate_fp16_bits(bits: int) -> int:
    if not isinstance(bits, int):
        raise TypeError(f"fp16 bits must be int, got {type(bits)!r}")
    if bits < 0 or bits > 0xFFFF:
        raise ValueError(f"fp16 bits out of range: {bits}")
    return bits


def decode_fp16_bits(bits: int) -> FP16Decoded:
    bits = _validate_fp16_bits(bits)
    sign = (bits >> 15) & 0x1
    exp = (bits >> 10) & 0x1F
    frac = bits & 0x03FF

    if exp == 0:
        cls = FP16Class.ZERO if frac == 0 else FP16Class.SUBNORMAL
    elif exp == 0x1F:
        cls = FP16Class.INF if frac == 0 else FP16Class.NAN
    else:
        cls = FP16Class.NORMAL

    return FP16Decoded(bits=bits, sign=sign, exp=exp, frac=frac, cls=cls)


def classify_fp16(bits: int) -> FP16Class:
    return decode_fp16_bits(bits).cls


def encode_fp16_bits(sign: int, exp: int, frac: int) -> int:
    if sign not in (0, 1):
        raise ValueError(f"sign must be 0/1, got {sign}")
    if exp < 0 or exp > 0x1F:
        raise ValueError(f"exp out of range: {exp}")
    if frac < 0 or frac > 0x03FF:
        raise ValueError(f"frac out of range: {frac}")
    return (sign << 15) | (exp << 10) | frac


def round_f32_to_fp16_bits(value: float) -> int:
    """Round an FP32 value to FP16 using round-ties-to-even (spec §10.3).

    Pure Python — no gmpy2 dependency. Operates via FP32 bit manipulation.
    Handles: zero, normal, subnormal, infinity, NaN.
    """
    # Extract FP32 bit pattern
    f32_bits = struct.unpack("<I", struct.pack("<f", float(value)))[0]
    sign = (f32_bits >> 31) & 1
    f32_exp = (f32_bits >> 23) & 0xFF
    f32_frac = f32_bits & 0x7FFFFF

    # NaN
    if f32_exp == 0xFF and f32_frac != 0:
        return 0x7E00  # canonical qNaN

    # Infinity
    if f32_exp == 0xFF:
        return (sign << 15) | 0x7C00

    # Zero
    if f32_exp == 0 and f32_frac == 0:
        return sign << 15

    # Finite non-zero: reconstruct unbiased exponent and normalized mantissa
    if f32_exp == 0:
        # FP32 subnormal: normalize
        shift = 23 - f32_frac.bit_length() + 1
        f32_frac = (f32_frac << shift) & 0x7FFFFF
        unbiased = -126 - shift
    else:
        unbiased = f32_exp - 127

    # FP16 biased exponent
    biased_16 = unbiased + 15

    if biased_16 > 30:
        # Overflow → infinity
        return (sign << 15) | 0x7C00

    if biased_16 >= 1:
        # Normal FP16 range
        # FP32 has 23-bit fraction, FP16 has 10-bit fraction
        # Round the lower 13 bits
        round_bits = f32_frac & 0x1FFF  # lower 13 bits
        fp16_frac = f32_frac >> 13      # upper 10 bits

        # Round: ties-to-even
        halfway = 1 << 12  # 0x1000
        if round_bits > halfway or (round_bits == halfway and (fp16_frac & 1)):
            fp16_frac += 1

        if fp16_frac > 0x3FF:
            # Carry into exponent
            fp16_frac = 0
            biased_16 += 1
            if biased_16 > 30:
                return (sign << 15) | 0x7C00  # overflow to inf

        return (sign << 15) | (biased_16 << 10) | fp16_frac

    # Subnormal FP16 range (biased_16 <= 0)
    # FP16 subnormal: value = frac × 2^(-24)
    shift = 1 - biased_16  # >= 1

    if shift > 24:
        # Underflow to zero
        return sign << 15

    # Reconstruct full mantissa with implicit 1
    full_mant = (1 << 23) | f32_frac

    # Shift right by (13 + shift) to get subnormal fraction
    total_shift = 13 + shift
    fp16_frac = full_mant >> total_shift

    # Round bits
    round_mask = (1 << total_shift) - 1
    round_bits = full_mant & round_mask
    halfway = 1 << (total_shift - 1)

    if round_bits > halfway or (round_bits == halfway and (fp16_frac & 1)):
        fp16_frac += 1

    if fp16_frac >= (1 << 10):
        # Rounded up to smallest normal
        return (sign << 15) | (1 << 10)

    return (sign << 15) | fp16_frac
