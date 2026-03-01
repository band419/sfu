from __future__ import annotations

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
