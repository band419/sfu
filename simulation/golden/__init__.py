"""FP16 sin/cos golden referee model."""

from .model import golden_fp16
from simulation.common.fp16 import (
    FP16Class,
    classify_fp16,
    decode_fp16_bits,
    encode_fp16_bits,
)

__all__ = [
    "golden_fp16",
    "FP16Class",
    "classify_fp16",
    "decode_fp16_bits",
    "encode_fp16_bits",
]
