"""Shared Q-flow helpers used by all DUT variants.

Implements the quadrant selection and sign reconstruction logic from
SinCos_Derivation.md §9, using the spec's exact bit-level formulas.
"""

from __future__ import annotations

from simulation.common.fp16 import FP16Class, decode_fp16_bits
from simulation.golden.model import CANONICAL_QNAN_BITS, POS_ONE_BITS

from .base import Mode

# ---------------------------------------------------------------------------
# gmpy2 lazy-import guard
# ---------------------------------------------------------------------------

try:
    import gmpy2
except ImportError as exc:  # pragma: no cover
    gmpy2 = None
    _gmpy2_import_error = exc
else:
    _gmpy2_import_error = None


def require_gmpy2() -> None:
    """Raise RuntimeError if gmpy2 is not available."""
    if gmpy2 is None:
        raise RuntimeError("gmpy2 is required for DUT evaluation") from _gmpy2_import_error


# ---------------------------------------------------------------------------
# Mode encoding: SIN=0, COS=1  (spec §1.1)
# ---------------------------------------------------------------------------

def _mode_bit(mode: Mode) -> int:
    """Return 0 for SIN, 1 for COS."""
    return 0 if mode == "sin" else 1


# ---------------------------------------------------------------------------
# Quadrant-mapping helpers — spec §9.3
# ---------------------------------------------------------------------------

def use_cos_signal(mode: Mode, q: int) -> bool:
    """LUT select signal: use_cos = mode ^ q[0]  (spec §9.3.1).

    Returns True when S3 should read the cos LUT.
    """
    return (_mode_bit(mode) ^ (q & 1)) == 1


def sign_u_bit(mode: Mode, q: int) -> int:
    """Sign of result for u=|x|  (spec §9.3.2).

    SIN: sign_u = q[1]
    COS: sign_u = q[1] ^ q[0]

    Returns 0 (positive) or 1 (negative).
    """
    q1 = (q >> 1) & 1
    q0 = q & 1
    if mode == "sin":
        return q1
    return q1 ^ q0


def sign_out_bit(mode: Mode, q: int, sign_x: int) -> int:
    """Final output sign  (spec §9.3.3).

    SIN: sign_out = sign_u ^ sign_x   (sin is odd)
    COS: sign_out = sign_u             (cos is even)

    Returns 0 (positive) or 1 (negative).
    """
    su = sign_u_bit(mode, q)
    if mode == "sin":
        return su ^ sign_x
    return su


# ---------------------------------------------------------------------------
# Special-value short-circuit — spec §5.4
# ---------------------------------------------------------------------------

def handle_special(mode: Mode, x_bits: int) -> tuple[bool, int]:
    """Return (True, result_bits) for NaN/Inf/Zero/Subnormal inputs, else (False, 0).
    
    Priority (§5.4): NaN > Inf > Zero > Subnormal(FTZ).
    Subnormals are flushed to zero per spec §5.3.
    """
    x = decode_fp16_bits(x_bits)
    if x.cls is FP16Class.NAN:
        return True, CANONICAL_QNAN_BITS
    if x.cls is FP16Class.INF:
        return True, CANONICAL_QNAN_BITS
    if x.cls is FP16Class.ZERO:
        if mode == "sin":
            return True, x_bits          # sin(±0) = ±0
        return True, POS_ONE_BITS        # cos(±0) = +1
    if x.cls is FP16Class.SUBNORMAL:
        # FTZ: treat subnormal as zero per spec §5.3
        if mode == "sin":
            return True, x_bits & 0x8000  # sin(±0) = ±0
        return True, POS_ONE_BITS            # cos(±0) = +1
    return False, 0
