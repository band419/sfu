"""Shared Q-flow helpers used by all DUT variants.

Centralizes the dual-kernel selection, sign-reconstruction, and
special-value handling that are common to every Q-flow DUT model.

The reconstruction follows the Quadrant (Q) scheme documented in
``SinCos_Derivation.md`` §6 / §10.6:

  The function kernel simultaneously provides
      s(y) = sin(π/2·y)   and   c(y) = cos(π/2·y)
  for the same fractional input y ∈ [0, 1).

  A MUX driven by (mode, q) selects **which** kernel output to forward,
  then sign reconstruction produces the final result.
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
# Quadrant-mapping helpers
# ---------------------------------------------------------------------------

def use_cos_kernel(mode: Mode, q: int) -> bool:
    """Return True when the reconstruction should pick c(y) = cos(π/2·y).

    Reconstruction table (SinCos_Derivation §10.6):

        mode  q   kernel
        SIN   0   s(y)      SIN   1   c(y)
        SIN   2   s(y)      SIN   3   c(y)
        COS   0   c(y)      COS   1   s(y)
        COS   2   c(y)      COS   3   s(y)

    Equivalent logic:
        SIN → use c(y) when q is odd
        COS → use c(y) when q is even
    """
    return ((mode == "sin") and ((q & 0x1) == 1)) or ((mode == "cos") and ((q & 0x1) == 0))


def sign_u_negative(mode: Mode, q: int) -> bool:
    """Decide whether the unsigned-path result should be negated.

    SIN: q=2,3 -> negative
    COS: q=1,2 -> negative
    """
    if mode == "sin":
        return (q & 0x2) != 0
    return q in (1, 2)


# ---------------------------------------------------------------------------
# Special-value short-circuit
# ---------------------------------------------------------------------------

def handle_special(mode: Mode, x_bits: int) -> tuple[bool, int]:
    """Return (True, result_bits) for NaN/Inf/Zero inputs, else (False, 0)."""
    x = decode_fp16_bits(x_bits)
    if x.cls is FP16Class.NAN:
        return True, CANONICAL_QNAN_BITS
    if x.cls is FP16Class.INF:
        return True, CANONICAL_QNAN_BITS
    if x.cls is FP16Class.ZERO:
        if mode == "sin":
            return True, x_bits
        return True, POS_ONE_BITS
    return False, 0
