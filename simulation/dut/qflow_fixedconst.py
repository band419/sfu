"""DUT model 1: Q-flow with fixed-format 2/π constant at configurable precision.

Reduction uses a 2/π constant stored/rounded to a fixed floating-point format
(fp16, fp32, or fp128) and performs the multiply-then-floor reduction at that
precision.  The dual-output kernel evaluates both sin(π/2·y) and cos(π/2·y)
via converging mpfr, then the reconstruction MUX selects the correct output
(see SinCos_Derivation §6 / §10.6).

Purpose
-------
Compare reduction accuracy across constant-format widths and show that
limited constant precision degrades ULP accuracy for large inputs, motivating
the Payne-Hanek (PH) effective-window approach in DUT 2.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

from simulation.common.fp16 import decode_fp16_bits
from simulation.golden.model import (
    _finite_mpfr_from_fp16_bits,
    _round_mpfr_to_fp16_bits,
)

from .base import DUTResult, Mode
from .qflow_common import gmpy2, require_gmpy2, handle_special, use_cos_kernel, sign_u_negative


# ---------------------------------------------------------------------------
# Constant-format definitions
# ---------------------------------------------------------------------------

ConstFormat = Literal["fp16", "fp32", "fp128"]

# IEEE-754 significand precision (including implicit leading bit).
_FORMAT_PRECISION: dict[str, int] = {
    "fp16": 11,    # binary16
    "fp32": 24,    # binary32
    "fp128": 113,  # binary128
}


# ---------------------------------------------------------------------------
# Core computation
# ---------------------------------------------------------------------------

def qflow_fixedconst_core_bits(
    mode: Mode,
    x_bits: int,
    *,
    const_precision: int,
    eval_precision: int,
) -> int:
    """Reduce via fixed-precision 2/π constant, evaluate dual kernel via mpfr.

    The kernel simultaneously computes sin(π/2·y) and cos(π/2·y) and
    selects the correct output per the (mode, q) reconstruction table.

    Parameters
    ----------
    const_precision : int
        Significand bits for the 2/π constant **and** the reduction arithmetic.
    eval_precision : int
        mpfr precision for the dual kernel (isolation from reduction error).
    """
    handled, bits = handle_special(mode, x_bits)
    if handled:
        return bits

    x = decode_fp16_bits(x_bits)

    # --- Reconstruct |x| exactly at high precision first ---
    hi_ctx = gmpy2.get_context().copy()
    hi_ctx.precision = max(64, eval_precision)
    with gmpy2.context(hi_ctx):
        u_exact = _finite_mpfr_from_fp16_bits(x_bits)
        if u_exact < 0:
            u_exact = -u_exact

    # --- Reduction step at const_precision ---
    red_ctx = gmpy2.get_context().copy()
    red_ctx.precision = const_precision
    with gmpy2.context(red_ctx):
        u = gmpy2.mpfr(u_exact)                          # exact for fp16 when prec>=11
        two_over_pi = gmpy2.mpfr(2) / gmpy2.const_pi()   # rounded to const_precision
        p = u * two_over_pi                               # rounded to const_precision
        k = int(gmpy2.floor(p))
        y = p - gmpy2.mpfr(k)
        q = k & 0x3

    # --- Dual-output kernel at eval_precision (high precision, for isolation) ---
    eval_ctx = gmpy2.get_context().copy()
    eval_ctx.precision = eval_precision
    with gmpy2.context(eval_ctx):
        pi = gmpy2.const_pi()
        y_mpfr = gmpy2.mpfr(y)
        sy = gmpy2.sin((pi * y_mpfr) / 2)   # s(y) = sin(π/2·y)
        cy = gmpy2.cos((pi * y_mpfr) / 2)   # c(y) = cos(π/2·y)

        result = cy if use_cos_kernel(mode, q) else sy

        if sign_u_negative(mode, q):
            result = -result
        if mode == "sin" and x.sign == 1:
            result = -result

        return _round_mpfr_to_fp16_bits(result)


# ---------------------------------------------------------------------------
# DUT class
# ---------------------------------------------------------------------------

@dataclass
class QFlowFixedConstDUT:
    """DUT: Q-flow with fixed-format 2/π constant multiplication.

    The reduction multiplies |x| by a 2/π constant that has been rounded to
    ``const_fmt`` precision.  The dual-output kernel evaluates both
    sin(π/2·y) and cos(π/2·y) via converging mpfr, selecting the
    correct output per the (mode, q) reconstruction table.
    """

    const_fmt: ConstFormat
    name: str = ""
    init_precision: int = 128
    max_precision: int = 8192
    _const_prec: int = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if not self.name:
            self.name = f"qflow-fixedconst-{self.const_fmt}"
        self._const_prec = _FORMAT_PRECISION[self.const_fmt]

    def eval(self, mode: Mode, x_bits: int) -> DUTResult:
        require_gmpy2()

        prev_bits: int | None = None
        precision = max(64, int(self.init_precision))

        while precision <= self.max_precision:
            bits = qflow_fixedconst_core_bits(
                mode, x_bits,
                const_precision=self._const_prec,
                eval_precision=precision,
            )
            if bits == prev_bits:
                return DUTResult(
                    y_bits=bits,
                    meta={
                        "impl": self.name,
                        "const_fmt": self.const_fmt,
                        "const_precision": str(self._const_prec),
                        "eval_precision": str(precision),
                    },
                )
            prev_bits = bits
            precision *= 2

        raise RuntimeError(
            f"{self.name} did not stabilize up to precision={self.max_precision} "
            f"for mode={mode}, x_bits=0x{x_bits:04X}"
        )


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def build_fixedconst_duts() -> list[QFlowFixedConstDUT]:
    """Create one DUT per supported constant format (fp16, fp32, fp128)."""
    return [QFlowFixedConstDUT(const_fmt=fmt) for fmt in ("fp16", "fp32", "fp128")]
