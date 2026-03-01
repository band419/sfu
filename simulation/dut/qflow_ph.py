"""DUT model 2: Q-flow with Payne-Hanek effective-window reduction + mpfr dual kernel.

Reduction uses a sliding-window extraction of 2/π bits aligned to the input
exponent (Payne-Hanek style), producing an exact-width fixed-point ``y``.
The dual-output kernel evaluates both sin(π/2·y) and cos(π/2·y) via
converging mpfr, then the reconstruction MUX selects the correct output
(see SinCos_Derivation §6 / §10.6).

Purpose
-------
Demonstrate that the PH effective-window method preserves ULP accuracy across
the full FP16 input range **without** requiring high-precision floating-point
arithmetic in the datapath.
"""

from __future__ import annotations

from dataclasses import dataclass

from simulation.common.fp16 import decode_fp16_bits
from simulation.golden.model import _round_mpfr_to_fp16_bits

from .base import DUTResult, Mode
from .ph_reduction import PHConfig, reduce_q_y_fp16_ph, y_fixed_to_mpfr
from .qflow_common import gmpy2, require_gmpy2, handle_special, use_cos_kernel, sign_u_negative


_PH_CFG = PHConfig()


# ---------------------------------------------------------------------------
# Core computation
# ---------------------------------------------------------------------------

def qflow_ph_core_bits(mode: Mode, x_bits: int, *, precision: int) -> int:
    """PH reduction → mpfr dual kernel {sin,cos}(π/2·y) → select & reconstruct."""
    handled, bits = handle_special(mode, x_bits)
    if handled:
        return bits

    x = decode_fp16_bits(x_bits)
    ctx = gmpy2.get_context().copy()
    ctx.precision = precision
    with gmpy2.context(ctx):
        pi = gmpy2.const_pi()
        red = reduce_q_y_fp16_ph(x_bits, cfg=_PH_CFG)
        q = red.q
        y = y_fixed_to_mpfr(red.y_fixed, red.y_frac_bits)

        sy = gmpy2.sin((pi * y) / 2)   # s(y) = sin(π/2·y)
        cy = gmpy2.cos((pi * y) / 2)   # c(y) = cos(π/2·y)

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
class QFlowPHDUT:
    """DUT: Q-flow with Payne-Hanek effective-window reduction.

    Uses PH-style fixed-point reduction (sliding 2/π window) for exact
    quadrant + fractional extraction, then evaluates both sin(π/2·y) and
    cos(π/2·y) via mpfr and selects the correct output.
    """

    name: str = "qflow-ph"
    init_precision: int = 128
    max_precision: int = 8192

    def eval(self, mode: Mode, x_bits: int) -> DUTResult:
        require_gmpy2()

        prev_bits: int | None = None
        precision = max(64, int(self.init_precision))

        while precision <= self.max_precision:
            bits = qflow_ph_core_bits(mode, x_bits, precision=precision)
            if bits == prev_bits:
                return DUTResult(y_bits=bits, meta={"impl": self.name, "precision": str(precision)})
            prev_bits = bits
            precision *= 2

        raise RuntimeError(
            f"{self.name} did not stabilize up to precision={self.max_precision} for mode={mode}, x_bits=0x{x_bits:04X}"
        )
