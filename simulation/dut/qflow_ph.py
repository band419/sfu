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
from .ph_reduction import stage1_unpack, stage2_ph_reduce, y_fixed_to_f32
from .qflow_common import gmpy2, require_gmpy2, handle_special, sign_out_bit



# ---------------------------------------------------------------------------
# Core computation
# ---------------------------------------------------------------------------



def qflow_ph_core_bits(mode: Mode, x_bits: int, *, precision: int) -> int:
    """PH reduction → mpfr dual kernel {sin,cos}(π/2·y) → select & reconstruct."""
    handled, bits = handle_special(mode, x_bits)
    if handled:
        return bits

    x = decode_fp16_bits(x_bits)
    sign_x = x.sign

    # S1 + S2: spec-aligned PH reduction
    mx, ex = stage1_unpack(x_bits)
    q, y_fixed, use_cos = stage2_ph_reduce(mx, ex, mode)

    # S3: mpfr kernel at requested precision
    ctx = gmpy2.get_context().copy()
    ctx.precision = precision
    with gmpy2.context(ctx):
        pi = gmpy2.const_pi()
        # Convert y_fixed to FP32 (models hardware int-to-float conversion),
        # then promote to mpfr for kernel evaluation.
        y = gmpy2.mpfr(y_fixed_to_f32(y_fixed))

        f_y = gmpy2.cos((pi * y) / 2) if use_cos else gmpy2.sin((pi * y) / 2)

        # S4: sign reconstruction
        s_out = sign_out_bit(mode, q, sign_x)
        if s_out:
            f_y = -f_y

        return _round_mpfr_to_fp16_bits(f_y)


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
