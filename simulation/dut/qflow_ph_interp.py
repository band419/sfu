"""DUT model 3: Q-flow with PH reduction + configurable hardware interpolation.

Reduction follows the same Payne-Hanek effective-window path as DUT 2
(``qflow_ph``).  The dual-output kernel provides both sin(π/2·y) and
cos(π/2·y) via *hardware-style* lookup + interpolation with parameters:

- **segments**: number of LUT entries / interpolation intervals
- **partition**: ``"uniform"`` or ``"power"`` (non-uniform density near zero)
- **order**: ``"linear"`` (piecewise-linear) or ``"quadratic"`` (3-point Lagrange)

Both sin and cos LUTs share the same node partition.  All interpolation
arithmetic is quantized to IEEE binary32 (fp32) to model a realistic
hardware datapath.

The reconstruction MUX then selects s(y) or c(y) per the quadrant table
(see SinCos_Derivation §6 / §10.6).

Purpose
-------
Explore concrete hardware implementation trade-offs (LUT size, segment
density, interpolation order) while keeping the reduction method fixed to
the PH approach validated by DUT 2.
"""

from __future__ import annotations

from bisect import bisect_right
from dataclasses import dataclass
import math
import struct
from typing import Literal

from simulation.common.fp16 import decode_fp16_bits
from simulation.golden.model import _round_mpfr_to_fp16_bits

from .base import DUTResult, Mode
from .ph_reduction import PHConfig, reduce_q_y_fp16_ph, y_fixed_to_mpfr
from .qflow_common import gmpy2, require_gmpy2, handle_special, use_cos_kernel, sign_u_negative


_PH_CFG = PHConfig()


PartitionStrategy = Literal["uniform", "power"]
InterpOrder = Literal["linear", "quadratic"]


# ---------------------------------------------------------------------------
# Interpolation configuration
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class PHInterpConfig:
    """Fully configurable PH + interpolation DUT parameters."""

    name: str
    segments: int
    partition: PartitionStrategy
    order: InterpOrder
    power: float = 2.0
    # LUT build context precision; LUT entries are explicitly quantized to fp32.
    table_precision: int = 32
    init_precision: int = 128
    max_precision: int = 8192


# ---------------------------------------------------------------------------
# Hardware-style interpolation kernel (fp32 arithmetic)
# ---------------------------------------------------------------------------

class InterpKernel:
    """Dual-output lookup + interpolation engine (fp32 arithmetic).

    Simultaneously provides both s(y) = sin(π/2·y) and c(y) = cos(π/2·y)
    via independent LUTs that share the same node partition.
    """

    @staticmethod
    def _f32(v: float) -> float:
        """Quantize scalar to IEEE-754 binary32."""
        return struct.unpack("<f", struct.pack("<f", float(v)))[0]

    @classmethod
    def _f32_add(cls, a: float, b: float) -> float:
        return cls._f32(cls._f32(a) + cls._f32(b))

    @classmethod
    def _f32_sub(cls, a: float, b: float) -> float:
        return cls._f32(cls._f32(a) - cls._f32(b))

    @classmethod
    def _f32_mul(cls, a: float, b: float) -> float:
        return cls._f32(cls._f32(a) * cls._f32(b))

    @classmethod
    def _f32_div(cls, a: float, b: float) -> float:
        return cls._f32(cls._f32(a) / cls._f32(b))

    def __init__(self, cfg: PHInterpConfig):
        require_gmpy2()
        if cfg.segments < 2:
            raise ValueError("segments must be >= 2")
        if cfg.order == "quadratic" and cfg.segments < 2:
            raise ValueError("quadratic interpolation needs at least 2 segments")
        if cfg.partition == "power" and cfg.power <= 0.0:
            raise ValueError("power partition requires power > 0")

        self._cfg = cfg
        self._nodes: list[float] = []
        self._sin_values: list[float] = []
        self._cos_values: list[float] = []

        n = cfg.segments
        half_pi = self._f32(math.pi / 2.0)
        pwr = self._f32(cfg.power)
        for i in range(n + 1):
            u = self._f32(i / n)
            if cfg.partition == "uniform":
                t = u
            else:
                t = self._f32(math.pow(u, pwr))
            sv = self._f32(math.sin(self._f32_mul(half_pi, t)))
            cv = self._f32(math.cos(self._f32_mul(half_pi, t)))
            self._nodes.append(t)
            self._sin_values.append(sv)
            self._cos_values.append(cv)

    def _find_segment(self, t) -> int:
        idx = bisect_right(self._nodes, t) - 1
        if idx < 0:
            return 0
        max_seg = len(self._nodes) - 2
        if idx > max_seg:
            return max_seg
        return idx

    @classmethod
    def _lerp(cls, x0, y0, x1, y1, x):
        if x1 == x0:
            return y0
        dy = cls._f32_sub(y1, y0)
        dx = cls._f32_sub(x1, x0)
        tx = cls._f32_sub(x, x0)
        ratio = cls._f32_div(tx, dx)
        return cls._f32_add(y0, cls._f32_mul(dy, ratio))

    @classmethod
    def _lagrange3(cls, x0, y0, x1, y1, x2, y2, x):
        """Second-order Lagrange interpolation on 3 support points."""
        n0 = cls._f32_mul(cls._f32_sub(x, x1), cls._f32_sub(x, x2))
        d0 = cls._f32_mul(cls._f32_sub(x0, x1), cls._f32_sub(x0, x2))
        n1 = cls._f32_mul(cls._f32_sub(x, x0), cls._f32_sub(x, x2))
        d1 = cls._f32_mul(cls._f32_sub(x1, x0), cls._f32_sub(x1, x2))
        n2 = cls._f32_mul(cls._f32_sub(x, x0), cls._f32_sub(x, x1))
        d2 = cls._f32_mul(cls._f32_sub(x2, x0), cls._f32_sub(x2, x1))

        t0 = cls._f32_mul(y0, cls._f32_div(n0, d0))
        t1 = cls._f32_mul(y1, cls._f32_div(n1, d1))
        t2 = cls._f32_mul(y2, cls._f32_div(n2, d2))
        return cls._f32_add(cls._f32_add(t0, t1), t2)

    def _interp(self, values: list[float], t: float) -> float:
        """Interpolate over a given value LUT at position t in [0, 1]."""
        t = self._f32(t)
        if t <= self._nodes[0]:
            return values[0]
        if t >= self._nodes[-1]:
            return values[-1]

        seg = self._find_segment(t)

        if self._cfg.order == "linear":
            x0 = self._nodes[seg]
            x1 = self._nodes[seg + 1]
            y0 = values[seg]
            y1 = values[seg + 1]
            return self._lerp(x0, y0, x1, y1, t)

        last_node = len(self._nodes) - 1
        if seg <= 0:
            i0, i1, i2 = 0, 1, 2
        elif seg >= last_node - 1:
            i0, i1, i2 = last_node - 2, last_node - 1, last_node
        else:
            i0, i1, i2 = seg - 1, seg, seg + 1

        return self._lagrange3(
            self._nodes[i0],
            values[i0],
            self._nodes[i1],
            values[i1],
            self._nodes[i2],
            values[i2],
            t,
        )

    def eval_sin(self, y: float) -> float:
        """Evaluate interpolated s(y) = sin(π/2·y) for y in [0, 1]."""
        return self._interp(self._sin_values, y)

    def eval_cos(self, y: float) -> float:
        """Evaluate interpolated c(y) = cos(π/2·y) for y in [0, 1]."""
        return self._interp(self._cos_values, y)


# ---------------------------------------------------------------------------
# Core computation
# ---------------------------------------------------------------------------

def qflow_ph_interp_core_bits(mode: Mode, x_bits: int, *, precision: int, kernel: InterpKernel) -> int:
    """PH reduction → dual-output hardware-style interpolation → select s(y)/c(y)."""
    handled, bits = handle_special(mode, x_bits)
    if handled:
        return bits

    x = decode_fp16_bits(x_bits)
    red = reduce_q_y_fp16_ph(x_bits, cfg=_PH_CFG)
    q = red.q

    # Qy (fixed-point) → fp32 value for interpolation pipeline.
    y = kernel._f32(red.y_fixed / (1 << red.y_frac_bits))

    # Dual-output kernel: both s(y) and c(y) evaluated simultaneously.
    sy = kernel.eval_sin(y)
    cy = kernel.eval_cos(y)

    # Select kernel output per reconstruction table (§10.6).
    result = cy if use_cos_kernel(mode, q) else sy

    if sign_u_negative(mode, q):
        result = kernel._f32(-result)
    if mode == "sin" and x.sign == 1:
        result = kernel._f32(-result)

    # Final output remains fp16.
    return _round_mpfr_to_fp16_bits(gmpy2.mpfr(result))


# ---------------------------------------------------------------------------
# DUT class
# ---------------------------------------------------------------------------

@dataclass
class QFlowPHInterpDUT:
    """DUT: Q-flow with PH reduction + configurable hardware interpolation."""

    cfg: PHInterpConfig

    def __post_init__(self) -> None:
        require_gmpy2()
        self.name = self.cfg.name
        self._kernel = InterpKernel(self.cfg)

    def eval(self, mode: Mode, x_bits: int) -> DUTResult:
        prev_bits: int | None = None
        precision = max(64, int(self.cfg.init_precision))

        while precision <= self.cfg.max_precision:
            bits = qflow_ph_interp_core_bits(mode, x_bits, precision=precision, kernel=self._kernel)
            if bits == prev_bits:
                return DUTResult(
                    y_bits=bits,
                    meta={
                        "impl": self.name,
                        "precision": str(precision),
                        "segments": str(self.cfg.segments),
                        "partition": self.cfg.partition,
                        "order": self.cfg.order,
                        "interp_numfmt": "fp32",
                    },
                )
            prev_bits = bits
            precision *= 2

        raise RuntimeError(
            f"{self.name} did not stabilize up to precision={self.cfg.max_precision} for "
            f"mode={mode}, x_bits=0x{x_bits:04X}"
        )


# ---------------------------------------------------------------------------
# Factory: default interpolation presets
# ---------------------------------------------------------------------------

def build_default_ph_interp_duts() -> list[QFlowPHInterpDUT]:
    """Create pre-configured interpolation DUT variants for horizontal comparison."""
    cfgs: list[PHInterpConfig] = []

    segments_list = [16, 32, 64, 128]
    for seg in segments_list:
        cfgs.append(
            PHInterpConfig(
                name=f"qflow-ph-interp-u{seg}-lin",
                segments=seg,
                partition="uniform",
                order="linear",
            )
        )
        cfgs.append(
            PHInterpConfig(
                name=f"qflow-ph-interp-u{seg}-quad",
                segments=seg,
                partition="uniform",
                order="quadratic",
            )
        )
        cfgs.append(
            PHInterpConfig(
                name=f"qflow-ph-interp-nu{seg}-lin",
                segments=seg,
                partition="power",
                order="linear",
                power=2.0,
            )
        )
        cfgs.append(
            PHInterpConfig(
                name=f"qflow-ph-interp-nu{seg}-quad",
                segments=seg,
                partition="power",
                order="quadratic",
                power=2.0,
            )
        )

    return [QFlowPHInterpDUT(cfg) for cfg in cfgs]
