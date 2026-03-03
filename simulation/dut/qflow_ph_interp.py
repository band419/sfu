"""DUT model 3: Q-flow with PH reduction + configurable hardware interpolation.

Structured as the spec's S0-S5 pipeline (SinCos_Derivation.md §5-§10):

  S0: Input unpack + exception classification
  S1: Absolute value + mantissa extraction
  S2: Payne-Hanek window multiply → (q, y_fixed, use_cos)
  S3: Selective function kernel (shared interpolation, LUT MUX before kernel)
  S4: Sign reconstruction
  S5: Exception merge + FP16 output pack

The function kernel is SELECTIVE: use_cos selects which LUT to read BEFORE
the shared interpolation circuit. Only one of sin/cos is computed per call.
All interpolation arithmetic is quantized to IEEE binary32 (FP32).
"""

from __future__ import annotations

from bisect import bisect_right
from dataclasses import dataclass
import math
import struct
from typing import Literal

from simulation.common.fp16 import (
    decode_fp16_bits, round_f32_to_fp16_bits,
)

from .base import DUTResult, Mode
from .ph_reduction import W_Y, stage1_unpack, stage2_ph_reduce, y_fixed_to_f32
from .qflow_common import handle_special, sign_out_bit


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
    table_precision: int = 32


# ---------------------------------------------------------------------------
# Hardware-style interpolation kernel (FP32 arithmetic)  — spec §8.4
# ---------------------------------------------------------------------------

class InterpKernel:
    """Selective-output lookup + interpolation engine (FP32 arithmetic).

    sin LUT and cos LUT share the same node partition.
    Per spec §8.1: each call reads ONLY ONE LUT (selected by use_cos),
    then feeds the values into a shared interpolation circuit.
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

    @classmethod
    def _f32_fma(cls, a: float, b: float, c: float) -> float:
        """Fused multiply-add quantized to FP32: round_f32(a * b + c).

        Models a hardware MAC unit: the product a*b is computed at full
        precision (no intermediate rounding), accumulated with c, then
        rounded once to FP32 at the output.
        """
        return cls._f32(math.fma(cls._f32(a), cls._f32(b), cls._f32(c)))
    def __init__(self, cfg: PHInterpConfig):
        if cfg.segments < 2:
            raise ValueError("segments must be >= 2")
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

        # --- Bit-slice precomputations for uniform partitions (§8.4) ---
        if cfg.partition == "uniform":
            self._log2_n = int(math.log2(n))
            if (1 << self._log2_n) != n:
                raise ValueError(f"uniform partition requires power-of-2 segments, got {n}")
            self._delta_shift = W_Y - self._log2_n
            self._delta_mask = (1 << self._delta_shift) - 1
            self._delta_scale = self._f32(1.0 / (1 << self._delta_shift))

            # Precompute per-segment polynomial coefficients.
            # LUT stores coefficients (offline computation, full precision),
            # each coefficient rounded to FP32 for ROM storage.
            self._sin_coeffs = self._build_minimax_coeffs("sin")
            self._cos_coeffs = self._build_minimax_coeffs("cos")


    # ------------------------------------------------------------------
    # Minimax (Remez exchange) coefficient computation
    # ------------------------------------------------------------------

    @staticmethod
    def _gauss_solve(A: list[list[float]], b: list[float]) -> list[float]:
        """Solve Ax = b via Gaussian elimination with partial pivoting."""
        n = len(b)
        M = [row[:] + [bi] for row, bi in zip(A, b)]
        for col in range(n):
            max_row = max(range(col, n), key=lambda r: abs(M[r][col]))
            M[col], M[max_row] = M[max_row], M[col]
            pivot = M[col][col]
            if abs(pivot) < 1e-30:
                raise RuntimeError("singular matrix in minimax solve")
            for row in range(col + 1, n):
                factor = M[row][col] / pivot
                for j in range(col, n + 1):
                    M[row][j] -= factor * M[col][j]
        x = [0.0] * n
        for row in range(n - 1, -1, -1):
            s = sum(M[row][j] * x[j] for j in range(row + 1, n))
            x[row] = (M[row][n] - s) / M[row][row]
        return x

    @staticmethod
    def _ulp_fp16(v: float) -> float:
        """Return the ULP (unit in the last place) of v in IEEE-754 binary16."""
        v = abs(v)
        if v < 2.0 ** -24:
            return 2.0 ** -24  # below smallest subnormal → clamp
        if v < 2.0 ** -14:
            return 2.0 ** -24  # subnormal range
        e = math.floor(math.log2(v))
        return 2.0 ** (e - 10)

    def _remez_segment(
        self, seg: int, n_segs: int, degree: int, func_name: str,
    ) -> tuple[float, ...]:
        """ULP-weighted Remez exchange for minimax polynomial on one segment.

        Minimises  max_{δ∈[0,1]}  |P(δ) − f(δ)| / ULP_fp16(f(δ)),
        i.e. the worst-case ULP error of the polynomial approximation.

        The equioscillation condition becomes:
            w(δ_i) · (P(δ_i) − f(δ_i)) = (−1)^i · E
        where w(δ) = 1 / ULP_fp16(f(δ)).

        Coefficients are computed in Python float64 (offline ROM generation),
        then each value is quantized to FP32 for storage.
        """
        half_pi = math.pi / 2.0
        h = 1.0 / n_segs
        fn = math.sin if func_name == "sin" else math.cos

        def f(d: float) -> float:
            return fn(half_pi * (seg + d) * h)

        def w(d: float) -> float:
            """Weight = 1 / ULP_fp16(f(d))."""
            return 1.0 / self._ulp_fp16(f(d))

        def poly_eval(coeffs: list[float], d: float) -> float:
            acc = 0.0
            for j in range(len(coeffs) - 1, -1, -1):
                acc = acc * d + coeffs[j]
            return acc

        num_refs = degree + 2  # 3 for linear, 4 for quadratic
        # Initial reference: Chebyshev nodes on [0, 1]
        refs = [
            (1.0 - math.cos(math.pi * k / (num_refs - 1))) / 2.0
            for k in range(num_refs)
        ]

        coeffs: list[float] = [0.0] * (degree + 1)
        for _it in range(200):
            # Weighted equioscillation system:
            #   P(δ_i) − f(δ_i) = (−1)^i · E / w(δ_i)
            #   ⇒  c0 + c1·δ_i + … + (−1)^i / w_i · E = f(δ_i)
            A: list[list[float]] = []
            b_vec: list[float] = []
            for i, d in enumerate(refs):
                wi = w(d)
                row = [d ** j for j in range(degree + 1)] + [(-1.0) ** i / wi]
                A.append(row)
                b_vec.append(f(d))
            sol = self._gauss_solve(A, b_vec)
            coeffs = sol[: degree + 1]
            E = sol[degree + 1]

            # Dense search for global max weighted |error| on [0, 1]
            n_pts = 4000
            best_werr = 0.0
            best_pt = 0.0
            for i in range(n_pts + 1):
                d = i / n_pts
                we = abs(w(d) * (poly_eval(coeffs, d) - f(d)))
                if we > best_werr:
                    best_werr = we
                    best_pt = d

            # Convergence: max weighted |error| ≈ |E|
            if abs(best_werr - abs(E)) < 1e-10 + 1e-8 * max(abs(E), 1e-30):
                break

            # Exchange: insert new extremum, drop one ref to keep alternation
            cand = sorted(refs + [best_pt])
            errs = [w(d) * (poly_eval(coeffs, d) - f(d)) for d in cand]
            removed = False
            for i in range(len(cand) - 1):
                if errs[i] * errs[i + 1] > 0:  # same sign → drop smaller
                    if abs(errs[i]) <= abs(errs[i + 1]):
                        del cand[i]
                    else:
                        del cand[i + 1]
                    del errs[min(i, len(errs) - 1)]  # keep errs in sync
                    removed = True
                    break
            if not removed:
                if abs(errs[0]) <= abs(errs[-1]):
                    del cand[0]
                else:
                    del cand[-1]
            refs = cand

        return tuple(self._f32(c) for c in coeffs)

    def _build_minimax_coeffs(
        self, func_name: str,
    ) -> list[tuple[float, ...]]:
        """Compute ULP-weighted minimax polynomial coefficients for all segments.

        Uses the Remez exchange algorithm with weight w(δ) = 1/ULP_fp16(f(δ))
        to minimise worst-case ULP error rather than absolute error.
        Coefficients are computed in float64, then quantized to FP32.
        """
        n = self._cfg.segments
        degree = 1 if self._cfg.order == "linear" else 2
        return [
            self._remez_segment(seg, n, degree, func_name)
            for seg in range(n)
        ]

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

    def _interp_uniform(
        self, coeffs: list[tuple[float, ...]], y_fixed: int,
    ) -> float:
        """Hardware MAC-based interpolation for uniform partitions.

        Segment address and local delta are extracted via integer shift/mask
        on y_fixed directly.  Interpolation uses FMA (fused multiply-add)
        to model the hardware MAC unit — multiply at full precision,
        single rounding at the MAC output.

        Linear  (1 MAC):  fma(c1, δ, c0)
        Quadratic (2 MACs, Horner):  fma(fma(c2, δ, c1), δ, c0)
        """
        # Segment = top log2_n bits of y_fixed
        seg = y_fixed >> self._delta_shift
        max_seg = len(coeffs) - 1
        if seg > max_seg:
            seg = max_seg

        # Delta = remaining bits, converted to FP32 in [0, 1)
        delta_int = y_fixed & self._delta_mask
        delta = self._f32_mul(self._f32(float(delta_int)), self._delta_scale)

        c = coeffs[seg]
        if self._cfg.order == "linear":
            # 1 MAC: result = round_f32(c1 * δ + c0)
            return self._f32_fma(c[1], delta, c[0])

        # Quadratic Horner — 2 cascaded MACs:
        #   stage 1: t      = round_f32(c2 * δ + c1)
        #   stage 2: result = round_f32(t  * δ + c0)
        t = self._f32_fma(c[2], delta, c[1])
        return self._f32_fma(t, delta, c[0])

    def _interp(self, values: list[float], t: float) -> float:
        """Shared interpolation circuit — operates on whichever LUT was selected."""
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
            self._nodes[i0], values[i0],
            self._nodes[i1], values[i1],
            self._nodes[i2], values[i2],
            t,
        )

    # --- Selective kernel API (spec §8.1) ---

    def eval_selective(self, y_fixed: int, *, use_cos: bool) -> float:
        """S3: Evaluate ONE basis function selected by use_cos.

        For uniform partitions, uses hardware bit-slice addressing on y_fixed
        (integer shift/mask — no comparators, no division).

        For non-uniform partitions, converts y_fixed to FP32 first and uses
        the generic bisect + Lagrange path.

        use_cos=False → read sin LUT → shared interpolation → s(y)
        use_cos=True  → read cos LUT → shared interpolation → c(y)

        This models the hardware: LUT MUX selects BEFORE the interpolation
        circuit. Only one LUT is read per call.
        """
        if self._cfg.partition == "uniform":
            coeffs = self._cos_coeffs if use_cos else self._sin_coeffs
            return self._interp_uniform(coeffs, y_fixed)
        # Non-uniform: convert to FP32 normalized t ∈ [0, 1) and use generic path
        t = y_fixed_to_f32(y_fixed)
        values = self._cos_values if use_cos else self._sin_values
        return self._interp(values, t)

    # --- Legacy dual-output API (backward compat) ---

    def eval_sin(self, y: float) -> float:
        return self._interp(self._sin_values, y)

    def eval_cos(self, y: float) -> float:
        return self._interp(self._cos_values, y)



def pipeline_s0_to_s5(mode: Mode, x_bits: int, kernel: InterpKernel) -> int:
    """Execute the full spec pipeline for one FP16 input.

    Returns FP16 output bit pattern.
    """
    # ── S0: Input unpack + exception classification (§5) ──
    x = decode_fp16_bits(x_bits)
    sign_x = x.sign

    is_special, special_result = handle_special(mode, x_bits)

    # ── S1 + S2: Mantissa extraction + PH reduction (§6-§7) ──
    if not is_special:
        mx, ex = stage1_unpack(x_bits)
        q, y_fixed, use_cos = stage2_ph_reduce(mx, ex, mode)

        # ── S3: Selective function kernel (§8) ──
        f_y = kernel.eval_selective(y_fixed, use_cos=use_cos)

        # ── S4: Sign reconstruction (§9) ──
        s_out = sign_out_bit(mode, q, sign_x)

        # Apply sign to magnitude
        if s_out:
            f_y = kernel._f32(-f_y)

        # ── S5: FP16 output pack (§10) ──
        main_result = round_f32_to_fp16_bits(f_y)
    else:
        main_result = 0  # unused

    # ── S5: Exception merge (§10.2) ──
    if is_special:
        return special_result
    return main_result


# ---------------------------------------------------------------------------
# DUT class
# ---------------------------------------------------------------------------

@dataclass
class QFlowPHInterpDUT:
    """DUT: Q-flow with PH reduction + configurable hardware interpolation."""

    cfg: PHInterpConfig

    def __post_init__(self) -> None:
        self.name = self.cfg.name
        self._kernel = InterpKernel(self.cfg)

    def eval(self, mode: Mode, x_bits: int) -> DUTResult:
        bits = pipeline_s0_to_s5(mode, x_bits, self._kernel)
        return DUTResult(
            y_bits=bits,
            meta={
                "impl": self.name,
                "segments": str(self.cfg.segments),
                "partition": self.cfg.partition,
                "order": self.cfg.order,
                "interp_numfmt": "fp32",
            },
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
                segments=seg, partition="uniform", order="linear",
            )
        )
        cfgs.append(
            PHInterpConfig(
                name=f"qflow-ph-interp-u{seg}-quad",
                segments=seg, partition="uniform", order="quadratic",
            )
        )
        cfgs.append(
            PHInterpConfig(
                name=f"qflow-ph-interp-nu{seg}-lin",
                segments=seg, partition="power", order="linear", power=2.0,
            )
        )
        cfgs.append(
            PHInterpConfig(
                name=f"qflow-ph-interp-nu{seg}-quad",
                segments=seg, partition="power", order="quadratic", power=2.0,
            )
        )

    return [QFlowPHInterpDUT(cfg) for cfg in cfgs]
