from __future__ import annotations

from dataclasses import dataclass

from simulation.golden.model import golden_fp16

from .base import DUTResult, Mode
from .registry import register_dut
from .qflow_fixedconst import build_fixedconst_duts
from .qflow_ph import QFlowPHDUT
from .qflow_ph_interp import build_default_ph_interp_duts


@dataclass
class GoldenMirrorDUT:
    """Reference adapter used to verify compare plumbing.

    This mirrors golden output and is useful as a harness sanity check.
    """

    name: str = "golden-mirror"

    def eval(self, mode: Mode, x_bits: int) -> DUTResult:
        return DUTResult(y_bits=golden_fp16(mode, x_bits), meta={"impl": "golden-mirror"})


# Auto-register built-in adapters.
register_dut(GoldenMirrorDUT())

# DUT 1: fixed-format 2/π constant (fp16 / fp32 / fp128)
for _dut in build_fixedconst_duts():
    register_dut(_dut)

# DUT 2: PH effective-window reduction + mpfr sin kernel
register_dut(QFlowPHDUT())

# DUT 3: PH reduction + hardware interpolation (multiple presets)
for _dut in build_default_ph_interp_duts():
    register_dut(_dut)
