"""
第三部分：后处理 —— 象限重构 + 异常处理

使用角度加法公式进行精确重构：
  sin(|x|) = sin(K·π/4)·cos(Y·π/4) + cos(K·π/4)·sin(Y·π/4)
  cos(|x|) = cos(K·π/4)·cos(Y·π/4) - sin(K·π/4)·sin(Y·π/4)

其中 sin(K·π/4) 和 cos(K·π/4) 为 K mod 8 的预存常量。

K 从 |x| × 4/π 的归约得到 (前处理始终处理 |x|)。
符号处理: sin(-x) = -sin(x), cos(-x) = cos(x)

异常处理:
  x = 0     → sin = 0, cos = 1
  x = ±∞   → sin = NaN, cos = NaN
  x = NaN   → sin = NaN, cos = NaN
"""

import math
import struct
from dataclasses import dataclass

from .config import HWConfig, DEFAULT_CONFIG


# sin(K·π/4) 和 cos(K·π/4) 的预存常量 (K mod 8)
_SQRT2_OVER_2 = math.sqrt(2.0) / 2.0

_SIN_K_PI_4 = [
    0.0,              # K=0: sin(0)
    _SQRT2_OVER_2,    # K=1: sin(π/4)
    1.0,              # K=2: sin(π/2)
    _SQRT2_OVER_2,    # K=3: sin(3π/4)
    0.0,              # K=4: sin(π)
    -_SQRT2_OVER_2,   # K=5: sin(5π/4)
    -1.0,             # K=6: sin(3π/2)
    -_SQRT2_OVER_2,   # K=7: sin(7π/4)
]

_COS_K_PI_4 = [
    1.0,              # K=0: cos(0)
    _SQRT2_OVER_2,    # K=1: cos(π/4)
    0.0,              # K=2: cos(π/2)
    -_SQRT2_OVER_2,   # K=3: cos(3π/4)
    -1.0,             # K=4: cos(π)
    -_SQRT2_OVER_2,   # K=5: cos(5π/4)
    0.0,              # K=6: cos(3π/2)
    _SQRT2_OVER_2,    # K=7: cos(7π/4)
]


@dataclass
class SinCosOutput:
    """最终输出"""
    sin_result: float
    cos_result: float
    sin_exn: int  # 异常编码 (0=zero, 1=normal, 2=inf, 3=nan)
    cos_exn: int


def quadrant_reconstruct(sin_val: float, cos_val: float,
                         sin_sign: int, k: int, S_x: int,
                         exn: int,
                         cfg: HWConfig = DEFAULT_CONFIG) -> SinCosOutput:
    """
    角度加法重构 + 异常处理。

    前处理计算 |x| × 4/π 的归约，k 和 y 对应 |x|。
    φ = Y·π/4, sin(φ) 和 cos(φ) 由表插值模块计算。

    重构公式:
      sin(|x|) = sin(Kπ/4)·cos(φ) + cos(Kπ/4)·sin(φ)
      cos(|x|) = cos(Kπ/4)·cos(φ) - sin(Kπ/4)·sin(φ)

    符号:
      sin(x) = S_x ? -sin(|x|) : sin(|x|)
      cos(x) = cos(|x|)
    """
    output = SinCosOutput(
        sin_result=0.0,
        cos_result=1.0,
        sin_exn=1,
        cos_exn=1,
    )

    # ── 异常处理 ──
    if exn == 0:
        output.sin_result = -0.0 if S_x else 0.0
        output.cos_result = 1.0
        output.sin_exn = 0
        output.cos_exn = 1
        return output

    if exn == 2:
        output.sin_result = float('nan')
        output.cos_result = float('nan')
        output.sin_exn = 3
        output.cos_exn = 3
        return output

    if exn == 3:
        output.sin_result = float('nan')
        output.cos_result = float('nan')
        output.sin_exn = 3
        output.cos_exn = 3
        return output

    # ── 正常值：角度加法重构 ──

    # sin(φ) 带符号 (y 的符号决定 sin(φ) 的符号)
    signed_sin_phi = -sin_val if sin_sign else sin_val
    cos_phi = cos_val  # cos 在此区间恒正

    # 查表: sin(K·π/4), cos(K·π/4)
    k_mod8 = k & 0x7
    S_k = _SIN_K_PI_4[k_mod8]
    C_k = _COS_K_PI_4[k_mod8]

    # 角度加法:
    #   sin(|x|) = S_k·cos(φ) + C_k·sin(φ)
    #   cos(|x|) = C_k·cos(φ) - S_k·sin(φ)
    raw_sin = S_k * cos_phi + C_k * signed_sin_phi
    raw_cos = C_k * cos_phi - S_k * signed_sin_phi

    # 输入符号: sin(-x) = -sin(x), cos(-x) = cos(x)
    if S_x:
        raw_sin = -raw_sin

    output.sin_result = raw_sin
    output.cos_result = raw_cos

    if raw_sin == 0.0:
        output.sin_exn = 0
    if raw_cos == 0.0:
        output.cos_exn = 0

    return output


def float32_round(val: float) -> float:
    """将 float64 舍入到 float32 精度 (模拟硬件输出)"""
    return struct.unpack('>f', struct.pack('>f', val))[0]

    return output


def float32_round(val: float) -> float:
    """将 float64 舍入到 float32 精度 (模拟硬件输出)"""
    return struct.unpack('>f', struct.pack('>f', val))[0]
