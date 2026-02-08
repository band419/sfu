"""
第二部分：表插值 —— HOTBM 行为建模

模拟硬件中 HOTBM (Hardware-Oriented Table-Based Method) 的行为：
  - cos 通路 (纯定点): 计算 f_cos(y) = 1 - cos(π/4 · y), 再用 1 - f_cos 恢复
  - sin 通路 (定点×浮点): 计算 f_sin(y) = π/4 - sin(π/4·y)/y, 再用 y×(π/4 - f_sin)

硬件中这两个函数由查找表 + 低次多项式实现；
此处用分段多项式表来行为级建模，重点模拟：
  1. 定点量化
  2. 表查找 + 插值
  3. 位宽截断/舍入

可通过 method 参数在 "ideal"（直接用 math 库）和 "table"（查表插值）间切换。
"""

import math
import numpy as np
from dataclasses import dataclass
from typing import Tuple

from .config import HWConfig, DEFAULT_CONFIG


# ─────────────────────────────────────────────
# 定点工具函数
# ─────────────────────────────────────────────

def float_to_fixed(val: float, frac_bits: int) -> int:
    """将浮点数转换为定点整数 (四舍五入)"""
    return int(round(val * (1 << frac_bits)))


def fixed_to_float(val: int, frac_bits: int) -> float:
    """将定点整数转换回浮点数"""
    return val / (1 << frac_bits)


def truncate_fixed(val: int, total_bits: int) -> int:
    """截断到指定位宽 (保留低 total_bits 位)"""
    mask = (1 << total_bits) - 1
    return val & mask


# ─────────────────────────────────────────────
# HOTBM 查找表构建
# ─────────────────────────────────────────────

@dataclass
class HOTBMTable:
    """
    HOTBM 表：将输入区间 [0, 0.5] 均匀分段，
    每段存储 2 阶 minimax 多项式的系数 (c0, c1, c2)。

    f(y) ≈ c0 + c1·δ + c2·δ²
    其中 δ = y - y_base (段内偏移)
    """
    num_segments: int       # 段数 (2^addr_bits)
    addr_bits: int          # 地址位宽
    coeff_bits: int         # 系数定点位宽
    # 系数表，每段 3 个系数 [c0, c1, c2]
    # 存储为定点整数
    c0_table: np.ndarray
    c1_table: np.ndarray
    c2_table: np.ndarray
    seg_width: float        # 每段宽度


def build_cos_table(cfg: HWConfig, addr_bits: int = 8) -> HOTBMTable:
    """
    构建 cos 通路查找表。
    目标函数: f_cos(y) = 1 - cos(π/4 · y)，y ∈ [0, 0.5]

    使用 2 阶多项式 fit 每个分段。
    """
    num_seg = 1 << addr_bits
    seg_width = 0.5 / num_seg
    coeff_bits = cfg.w_F + cfg.g + 4  # 多留几位内部精度

    c0 = np.zeros(num_seg, dtype=np.float64)
    c1 = np.zeros(num_seg, dtype=np.float64)
    c2 = np.zeros(num_seg, dtype=np.float64)

    for i in range(num_seg):
        y_lo = i * seg_width
        y_hi = (i + 1) * seg_width
        y_mid = (y_lo + y_hi) / 2

        # 在段内拟合 f(y) = 1 - cos(π/4 · y)
        # 用 3 个点做 2 阶插值
        pts = [y_lo, y_mid, y_hi]
        vals = [1.0 - math.cos(math.pi / 4 * p) for p in pts]

        # δ = y - y_lo
        deltas = [p - y_lo for p in pts]

        # 解 Vandermonde: vals = c0 + c1*δ + c2*δ²
        V = np.array([[1, d, d * d] for d in deltas])
        coeffs = np.linalg.solve(V, vals)

        c0[i] = coeffs[0]
        c1[i] = coeffs[1]
        c2[i] = coeffs[2]

    return HOTBMTable(
        num_segments=num_seg,
        addr_bits=addr_bits,
        coeff_bits=coeff_bits,
        c0_table=c0,
        c1_table=c1,
        c2_table=c2,
        seg_width=seg_width,
    )


def build_sin_table(cfg: HWConfig, addr_bits: int = 8) -> HOTBMTable:
    """
    构建 sin 通路查找表。
    目标函数: f_sin(y) = π/4 - sin(π/4 · y)/y，y ∈ (0, 0.5]
    (y=0 时极限为 0)

    使用 2 阶多项式 fit 每个分段。
    """
    num_seg = 1 << addr_bits
    seg_width = 0.5 / num_seg
    coeff_bits = cfg.w_F + cfg.g + 4

    c0 = np.zeros(num_seg, dtype=np.float64)
    c1 = np.zeros(num_seg, dtype=np.float64)
    c2 = np.zeros(num_seg, dtype=np.float64)

    def f_sin_over_y(y):
        """π/4 - sin(π/4·y)/y, 带 y→0 极限处理"""
        if abs(y) < 1e-30:
            return 0.0  # π/4 - π/4 = 0
        return math.pi / 4 - math.sin(math.pi / 4 * y) / y

    for i in range(num_seg):
        y_lo = i * seg_width
        y_hi = (i + 1) * seg_width
        y_mid = (y_lo + y_hi) / 2

        # 避免 y_lo = 0 的除零问题
        eps = 1e-20
        pts = [max(y_lo, eps), y_mid if y_mid > eps else eps, y_hi]
        vals = [f_sin_over_y(p) for p in pts]
        deltas = [p - max(y_lo, eps) + (max(y_lo, eps) - y_lo) for p in pts]
        # 更简洁：δ 相对 y_lo
        deltas = [p - y_lo for p in pts]

        V = np.array([[1, d, d * d] for d in deltas])
        try:
            coeffs = np.linalg.solve(V, vals)
        except np.linalg.LinAlgError:
            coeffs = [vals[0], 0.0, 0.0]

        c0[i] = coeffs[0]
        c1[i] = coeffs[1]
        c2[i] = coeffs[2]

    return HOTBMTable(
        num_segments=num_seg,
        addr_bits=addr_bits,
        coeff_bits=coeff_bits,
        c0_table=c0,
        c1_table=c1,
        c2_table=c2,
        seg_width=seg_width,
    )


# ─────────────────────────────────────────────
# HOTBM 表查找 + 插值
# ─────────────────────────────────────────────

def table_lookup(table: HOTBMTable, y_abs: float, coeff_frac_bits: int) -> float:
    """
    模拟硬件查表 + 二阶插值。

    1. 用 y_abs 的高位做地址查表
    2. 用低位做段内偏移 δ
    3. 计算 c0 + c1·δ + c2·δ²（量化系数做定点乘法）

    Args:
        table: HOTBM 查找表
        y_abs: |y| ∈ [0, 0.5]
        coeff_frac_bits: 系数定点量化的小数位宽

    Returns:
        表查找结果 (float, 模拟定点计算)
    """
    # 限幅
    y_abs = max(0.0, min(y_abs, 0.5 - 1e-15))

    # 段地址
    seg_idx = int(y_abs / table.seg_width)
    seg_idx = min(seg_idx, table.num_segments - 1)

    # 段内偏移 δ
    delta = y_abs - seg_idx * table.seg_width

    # 量化系数 (模拟定点存储)
    c0_q = float_to_fixed(table.c0_table[seg_idx], coeff_frac_bits)
    c1_q = float_to_fixed(table.c1_table[seg_idx], coeff_frac_bits)
    c2_q = float_to_fixed(table.c2_table[seg_idx], coeff_frac_bits)

    # 量化 δ
    delta_q = float_to_fixed(delta, coeff_frac_bits)

    # 定点计算 c0 + c1·δ + c2·δ² (Horner: c0 + δ·(c1 + c2·δ))
    # 先计算 c2·δ (结果右移 coeff_frac_bits 位对齐)
    c2_delta = (c2_q * delta_q) >> coeff_frac_bits
    # c1 + c2·δ
    inner = c1_q + c2_delta
    # δ·(c1 + c2·δ)
    outer = (inner * delta_q) >> coeff_frac_bits
    # c0 + δ·(c1 + c2·δ)
    result = c0_q + outer

    return fixed_to_float(result, coeff_frac_bits)


# ─────────────────────────────────────────────
# 两条通路的顶层函数
# ─────────────────────────────────────────────

class SinCosEvalResult:
    """sin/cos 求值结果"""
    __slots__ = [
        'sin_val',   # sin(π/4 · y) 的值 (浮点)
        'cos_val',   # cos(π/4 · y) 的值 (浮点)
        'sin_sign',  # sin 的符号 (来自 y 的符号)
        'cos_sign',  # cos 的符号 (恒正, 因为 cos 在此区间 > 0)
    ]

    def __init__(self):
        self.sin_val = 0.0
        self.cos_val = 1.0
        self.sin_sign = 0
        self.cos_sign = 0


# 全局表缓存
_cos_table_cache = {}
_sin_table_cache = {}


def _get_tables(cfg: HWConfig, addr_bits: int = 8):
    """获取或构建表（带缓存）"""
    key = (cfg.w_F, cfg.g, cfg.g_K, addr_bits)
    if key not in _cos_table_cache:
        _cos_table_cache[key] = build_cos_table(cfg, addr_bits)
        _sin_table_cache[key] = build_sin_table(cfg, addr_bits)
    return _cos_table_cache[key], _sin_table_cache[key]


def evaluate_sincos(Y_fix: int, E_y: int, M_y: int, y_sign: int,
                    cfg: HWConfig = DEFAULT_CONFIG,
                    method: str = "table",
                    addr_bits: int = 8) -> SinCosEvalResult:
    """
    计算 sin(π/4 · y) 和 cos(π/4 · y)。

    Args:
        Y_fix:  |y| 的定点表示 (w_F+g 位, 纯小数)
        E_y:    |y| 的浮点指数 (unbiased)
        M_y:    |y| 的浮点尾数 (w_F+1 位整数, 含隐含位)
        y_sign: y 的符号 (0=正, 1=负)
        cfg:    硬件配置
        method: "ideal" 用 math 库直接算, "table" 用查表插值
        addr_bits: 查找表地址位宽

    Returns:
        SinCosEvalResult
    """
    result = SinCosEvalResult()
    result.sin_sign = y_sign

    w_F = cfg.w_F
    g = cfg.g

    # 将定点 Y 转换为浮点值用于查表
    y_fix_bits = cfg.y_fixedpoint_bits  # w_F + g
    y_abs_from_fix = Y_fix / (1 << y_fix_bits)

    # 将浮点 (E_y, M_y) 转换为浮点值
    if M_y == 0:
        y_abs_from_float = 0.0
    else:
        y_abs_from_float = (M_y / (1 << w_F)) * (2.0 ** E_y)

    if method == "ideal":
        # ── 理想模式：直接用 math 库 ──
        # 两条通路使用同一个 y 参数，保证 sin(φ) 和 cos(φ) 的一致性
        # 优先使用浮点表示（精度更好，尤其当 y 接近 0 时）
        if y_abs_from_float > 0:
            y_eval = y_abs_from_float
        else:
            y_eval = y_abs_from_fix
        cos_val = math.cos(math.pi / 4 * y_eval)
        sin_val = math.sin(math.pi / 4 * y_eval)

        result.cos_val = cos_val
        result.sin_val = sin_val

    elif method == "table":
        # ── 查表模式：模拟 HOTBM ──
        cos_table, sin_table = _get_tables(cfg, addr_bits)
        coeff_frac_bits = w_F + g + 4

        # ── cos 通路 (纯定点) ──
        # HOTBM 计算 f_cos(|y|) = 1 - cos(π/4 · |y|)
        f_cos = table_lookup(cos_table, y_abs_from_fix, coeff_frac_bits)

        # cos(π/4 · y) = 1 - f_cos
        # 定点减法 (量化到 w_F+g 位)
        one_fixed = 1 << (w_F + g)
        f_cos_fixed = float_to_fixed(f_cos, w_F + g)
        cos_fixed = one_fixed - f_cos_fixed
        result.cos_val = fixed_to_float(cos_fixed, w_F + g)

        # ── sin 通路 (定点 × 浮点) ──
        # HOTBM 计算 f_sin(|y|) = π/4 - sin(π/4·|y|)/|y|
        f_sin = table_lookup(sin_table, y_abs_from_fix, coeff_frac_bits)

        # sin(π/4·y) = y × (π/4 - f_sin)
        # = M_y × 2^E_y × (π/4 - f_sin)
        pi_over_4 = math.pi / 4
        sinc_val = pi_over_4 - f_sin  # sin(π/4·y)/y 的近似

        # 定点 × 浮点乘法
        # sinc_val 量化为 (w_F+g) 位定点
        sinc_fixed = float_to_fixed(sinc_val, w_F + g)
        sinc_float = fixed_to_float(sinc_fixed, w_F + g)

        # sin = |y| × sinc_val
        if M_y == 0:
            result.sin_val = 0.0
        else:
            # 浮点乘: M_y/2^w_F × 2^E_y × sinc_float
            m_float = M_y / (1 << w_F)  # 1.xxx 格式
            result.sin_val = m_float * (2.0 ** E_y) * sinc_float

    return result
